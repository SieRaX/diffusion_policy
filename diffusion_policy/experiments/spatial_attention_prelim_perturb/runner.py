"""Perturbation-sensitivity runner: load a flow-matching checkpoint, DERIVE the
task/obs/dataset/dims from its embedded training cfg, build a single env, and walk
one demonstration episode measuring the coupled endpoint distance under grasp-aware
SE(3) perturbations. Saves a self-describing npz.
"""
import os
import glob
import json
import hashlib

import dill
import h5py
import hydra
import numpy as np
import torch
from omegaconf import OmegaConf, open_dict

from diffusion_policy.experiments.spatial_attention_prelim_perturb.metric.crn import CRNManager
from diffusion_policy.experiments.spatial_attention_prelim_perturb.metric.endpoint_distance import (
    CoupledEndpointDistance,
)
from diffusion_policy.experiments.spatial_attention_prelim_perturb.perturbation import bodies as bodies_mod
from diffusion_policy.experiments.spatial_attention_prelim_perturb.perturbation.backend import make_backend
from diffusion_policy.experiments.spatial_attention_prelim_perturb.obs_builder import obs_builder

_POLICY_TARGETS = {
    'lowdim': 'diffusion_policy.experiments.spatial_attention_prelim_perturb.policy.'
              'perturb_sensitivity_policy.PerturbSensitivityLowdimPolicy',
    'image': 'diffusion_policy.experiments.spatial_attention_prelim_perturb.policy.'
             'perturb_sensitivity_policy.PerturbSensitivityHybridImagePolicy',
}


def _require(node, key, ctx):
    if key not in node:
        raise KeyError(f"Checkpoint cfg is missing required field '{ctx}.{key}'.")
    return node[key]


def _resolve_from_checkpoint(cfg, dataset_path_override):
    """Everything derived from the checkpoint's embedded training cfg."""
    task = _require(cfg, 'task', 'cfg')
    task_name = str(_require(task, 'name', 'cfg.task'))
    variant = 'image' if 'shape_meta' in task else 'lowdim'
    n_obs_steps = int(_require(cfg, 'n_obs_steps', 'cfg'))
    horizon = int(_require(cfg, 'horizon', 'cfg'))
    abs_action = bool(task.get('abs_action', False))

    # dataset path (override wins; else read defensively)
    if dataset_path_override:
        dataset_path = dataset_path_override
    elif 'dataset' in task and 'dataset_path' in task.dataset:
        dataset_path = str(task.dataset.dataset_path)
    elif 'dataset_path' in task:
        dataset_path = str(task.dataset_path)
    else:
        raise KeyError("No dataset path in cfg.task.dataset.dataset_path; pass "
                       "dataset_path_override.")

    obs_keys = list(task.obs_keys) if (variant == 'lowdim' and 'obs_keys' in task) else None
    shape_meta = OmegaConf.to_container(task.shape_meta, resolve=True) if variant == 'image' else None

    pol_target = str(_require(cfg, 'policy', 'cfg')['_target_'])
    if 'flow_matching' not in pol_target.lower():
        raise ValueError(f"Checkpoint policy is not flow-matching ({pol_target}); this "
                         "experiment assumes a deterministic FM sampler.")
    return dict(task_name=task_name, variant=variant, n_obs_steps=n_obs_steps,
                horizon=horizon, abs_action=abs_action, dataset_path=dataset_path,
                obs_keys=obs_keys, shape_meta=shape_meta)


def _load_policy(cfg, payload, variant, device, output_dir):
    """Reconstruct the policy as the injectable, normalized-exposing subclass and
    load the checkpoint weights (identical params) into it."""
    with open_dict(cfg):
        cfg.policy._target_ = _POLICY_TARGETS[variant]
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, output_dir=output_dir)
    workspace.load_payload(payload)
    policy = workspace.ema_model if cfg.training.use_ema else workspace.model
    assert getattr(policy, 'obs_as_global_cond', True), \
        "obs_as_global_cond must be True so the entire ODE init noise is CRN-controlled."
    policy.to(device)
    policy.eval()
    return policy


def run(cfg):
    device = cfg.device
    ckpt = cfg.checkpoint
    payload = torch.load(open(ckpt, 'rb'), pickle_module=dill, map_location='cpu')
    train_cfg = payload['cfg']
    r = _resolve_from_checkpoint(train_cfg, cfg.get('dataset_path_override', None))

    # output_dir is required (config `output_dir: ???`) and is the Hydra run dir
    # (hydra.run.dir=${output_dir}), so Hydra's .hydra snapshot and the results below
    # land in the same directory. The caller owns the layout (e.g. the launch scripts
    # build <ckpt>/prelim_perturb_<task>_<abs|rel>/<backend>/episode_<demo>).
    output_dir = str(cfg.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    policy = _load_policy(train_cfg, payload, r['variant'], device, output_dir)
    Da = int(policy.action_dim)
    H = int(policy.horizon)
    To = r['n_obs_steps']

    # --- env + demo states ---
    wrapper = obs_builder.build_env(
        r['dataset_path'], r['variant'], obs_keys=r['obs_keys'],
        shape_meta=r['shape_meta'], abs_action=r['abs_action'])
    rs_env, sim = wrapper.env.env, wrapper.env.env.sim
    with h5py.File(r['dataset_path'], 'r') as f:
        states = f[f"data/demo_{int(cfg.demo_index)}/states"][:]
    ep_len = len(states)

    # --- perturbable bodies (checkpoint-derived task) ---
    task_base = bodies_mod.task_base_from_name(train_cfg.task)
    override = None
    if cfg.get('object_bodies', None) is not None:
        override = list(cfg.object_bodies.get(task_base, [])) or None
    bodies = bodies_mod.resolve_perturb_bodies(rs_env, sim, task_base, object_names_override=override)
    body_names = [b.name for b in bodies]

    # --- metric + CRN (shared; the {eps0_j} seed is what makes runs overlayable) ---
    N = int(cfg.N)
    M = int(cfg.M) if cfg.get('M', None) is not None else int(policy.num_inference_steps)
    crn = CRNManager(k_s=N, horizon=H, action_dim=Da, eps0_seed=int(cfg.seeds.crn))
    metric = CoupledEndpointDistance(crn, ode_steps=M,
                                     distance_space=cfg.distance_space,
                                     max_batch=int(cfg.max_batch))

    # --- perturbation backend (sim_state | obs_noise), selected purely by config ---
    K = int(cfg.K)
    backend = make_backend(cfg, r, policy, wrapper, bodies, rs_env)

    # --- episode loop ---
    ts = list(range(0, ep_len, int(cfg.stride)))
    spaces = metric.spaces
    acc = {sp: {'S': [], 'S_first': [], 'per_index': [], 'D_k': [], 'control': []} for sp in spaces}
    grasp_rec = []

    for step_i, t in enumerate(ts):
        # grasp flags come from the NOMINAL sim state under BOTH backends (figure
        # shading only; the obs_noise backend never uses them to perturb).
        grasp_flags = obs_builder.detect_grasp_flags(wrapper, states, t, bodies, float(cfg.grasp_qpos_threshold))
        grasp_rec.append(grasp_flags)

        nominal_obs = backend.build_nominal(wrapper, states, t)
        perturbed = backend.build_perturbed(wrapper, states, t, K)

        do_control = (step_i % int(cfg.control_stride) == 0)
        res = metric.compute(policy, nominal_obs, perturbed, compute_control=do_control)
        for sp in spaces:
            acc[sp]['S'].append(res[sp]['S'])
            acc[sp]['S_first'].append(res[sp]['S_first'])
            acc[sp]['per_index'].append(res[sp]['per_index'])
            acc[sp]['D_k'].append(res[sp]['D_k'])
            acc[sp]['control'].append(res[sp].get('control', np.nan))
        print(f"[perturb] t={t} ({step_i+1}/{len(ts)}) "
              + " ".join(f"S_{sp}={res[sp]['S']:.3e}" for sp in spaces))

    # --- save ---
    resolved = OmegaConf.to_container(cfg, resolve=True)
    resolved.update({'_resolved_task': r['task_name'], '_resolved_variant': r['variant'],
                     '_resolved_dataset': r['dataset_path'], '_M': M})
    cfg_hash = hashlib.sha1(json.dumps(resolved, sort_keys=True, default=str).encode()).hexdigest()[:12]

    save = dict(
        timesteps=np.asarray(ts, dtype=np.int64),
        episode_length=np.int64(ep_len),
        task_name=str(r['task_name']), obs_variant=str(r['variant']),
        abs_action=bool(r['abs_action']), demo_index=np.int64(cfg.demo_index),
        body_names=np.asarray(body_names, dtype=object),
        grasp_flags=np.asarray(grasp_rec, dtype=bool),
        distance_spaces=np.asarray(list(spaces), dtype=object),
        executed_start=np.int64(To - 1), horizon=np.int64(H), action_dim=np.int64(Da),
        K=np.int64(K), N=np.int64(N), M=np.int64(M),
        history_mode=str(cfg.history_mode), perturbation_backend=str(backend.name),
        seed_perturb=np.int64(cfg.seeds.perturb), seed_crn=np.int64(cfg.seeds.crn),
        control_stride=np.int64(cfg.control_stride), config_hash=str(cfg_hash),
    )
    save.update(backend.metadata())  # sim_state: perturb_targets + sigmas; obs_noise: noise params + std cache
    for sp in spaces:
        save[f'S_{sp}'] = np.asarray(acc[sp]['S'], dtype=np.float64)
        save[f'S_first_{sp}'] = np.asarray(acc[sp]['S_first'], dtype=np.float64)
        save[f'per_index_{sp}'] = np.asarray(acc[sp]['per_index'], dtype=np.float64)  # (T,H)
        save[f'D_k_{sp}'] = np.asarray(acc[sp]['D_k'], dtype=np.float64)              # (T,K)
        save[f'control_{sp}'] = np.asarray(acc[sp]['control'], dtype=np.float64)      # (T,)

    out_path = os.path.join(output_dir, 'perturb.npz')
    np.savez(out_path, **save)
    print(f"[perturb] wrote {out_path}  (T={len(ts)} timesteps, spaces={list(spaces)})")
    return out_path
