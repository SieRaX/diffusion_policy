"""Experiment-2 runner: load an FM checkpoint, build the full-dataset similarity index
(probe pool), evaluate S_vel on all query states and S_end on a stratified subset, save a
self-describing npz. Reuses the prelim's checkpoint resolution / policy loading / obs
sourcing / coupled-endpoint metric and Exp1's CRN.

No policy rollout; inference is done directly on dataset states via env.reset_to.
"""
import os
import json
import hashlib

import dill
import numpy as np
import torch
from omegaconf import OmegaConf

from diffusion_policy.experiments.spatial_attention_prelim_perturb.runner import (
    _resolve_from_checkpoint, _load_policy,
)
from diffusion_policy.experiments.spatial_attention_prelim_perturb.perturbation import bodies as bodies_mod
from diffusion_policy.experiments.spatial_attention_exp1.mse_metric.crn import CRNManager
from diffusion_policy.experiments.spatial_attention_prelim_perturb.metric.endpoint_distance import (
    CoupledEndpointDistance,
)
from diffusion_policy.experiments.spatial_attention_exp2 import obs_source as obs_src
from diffusion_policy.experiments.spatial_attention_exp2.similarity import spaces as sim_spaces
from diffusion_policy.experiments.spatial_attention_exp2.similarity.build import (
    build_or_load_index, index_cache_path,
)
from diffusion_policy.experiments.spatial_attention_exp2.probes.base import make_probe_type
from diffusion_policy.experiments.spatial_attention_exp2.probes.fallback import apply_fallback
from diffusion_policy.experiments.spatial_attention_exp2.estimators.velocity import VelocitySensitivity
from diffusion_policy.experiments.spatial_attention_exp2.estimators.endpoint import (
    EndpointSensitivity, stratified_subset,
)
from diffusion_policy.experiments.spatial_attention_exp2.segmentation import phases as seg
from diffusion_policy.experiments.spatial_attention_exp2.util import validate_timeline_episode


def _directions(name):
    if name == 'both':
        return ['temporal', 'config']
    if name in ('temporal', 'config'):
        return [name]
    raise ValueError(f"directions must be temporal|config|both, got {name!r}")


def _probe_params(cfg):
    p = cfg.probe_type      # the probe_type group (neighbor + noise params)
    return dict(
        K=int(cfg.K), max_radius=float(p.max_radius), min_neighbors=int(p.min_neighbors),
        same_episode_window=int(p.same_episode_window),
        project_out_temporal=bool(p.project_out_temporal),
        exclude_same_episode=bool(p.exclude_same_episode),
        rank=int(p.get('rank', 16)), seed_base=int(cfg.seeds.crn),
        fallback=str(cfg.fallback))


def _materialize(plan, nominal_obs, obs_source):
    """ProbePlan -> list of predict_action-ready probe obs_dicts (batch 1)."""
    if plan.kind == 'real':
        return [obs_source.obs_at(ep, t) for (ep, t) in plan.targets]
    if plan.kind == 'noise':
        out = []
        for k in range(plan.K):
            delta = torch.as_tensor(plan.deltas[k], dtype=nominal_obs['obs'].dtype)
            out.append({'obs': nominal_obs['obs'] + delta})   # add to every history frame
        return out
    return []


def run(cfg):
    device = cfg.device
    ckpt = cfg.checkpoint
    payload = torch.load(open(ckpt, 'rb'), pickle_module=dill, map_location='cpu')
    train_cfg = payload['cfg']
    r = _resolve_from_checkpoint(train_cfg, cfg.get('dataset_path_override', None))

    output_dir = str(cfg.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    policy = _load_policy(train_cfg, payload, r['variant'], device, output_dir)
    Da, H, To = int(policy.action_dim), int(policy.horizon), r['n_obs_steps']
    variant, shape_meta = r['variant'], r['shape_meta']

    # --- state sourcing (env obs) + normalized a_GT actions ---
    obs_source = obs_src.ObsSource(r['dataset_path'], variant, r['obs_keys'], shape_meta,
                                   r['abs_action'], cfg.history_mode, To)
    gt = obs_src.build_gt_actions(train_cfg, cfg.get('dataset_path_override', None),
                                  To, H, policy.normalizer['action'])

    # --- probe pool (ALL demos) + query set (eval_episodes) ---
    pool, n_demos = obs_src.enumerate_dataset(r['dataset_path'], int(cfg.stride))
    pool_eps = np.array([e for e, _ in pool], np.int64)
    pool_ts = np.array([t for _, t in pool], np.int64)
    eval_episodes = [int(e) for e in cfg.eval_episodes]
    assert all(0 <= e < n_demos for e in eval_episodes), \
        f"eval_episodes {eval_episodes} out of range [0,{n_demos})"
    validate_timeline_episode(cfg.figures.timeline_episode, eval_episodes)  # fail fast on bad figure cfg

    # --- similarity index over the FULL dataset (cached; reused across ablation cells) ---
    scfg = cfg.similarity_space
    enc = str(scfg.get('encoder', 'na')) if variant == 'image' else 'state'
    metric = str(scfg.get('metric', 'euclidean'))
    cache = index_cache_path(ckpt, str(scfg.name), enc, metric, int(cfg.stride))
    index, reused = build_or_load_index(
        cache, pool_eps, pool_ts,
        vectors_fn=lambda: sim_spaces.build_vectors(obs_source, policy, pool, variant,
                                                    scfg, shape_meta)[0],
        metric=metric, meta={'checkpoint': ckpt, 'space': str(scfg.name), 'stride': int(cfg.stride)})
    assert index.dim > 0, "empty similarity space"

    query_rows = index.query_rows(eval_episodes)
    query_ep = index.episodes[query_rows]
    query_t = index.timesteps[query_rows]
    Q = len(query_rows)

    # --- phases: query (env grasp) always; pool (gripper proxy) only if dtw fallback ---
    task_base = bodies_mod.task_base_from_name(train_cfg.task)
    bodies = bodies_mod.resolve_perturb_bodies(
        obs_source.wrapper.env.env, obs_source.wrapper.env.env.sim, task_base,
        object_names_override=None)
    query_phase = _query_phases(obs_source, gt, bodies, eval_episodes, index, query_rows,
                                float(cfg.grasp_qpos_threshold))
    pool_phases = _pool_phases(gt, index) if str(cfg.fallback) == 'dtw' else None

    # --- estimators ---
    include_gripper = bool(cfg.get('include_gripper', True))
    action_dims = None if include_gripper else list(range(Da - 1))
    N = int(cfg.N)
    M = int(cfg.M) if cfg.get('M', None) is not None else int(policy.num_inference_steps)
    crn_vel = CRNManager(k_s=1, horizon=H, action_dim=Da,
                         num_fm=int(cfg.num_tz_draws), fm_seed=int(cfg.seeds.tz_draws))
    crn_end = CRNManager(k_s=N, horizon=H, action_dim=Da, eps0_seed=int(cfg.seeds.crn))
    vel = VelocitySensitivity(crn_vel, action_dims=action_dims, max_batch=int(cfg.max_batch))
    end_metric = CoupledEndpointDistance(crn_end, ode_steps=M, distance_space=cfg.distance_space,
                                         max_batch=int(cfg.max_batch), action_dims=action_dims)
    endpoint = EndpointSensitivity(end_metric)
    end_spaces = list(end_metric.spaces)

    dirs = _directions(str(cfg.directions))
    both_w = [float(w) for w in cfg.both_weights]
    probe = make_probe_type(str(cfg.probe_type.name))
    params = _probe_params(cfg)
    ctx = {'pool_phases': pool_phases}

    # --- S_vel over ALL query states ---
    svel = {d: {'scalar': np.full(Q, np.nan), 'per_index': np.full((Q, H), np.nan),
                'first': np.full(Q, np.nan)} for d in dirs}
    flagged = np.zeros(Q, bool)
    weights = np.ones(Q, np.float64)
    plans_cache = [dict() for _ in range(Q)]     # reused for S_end (same probes)
    probes_cache = [dict() for _ in range(Q)]
    for qi in range(Q):
        row = int(query_rows[qi])
        nominal = obs_source.obs_at(int(query_ep[qi]), int(query_t[qi]))
        a_gt = gt.norm_chunk(int(query_ep[qi]), int(query_t[qi]))
        for d in dirs:
            plan = apply_fallback(probe.plan(row, index, d, params), row, index, params, ctx)
            plans_cache[qi][d] = plan
            if plan.flagged:
                flagged[qi] = True
                weights[qi] = plan.weight
            if plan.kind == 'invalid' or plan.K == 0:
                continue
            probes = _materialize(plan, nominal, obs_source)
            probes_cache[qi][d] = (nominal, probes, a_gt, plan.denom)
            res = vel.compute(policy, nominal, probes, a_gt, plan.denom)
            svel[d]['scalar'][qi] = res['scalar']
            svel[d]['per_index'][qi] = res['per_index']
            svel[d]['first'][qi] = res['first']
        print(f"[exp2:svel] {qi+1}/{Q} ep{int(query_ep[qi])} t{int(query_t[qi])} "
              + " ".join(f"{d}={svel[d]['scalar'][qi]:.3e}" for d in dirs))

    svel_c = _combine(svel, dirs, both_w, H, Q)   # direction-combined S_vel map

    # --- S_end on a stratified subset (reusing the same probes) ---
    subset = stratified_subset(Q, np.nan_to_num(svel_c['scalar']), query_phase,
                               float(cfg.endpoint_subset_frac), int(cfg.endpoint_subset_min),
                               seed=int(cfg.seeds.crn))
    send = {sp: {d: {'scalar': np.full(Q, np.nan), 'per_index': np.full((Q, H), np.nan),
                     'first': np.full(Q, np.nan)} for d in dirs} for sp in end_spaces}
    for qi in subset:
        for d in dirs:
            cached = probes_cache[int(qi)].get(d, None)
            if cached is None:
                continue
            nominal, probes, _, denom = cached
            res = endpoint.compute(policy, nominal, probes, denom)
            for sp in end_spaces:
                send[sp][d]['scalar'][int(qi)] = res[sp]['S']
                send[sp][d]['per_index'][int(qi)] = res[sp]['per_index']
                send[sp][d]['first'][int(qi)] = res[sp]['S_first']
    send_c = {sp: _combine(send[sp], dirs, both_w, H, Q) for sp in end_spaces}

    # --- save ---
    _save(cfg, r, output_dir, ckpt, dirs, end_spaces, include_gripper, N, M, To, H, Da,
          pool_eps, pool_ts, query_ep, query_t, query_phase, flagged, weights, reused,
          svel, svel_c, send, send_c, subset, n_demos)
    return os.path.join(output_dir, 'exp2.npz')


def _combine(maps, dirs, both_w, H, Q):
    if len(dirs) == 1:
        return maps[dirs[0]]
    wt, wc = both_w[0], both_w[1]
    out = {}
    for key, shape in (('scalar', (Q,)), ('per_index', (Q, H)), ('first', (Q,))):
        out[key] = wt * maps['temporal'][key] + wc * maps['config'][key]
    return out


def _query_phases(obs_source, gt, bodies, eval_episodes, index, query_rows, thr):
    """Phase id per query state (env grasp detection for accuracy)."""
    per_ep = {}
    for e in set(int(x) for x in eval_episodes):
        st = obs_source.states(e)
        grasped = seg.grasped_from_env(obs_source, bodies, e, st, thr)
        gopen = seg.gripper_open_from_actions(gt.actions[gt.ep_starts[e]:gt.episode_ends[e]])
        L = min(len(grasped), len(gopen))
        per_ep[e] = seg.segment_episode(gopen[:L], grasped[:L])
    out = np.zeros(len(query_rows), np.int64)
    for i, row in enumerate(query_rows):
        e, t = int(index.episodes[row]), int(index.timesteps[row])
        lab = per_ep[e]
        out[i] = lab[min(t, len(lab) - 1)]
    return out


def _pool_phases(gt, index):
    """Cheap gripper-proxy phase id per pool row (for the dtw fallback phase-match)."""
    per_ep = {}
    for e in np.unique(index.episodes):
        acts = gt.actions[gt.ep_starts[int(e)]:gt.episode_ends[int(e)]]
        per_ep[int(e)] = seg.segment_episode(seg.gripper_open_from_actions(acts),
                                             seg.grasped_from_gripper(acts))
    out = np.zeros(len(index.episodes), np.int64)
    for i in range(len(index.episodes)):
        lab = per_ep[int(index.episodes[i])]
        out[i] = lab[min(int(index.timesteps[i]), len(lab) - 1)]
    return out


def _save(cfg, r, output_dir, ckpt, dirs, end_spaces, include_gripper, N, M, To, H, Da,
          pool_eps, pool_ts, query_ep, query_t, query_phase, flagged, weights, reused,
          svel, svel_c, send, send_c, subset, n_demos):
    resolved = OmegaConf.to_container(cfg, resolve=True)
    resolved.update({'_task': r['task_name'], '_variant': r['variant'], '_M': M,
                     '_dataset': r['dataset_path']})
    cfg_hash = hashlib.sha1(json.dumps(resolved, sort_keys=True, default=str).encode()).hexdigest()[:12]
    save = dict(
        task_name=str(r['task_name']), obs_variant=str(r['variant']),
        abs_action=bool(r['abs_action']), checkpoint=str(ckpt), config_hash=str(cfg_hash),
        probe_type=str(cfg.probe_type.name), similarity_space=str(cfg.similarity_space.name),
        directions=str(cfg.directions), both_weights=np.asarray(cfg.both_weights, float),
        include_gripper=bool(include_gripper), distance_spaces=np.asarray(end_spaces, object),
        executed_start=np.int64(To - 1), horizon=np.int64(H), action_dim=np.int64(Da),
        K=np.int64(cfg.K), N=np.int64(N), M=np.int64(M), num_tz_draws=np.int64(cfg.num_tz_draws),
        stride=np.int64(cfg.stride), n_demos=np.int64(n_demos),
        eval_episodes=np.asarray(list(cfg.eval_episodes), np.int64),
        pool_episodes=pool_eps, pool_timesteps=pool_ts,
        query_episodes=np.asarray(query_ep, np.int64), query_timesteps=np.asarray(query_t, np.int64),
        query_phase=np.asarray(query_phase, np.int64), phase_names=np.asarray(seg.PHASES, object),
        flagged=np.asarray(flagged, bool), flagged_fraction=np.float64(np.mean(flagged) if len(flagged) else 0.0),
        weights=np.asarray(weights, float), index_reused=bool(reused),
        endpoint_subset=np.asarray(subset, np.int64),
        seed_tz=np.int64(cfg.seeds.tz_draws), seed_crn=np.int64(cfg.seeds.crn),
        fallback=str(cfg.fallback), timeline_episode=np.int64(cfg.figures.timeline_episode),
        Svel_scalar=svel_c['scalar'], Svel_per_index=svel_c['per_index'], Svel_first=svel_c['first'],
    )
    for d in dirs:
        save[f'Svel_scalar_{d}'] = svel[d]['scalar']
        save[f'Svel_per_index_{d}'] = svel[d]['per_index']
        save[f'Svel_first_{d}'] = svel[d]['first']
    for sp in end_spaces:
        save[f'Send_scalar_{sp}'] = send_c[sp]['scalar']
        save[f'Send_per_index_{sp}'] = send_c[sp]['per_index']
        save[f'Send_first_{sp}'] = send_c[sp]['first']
        for d in dirs:
            save[f'Send_scalar_{sp}_{d}'] = send[sp][d]['scalar']
    out = os.path.join(output_dir, 'exp2.npz')
    np.savez(out, **save)
    print(f"[exp2] wrote {out}  (Q={len(query_ep)} query states, S_end on {len(subset)}, "
          f"flagged {np.mean(flagged) if len(flagged) else 0:.1%}, index_reused={reused})")
