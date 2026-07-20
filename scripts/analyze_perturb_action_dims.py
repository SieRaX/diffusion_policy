"""Per-action-dimension decomposition of the obs-noise endpoint distance.

The saved perturb.npz sums the squared chunk difference over action dims, so it can't
say WHICH action dimension drives S(t). This script recomputes the metric with the exact
same pipeline (policy loading, ObsNoiseBackend, CRN, CoupledEndpointDistance._sample) but
reduces per action dimension:

    contrib_d(t) = mean_k  mean_N  sum_H  ( f_d(eps0, o'_k) - f_d(eps0, o) )^2

so sum_d contrib_d(t) == S(t). Reveals e.g. whether the gripper dim dominates.

Abs-action layout (D=10): position = dims 0-2, rot6d = dims 3-8, gripper = dim 9.

Usage (GPU):
    MUJOCO_GL=osmesa python scripts/analyze_perturb_action_dims.py \
        --checkpoint <ckpt> --output_dir <dir> --device cuda:0 --stride 3
"""
import argparse
import multiprocessing as mp
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root

import dill
import h5py
import numpy as np
import torch
from omegaconf import OmegaConf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

mp.set_start_method("spawn", force=True)
OmegaConf.register_new_resolver("eval", eval, replace=True)

from diffusion_policy.experiments.spatial_attention_prelim_perturb import runner
from diffusion_policy.experiments.spatial_attention_prelim_perturb.obs_builder import obs_builder
from diffusion_policy.experiments.spatial_attention_prelim_perturb.perturbation.backend import ObsNoiseBackend
from diffusion_policy.experiments.spatial_attention_prelim_perturb.metric.crn import CRNManager
from diffusion_policy.experiments.spatial_attention_prelim_perturb.metric.endpoint_distance import (
    CoupledEndpointDistance,
)

GROUPS = [('position', [0, 1, 2]), ('rot6d', [3, 4, 5, 6, 7, 8]), ('gripper', [9])]


def compute(cfg):
    payload = torch.load(open(cfg.checkpoint, 'rb'), pickle_module=dill, map_location='cpu')
    train_cfg = payload['cfg']
    r = runner._resolve_from_checkpoint(train_cfg, None)
    assert r['variant'] == 'lowdim', "per-dim obs_noise decomposition is low_dim only"
    policy = runner._load_policy(train_cfg, payload, r['variant'], cfg.device, cfg.output_dir)
    Da, H, To = int(policy.action_dim), int(policy.horizon), r['n_obs_steps']

    wrapper = obs_builder.build_env(r['dataset_path'], r['variant'], obs_keys=r['obs_keys'],
                                    shape_meta=r['shape_meta'], abs_action=r['abs_action'])
    with h5py.File(r['dataset_path'], 'r') as f:
        states = f[f"data/demo_{int(cfg.demo)}/states"][:]
    ep_len = len(states)

    obs_std = policy.normalizer['obs'].get_input_stats()['std'].detach().cpu().numpy().reshape(-1)
    wrapper.env.reset_to({'states': states[0]})
    key_ranges = obs_builder.lowdim_obs_key_ranges(wrapper)
    backend = ObsNoiseBackend('lowdim', To, 'tile_perturbed', obs_dim=len(obs_std),
                              obs_std=obs_std, obs_key_ranges=key_ranges, K=int(cfg.K),
                              noise_mode='per_dim_std', noise_scale=float(cfg.noise_scale),
                              sigma_abs=0.01, noise_dim_mask=None, seed=int(cfg.seed))
    crn = CRNManager(k_s=int(cfg.N), horizon=H, action_dim=Da, eps0_seed=int(cfg.seed_crn))
    M = int(policy.num_inference_steps)
    metric = CoupledEndpointDistance(crn, ode_steps=M, distance_space='both', max_batch=int(cfg.max_batch))

    ts = list(range(0, ep_len, int(cfg.stride)))
    pd_norm = np.zeros((len(ts), Da)); pd_raw = np.zeros((len(ts), Da))
    for i, t in enumerate(ts):
        nom = backend.build_nominal(wrapper, states, t)
        pert = backend.build_perturbed(wrapper, states, t, int(cfg.K))
        sn = metric._sample(policy, nom)                       # {'raw','norm'} (N,H,D) cpu f64
        acc = {'raw': np.zeros(Da), 'norm': np.zeros(Da)}
        for o in pert:
            sp = metric._sample(policy, o)
            for space in ('raw', 'norm'):
                diff2 = (sp[space] - sn[space]) ** 2           # (N,H,D)
                acc[space] += diff2.mean(0).sum(0).numpy()     # mean over N, sum over H -> (D,)
        pd_norm[i] = acc['norm'] / int(cfg.K)
        pd_raw[i] = acc['raw'] / int(cfg.K)
        if i % 20 == 0:
            g = {n: pd_norm[i][ix].sum() for n, ix in GROUPS}
            print(f"[dim] t={t} S_norm={pd_norm[i].sum():.3e}  "
                  + " ".join(f"{n}={v:.2e}" for n, v in g.items()))
    return dict(timesteps=np.asarray(ts), per_dim_norm=pd_norm, per_dim_raw=pd_raw,
                action_dim=Da, grasp=None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--output_dir', required=True)
    ap.add_argument('--dataset_grasp_npz', default=None, help='an existing perturb.npz for grasp shading')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--demo', type=int, default=0)
    ap.add_argument('--stride', type=int, default=3)
    ap.add_argument('--K', type=int, default=8)
    ap.add_argument('--N', type=int, default=16)
    ap.add_argument('--noise_scale', type=float, default=0.1)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--seed_crn', type=int, default=0)
    ap.add_argument('--max_batch', type=int, default=64)
    cfg = ap.parse_args()
    os.makedirs(cfg.output_dir, exist_ok=True)

    out = compute(cfg)
    ts, pdn = out['timesteps'], out['per_dim_norm']
    np.savez(os.path.join(cfg.output_dir, 'per_dim.npz'), **out)

    S = pdn.sum(1)
    grp = {n: pdn[:, ix].sum(1) for n, ix in GROUPS}
    frac = {n: grp[n] / np.clip(S, 1e-12, None) for n in grp}

    # phase windows (demo_0 tool_hang, from the phase diagnostic)
    def win(*sp):
        m = np.zeros(len(ts), bool)
        for a, b in sp: m |= (ts >= a) & (ts <= b)
        return m
    phases = [('reach', win((0, 75), (455, 505))),
              ('transport', win((85, 285), (515, 610))),
              ('insertion', win((295, 380), (615, 672)))]

    # Fig 1: stacked group contribution over t
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.stackplot(ts, grp['position'], grp['rot6d'], grp['gripper'],
                 labels=['position', 'rot6d', 'gripper'], colors=['C0', 'C1', 'C3'], alpha=0.85)
    ax.set_yscale('symlog', linthresh=1e-2)
    ax.set_xlabel('episode timestep t'); ax.set_ylabel('per-group contribution to S (norm)')
    ax.set_title('Which action dims drive S(t)?  (stacked group contribution)')
    ax.legend(fontsize=9, loc='upper center'); ax.grid(True, alpha=0.2)
    fig.tight_layout(); fig.savefig(os.path.join(cfg.output_dir, 'action_dim_stacked.png'), dpi=120); plt.close(fig)

    # Fig 2: gripper fraction over t
    fig, ax = plt.subplots(figsize=(12, 4))
    for n, c in [('position', 'C0'), ('rot6d', 'C1'), ('gripper', 'C3')]:
        ax.plot(ts, frac[n], color=c, lw=1.3, label=n)
    ax.set_ylim(0, 1.02); ax.set_xlabel('episode timestep t'); ax.set_ylabel('fraction of S')
    ax.set_title('Fraction of S contributed by each action-dim group')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.2)
    fig.tight_layout(); fig.savefig(os.path.join(cfg.output_dir, 'action_dim_fraction.png'), dpi=120); plt.close(fig)

    # Fig 3: phase-averaged per-dim bar
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(out['action_dim'])
    for (pn, m), off in zip(phases, [-0.25, 0, 0.25]):
        ax.bar(x + off, np.nanmean(pdn[m], 0), width=0.25, label=pn)
    ax.set_yscale('log'); ax.set_xticks(x)
    ax.set_xticklabels(['px', 'py', 'pz', 'r0', 'r1', 'r2', 'r3', 'r4', 'r5', 'grip'])
    ax.set_xlabel('action dimension'); ax.set_ylabel('mean contribution to S (norm)')
    ax.set_title('Per-action-dim contribution by phase')
    ax.legend(fontsize=9); ax.grid(True, which='both', axis='y', alpha=0.2)
    fig.tight_layout(); fig.savefig(os.path.join(cfg.output_dir, 'action_dim_by_phase.png'), dpi=120); plt.close(fig)

    # report
    lines = ["# Per-action-dim decomposition of obs-noise S(t)\n",
             "| phase | position | rot6d | gripper | gripper % |", "|---|---|---|---|---|"]
    for pn, m in phases:
        g = {n: float(np.nanmean(grp[n][m])) for n in grp}
        tot = sum(g.values())
        lines.append(f"| {pn} | {g['position']:.3e} | {g['rot6d']:.3e} | {g['gripper']:.3e} | "
                     f"{100*g['gripper']/max(tot,1e-12):.1f}% |")
    lines.append(f"\n- overall gripper share of total S: "
                 f"{100*grp['gripper'].sum()/max(S.sum(),1e-12):.1f}%")
    with open(os.path.join(cfg.output_dir, 'report.md'), 'w') as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"[dim] wrote {cfg.output_dir}")


if __name__ == '__main__':
    main()
