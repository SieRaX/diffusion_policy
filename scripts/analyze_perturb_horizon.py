"""Horizon / phase decomposition of the obs-noise perturbation-sensitivity metric.

Investigates why the coupled endpoint distance S(t) is large at grasp/transition
moments but small during a precise insertion phase. The metric sums the per-chunk-index
distance over the full H-step prediction horizon; this script separates the EXECUTED
window (the actions actually run in closed loop) from the far-future TAIL, and correlates
the S(t) spikes with motion transitions in the demo.

Usage (read-only; writes figures + report into --output_dir):
    python scripts/analyze_perturb_horizon.py \
        --npz  <.../obs_noise/perturb.npz> \
        --dataset data/robomimic/datasets/tool_hang/ph/low_dim.hdf5 --demo 0 \
        --output_dir <.../analysis_horizon> [--n_action_steps 8] \
        [--compare label1=path1/perturb.npz label2=path2/perturb.npz ...]

The --compare npz files (E2/E3 attribution runs) are overlaid on one S(t) figure.
"""
import argparse
import os

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr


def load_npz(path):
    d = np.load(path, allow_pickle=True)
    return {k: d[k] for k in d.files}


def grasped_any(data):
    gf = data['grasp_flags']
    if gf.ndim == 2 and gf.shape[1] > 0:
        return gf.any(axis=1)
    return np.zeros(len(data['timesteps']), bool)


def shade_grasp(ax, ts, ga):
    i, n, first = 0, len(ts), True
    while i < n:
        if ga[i]:
            j = i
            while j + 1 < n and ga[j + 1]:
                j += 1
            ax.axvspan(ts[i], ts[j], color='orange', alpha=0.13,
                       label='grasped' if first else None)
            first = False
            i = j + 1
        else:
            i += 1


def eef_speed(dataset, demo):
    """Per-timestep end-effector speed from the demo obs (motion / transition proxy)."""
    with h5py.File(dataset, 'r') as f:
        pos = f[f'data/demo_{demo}/obs/robot0_eef_pos'][:]  # (T,3)
    v = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    return np.concatenate([v, v[-1:]])  # (T,)


def eef_gripper(dataset, demo):
    """End-effector height and gripper openness over the demo (phase diagnostic)."""
    with h5py.File(dataset, 'r') as f:
        z = f[f'data/demo_{demo}/obs/robot0_eef_pos'][:, 2]
        grip = np.abs(f[f'data/demo_{demo}/obs/robot0_gripper_qpos'][:]).sum(1)
    return z, grip


def decompose(data, n_action_steps):
    ts = data['timesteps']
    per = data['per_index_norm']                 # (T,H), sum over H == S_norm
    H = int(data['horizon'])
    start = int(data['executed_start'])          # To-1
    end = min(start + int(n_action_steps), H)
    S_full = per.sum(1)
    S_exec = per[:, start:end].sum(1)            # actions actually executed in closed loop
    S_tail = per[:, end:].sum(1)                 # far-future prediction tail
    S_first = per[:, start]
    tail_frac = S_tail / np.clip(S_full, 1e-12, None)
    return dict(ts=ts, H=H, start=start, end=end, per=per, S_full=S_full,
                S_exec=S_exec, S_tail=S_tail, S_first=S_first, tail_frac=tail_frac)


def fig_horizon_decomposition(dec, ga, out):
    """Per-horizon-index distance profile at representative insertion vs transition t."""
    ts, per, S_full = dec['ts'], dec['per'], dec['S_full']
    ins = (ts >= 150) & (ts <= 300)
    ins_idx = np.where(ins)[0]
    lows = ins_idx[np.argsort(S_full[ins_idx])[:2]] if ins_idx.size else []
    highs = np.argsort(S_full)[-2:]
    fig, ax = plt.subplots(figsize=(9, 5))
    for i in lows:
        ax.plot(range(dec['H']), per[i], color='C0', alpha=0.8,
                label=f'insertion t={int(ts[i])} (S={S_full[i]:.2f})')
    for i in highs:
        ax.plot(range(dec['H']), per[i], color='C3', alpha=0.8,
                label=f'transition t={int(ts[i])} (S={S_full[i]:.1f})')
    ax.axvspan(dec['start'], dec['end'] - 1, color='green', alpha=0.12,
               label=f'executed window [{dec["start"]}:{dec["end"]}]')
    ax.set_yscale('log')
    ax.set_xlabel('chunk / horizon index h')
    ax.set_ylabel('per-index distance  (mean over N, sum over D)')
    ax.set_title('Where in the predicted chunk does S live?  insertion vs transition')
    ax.legend(fontsize=8)
    ax.grid(True, which='both', alpha=0.2)
    fig.tight_layout(); fig.savefig(out, dpi=120); plt.close(fig)


def fig_near_vs_tail(dec, ga, out):
    ts = dec['ts']
    fig, ax = plt.subplots(figsize=(11, 5))
    shade_grasp(ax, ts, ga)
    floor = 1e-3
    clip = lambda a: np.clip(a, floor, None)
    ax.plot(ts, clip(dec['S_full']), color='C3', lw=1.4, label='S_full (whole chunk)')
    ax.plot(ts, clip(dec['S_exec']), color='C0', lw=1.4, label='S_exec (executed window)')
    ax.plot(ts, clip(dec['S_first']), color='C2', lw=1.0, alpha=0.8, label='S_first (1st executed)')
    ax.set_yscale('log')
    ax.set_xlabel('episode timestep t'); ax.set_ylabel('coupled endpoint distance (norm)')
    ax.set_title('Transition/precision > transport contrast holds at every horizon depth '
                 '(near-term action far more robust in absolute terms)')
    ax.legend(fontsize=8, ncol=2); ax.grid(True, which='both', alpha=0.2)
    fig.tight_layout(); fig.savefig(out, dpi=120); plt.close(fig)


def fig_phase_diagnostic(data, dec, dataset, demo, out):
    """Overlay S_full with eef height + gripper openness so phases (reach / lift /
    transport / insert / release) are legible against the sensitivity curve."""
    ts = dec['ts']
    z, grip = eef_gripper(dataset, demo)
    ga = grasped_any(data)
    fig, ax = plt.subplots(figsize=(12, 5))
    shade_grasp(ax, ts, ga)
    ax.plot(ts, np.clip(dec['S_full'], 1e-3, None), color='C3', lw=1.5, label='S_full (norm)')
    ax.set_yscale('log'); ax.set_ylabel('S_full (norm)', color='C3')
    ax.set_xlabel('episode timestep t')
    axb = ax.twinx()
    axb.plot(ts, z[ts], color='C0', lw=1.1, alpha=0.8, label='eef height z')
    axb.plot(ts, grip[ts], color='C2', lw=1.1, alpha=0.8, label='gripper openness')
    axb.set_ylabel('eef z (m) / gripper openness')
    ax.set_title('S_full vs phase: high at reach-to-grasp (gripper open, descending) and '
                 'insertion (grasped, placing); low during transport')
    l1, la1 = ax.get_legend_handles_labels()
    l2, la2 = axb.get_legend_handles_labels()
    ax.legend(l1 + l2, la1 + la2, fontsize=8, ncol=2, loc='upper center')
    ax.grid(True, which='both', alpha=0.15)
    fig.tight_layout(); fig.savefig(out, dpi=120); plt.close(fig)


def fig_tail_fraction(dec, ga, out):
    ts = dec['ts']
    fig, ax = plt.subplots(figsize=(11, 4))
    shade_grasp(ax, ts, ga)
    ax.plot(ts, dec['tail_frac'], color='C4', lw=1.2, label='tail fraction of S')
    ax.axhline(0.5, color='k', ls=':', alpha=0.5)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel('episode timestep t'); ax.set_ylabel('S_tail / S_full')
    ax.set_title('Fraction of S contributed by the far-future prediction tail')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.2)
    fig.tight_layout(); fig.savefig(out, dpi=120); plt.close(fig)


def fig_transition_corr(dec, ga, spd, out):
    ts = dec['ts']
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
    shade_grasp(a1, ts, ga)
    a1.plot(ts, np.clip(dec['S_full'], 1e-3, None), color='C3', lw=1.3, label='S_full')
    a1.set_yscale('log'); a1.set_ylabel('S_full (norm)', color='C3')
    a1b = a1.twinx()
    a1b.plot(ts, spd, color='C0', lw=1.0, alpha=0.7, label='eef speed')
    a1b.set_ylabel('eef speed (m/step)', color='C0')
    # mark grasp-state changes
    chg = np.where(np.diff(ga.astype(int)) != 0)[0]
    for c in chg:
        a1.axvline(ts[c], color='purple', ls='--', alpha=0.5)
    a1.set_xlabel('episode timestep t')
    a1.set_title('S_full vs motion (eef speed); dashed = grasp-state change')
    # scatter
    m = np.isfinite(dec['S_full']) & (dec['S_full'] > 0)
    rho_full, _ = spearmanr(spd[m], dec['S_full'][m])
    rho_exec, _ = spearmanr(spd[m], dec['S_exec'][m])
    a2.scatter(spd[m], dec['S_full'][m], s=10, alpha=0.5, color='C3', label=f'S_full (ρ={rho_full:.2f})')
    a2.scatter(spd[m], dec['S_exec'][m], s=10, alpha=0.5, color='C0', label=f'S_exec (ρ={rho_exec:.2f})')
    a2.set_yscale('log'); a2.set_xlabel('eef speed (m/step)'); a2.set_ylabel('distance (norm)')
    a2.set_title('S vs eef speed (Spearman)')
    a2.legend(fontsize=8); a2.grid(True, which='both', alpha=0.2)
    fig.tight_layout(); fig.savefig(out, dpi=120); plt.close(fig)
    return rho_full, rho_exec


def fig_compare(compare, out):
    fig, ax = plt.subplots(figsize=(11, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(compare), 1)))
    for (label, data), c in zip(compare, colors):
        ax.plot(data['timesteps'], np.clip(data['S_norm'], 1e-3, None), lw=1.2, color=c, label=label)
    ga = grasped_any(compare[0][1])
    shade_grasp(ax, compare[0][1]['timesteps'], ga)
    ax.set_yscale('log'); ax.set_xlabel('episode timestep t'); ax.set_ylabel('S_full (norm)')
    ax.set_title('Attribution: S(t) under per-obs-group masks and isotropic noise')
    ax.legend(fontsize=8, ncol=2); ax.grid(True, which='both', alpha=0.2)
    fig.tight_layout(); fig.savefig(out, dpi=120); plt.close(fig)


def _win(ts, *spans):
    m = np.zeros(len(ts), bool)
    for a, b in spans:
        m |= (ts >= a) & (ts <= b)
    return m


def phase_stats(dec, ga):
    """Phase windows read off the eef-height / gripper / grasp diagnostic (demo_0)."""
    ts = dec['ts']
    phases = [
        ('reach-to-grasp', _win(ts, (0, 75), (455, 505))),
        ('lift+transport', _win(ts, (85, 285), (515, 610))),
        ('insertion/place', _win(ts, (295, 380), (615, 672))),
    ]
    def mean(a, m): return float(np.nanmean(a[m])) if m.any() else float('nan')
    rows = []
    for name, m in phases + [('all', np.ones(len(ts), bool))]:
        rows.append((name, mean(dec['S_full'], m), mean(dec['S_exec'], m),
                     mean(dec['S_first'], m), mean(dec['tail_frac'], m)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz', required=True)
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--demo', type=int, default=0)
    ap.add_argument('--output_dir', required=True)
    ap.add_argument('--n_action_steps', type=int, default=8)
    ap.add_argument('--compare', nargs='*', default=[], help='label=path/perturb.npz ...')
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    data = load_npz(args.npz)
    ga = grasped_any(data)
    spd = eef_speed(args.dataset, args.demo)[data['timesteps']]
    dec = decompose(data, args.n_action_steps)

    fig_horizon_decomposition(dec, ga, os.path.join(args.output_dir, 'horizon_decomposition.png'))
    fig_near_vs_tail(dec, ga, os.path.join(args.output_dir, 'near_vs_tail_timeline.png'))
    fig_tail_fraction(dec, ga, os.path.join(args.output_dir, 'tail_fraction.png'))
    fig_phase_diagnostic(data, dec, args.dataset, args.demo, os.path.join(args.output_dir, 'phase_diagnostic.png'))
    rho_full, rho_exec = fig_transition_corr(dec, ga, spd, os.path.join(args.output_dir, 'transition_correlation.png'))

    compare = [(args.npz.split('/')[-2], data)]  # full-mask reference
    for spec in args.compare:
        label, path = spec.split('=', 1)
        if os.path.exists(path):
            compare.append((label, load_npz(path)))
    if len(compare) > 1:
        fig_compare(compare, os.path.join(args.output_dir, 'attribution_compare.png'))

    rows = phase_stats(dec, ga)
    lines = [f"# Horizon / phase decomposition — {data['task_name']} / {data['obs_variant']}\n",
             f"- npz: `{args.npz}`",
             f"- H={dec['H']}, executed window = idx[{dec['start']}:{dec['end']}] "
             f"(n_action_steps={args.n_action_steps}), tail = idx[{dec['end']}:{dec['H']}]",
             f"- Spearman(eef_speed, S_full) = **{rho_full:.3f}**, "
             f"Spearman(eef_speed, S_exec) = **{rho_exec:.3f}**\n",
             "## Phase means",
             "| phase | S_full | S_exec | S_first | tail_frac |",
             "|---|---|---|---|---|"]
    for name, sf, se, s1, tf in rows:
        lines.append(f"| {name} | {sf:.3e} | {se:.3e} | {s1:.3e} | {tf:.3f} |")
    lines += ["",
              "## Reading",
              "- Phases (demo_0 eef-height/gripper/grasp): reach-to-grasp and insertion/place are "
              "**high**-S; lift+transport is **low**-S. The t=150-300 window the eye reads as "
              "'insertion' is actually the transport glide; the true hook-into-base insertion "
              "(t~300-380) is the largest S peak.",
              f"- S is NOT driven by fast motion — Spearman(eef_speed, S_full)={rho_full:.2f} "
              "(mildly negative: precision phases are slow).",
              "- The precision>transport contrast holds at every horizon depth (S_first, S_exec, "
              "S_full), amplified by prediction depth (tail_frac high everywhere).",
              "", "## Figures",
              "- phase_diagnostic.png — S_full vs eef height + gripper (phase labels)",
              "- horizon_decomposition.png — per-index profile, transport vs insertion",
              "- near_vs_tail_timeline.png — S_full vs S_exec vs S_first",
              "- tail_fraction.png — S_tail/S_full over t",
              "- transition_correlation.png — S vs eef speed + grasp-change markers"]
    if len(compare) > 1:
        lines.append("- attribution_compare.png — per-group-mask / isotropic S(t)")
    md = os.path.join(args.output_dir, 'report.md')
    with open(md, 'w') as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"[horizon] wrote {md} and figures to {args.output_dir}")
    for r in rows:
        print(r)


if __name__ == '__main__':
    main()
