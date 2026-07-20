"""Figures for the perturbation-sensitivity experiment. Pure matplotlib (Agg),
consumes arrays loaded from perturb.npz. Never mixes distance spaces in one panel
without a label (each carries its _raw / _norm suffix)."""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def _grasped_any(data):
    gf = data['grasp_flags']
    return gf.any(axis=1) if gf.ndim == 2 and gf.shape[1] > 0 else np.zeros(len(data['timesteps']), bool)


def _shade_grasp(ax, ts, grasped_any):
    """Shade contiguous grasped intervals."""
    i = 0
    n = len(ts)
    first = True
    while i < n:
        if grasped_any[i]:
            j = i
            while j + 1 < n and grasped_any[j + 1]:
                j += 1
            ax.axvspan(ts[i], ts[j], color='orange', alpha=0.15,
                       label='grasp' if first else None)
            first = False
            i = j + 1
        else:
            i += 1


def plot_timeline(data, space, out_dir, yscale='log'):
    """S(t) & S_first(t) vs t for one distance space, with grasp shading, D_k
    10-90 percentile band, and the nominal-vs-nominal control overlaid.

    yscale='log' (default): y-limits are taken from the SIGNAL (S/S_first/D_k), not
    the ~0 control (which would otherwise force ~30 wasted decades); the control is
    clipped to the axis floor and its true magnitude is shown in the legend. Decade
    major ticks + 2-9 minor ticks + a grid are added. yscale='linear': axis from 0."""
    from matplotlib.ticker import LogLocator
    path = os.path.join(out_dir, f'timeline_{space}.png')
    ts = data['timesteps']
    S = data[f'S_{space}']
    Sf = data[f'S_first_{space}']
    Dk = data[f'D_k_{space}']                     # (T, K)
    ctrl = data[f'control_{space}']               # (T,)
    grasped_any = _grasped_any(data)

    if yscale == 'log':
        sig = np.concatenate([S, Sf, Dk.ravel()])
        sig = sig[np.isfinite(sig) & (sig > 0)]
        floor = float(sig.min() * 0.5) if sig.size else 1e-9
        top = float(sig.max() * 2.0) if sig.size else 1.0
        clip = lambda a: np.clip(a, floor, None)
    else:
        clip = lambda a: a

    fig, ax = plt.subplots(figsize=(11, 5))
    _shade_grasp(ax, ts, grasped_any)
    if Dk.shape[1] > 1:
        lo = clip(np.percentile(Dk, 10, axis=1))
        hi = clip(np.percentile(Dk, 90, axis=1))
        ax.fill_between(ts, lo, hi, color='C0', alpha=0.15, label='D_k 10-90%')
    ax.plot(ts, clip(S), marker='.', color='C0', label=f'S (full chunk) [{space}]')
    ax.plot(ts, clip(Sf), marker='.', color='C1', label=f'S_first (executed) [{space}]')
    cm = np.isfinite(ctrl)
    if cm.any():
        cmax = float(np.nanmax(ctrl[cm]))
        ax.plot(ts[cm], clip(ctrl[cm]), 'x', color='k', markersize=5,
                label=f'control (nom-vs-nom, max={cmax:.1e})')
    if yscale == 'log':
        ax.set_yscale('log')
        ax.set_ylim(floor, top)
        ax.yaxis.set_major_locator(LogLocator(base=10.0))
        ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10), numticks=100))
        ax.grid(True, which='major', alpha=0.35)
        ax.grid(True, which='minor', alpha=0.12)
    else:
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.25)
    ax.set_xlabel('episode timestep t')
    ax.set_ylabel(f'coupled endpoint distance ({space})')
    grip_suffix = '' if (bool(data['include_gripper']) if 'include_gripper' in data else True) \
        else ' (gripper excluded)'
    ax.set_title(f"Perturbation sensitivity vs timestep — {data['task_name']} "
                 f"/ {data['obs_variant']} [{space}]{grip_suffix}")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)
    return path


def plot_rank_comparison(data, out_dir):
    """Scatter of per-timestep rank(S_raw) vs rank(S_norm); Spearman in the title."""
    from scipy.stats import spearmanr, rankdata
    if 'S_raw' not in data or 'S_norm' not in data:
        return None
    path = os.path.join(out_dir, 'rank_raw_vs_norm.png')
    sr, sn = data['S_raw'], data['S_norm']
    rho, _ = spearmanr(sr, sn)
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(rankdata(sr), rankdata(sn), s=14, alpha=0.6)
    ax.plot([1, len(sr)], [1, len(sr)], 'k--', alpha=0.4)
    ax.set_xlabel('rank of S_raw(t)'); ax.set_ylabel('rank of S_norm(t)')
    ax.set_title(f'Do raw/norm reorder the peaks?  Spearman ρ = {rho:.3f}')
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)
    return path


def plot_heatmap(data, space, out_dir):
    """Heatmap of S(t,k): timestep × chunk index for one distance space."""
    path = os.path.join(out_dir, f'heatmap_{space}.png')
    per_index = data[f'per_index_{space}']        # (T, H)
    ts = data['timesteps']
    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(np.log10(np.clip(per_index.T, 1e-30, None)), aspect='auto',
                   origin='lower', cmap='viridis',
                   extent=[ts[0], ts[-1], -0.5, per_index.shape[1] - 0.5])
    ax.set_xlabel('episode timestep t'); ax.set_ylabel('chunk index k')
    ax.set_title(f"S(t,k) log10 — {data['task_name']} / {data['obs_variant']} [{space}]")
    fig.colorbar(im, ax=ax, label='log10 S(t,k)')
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)
    return path
