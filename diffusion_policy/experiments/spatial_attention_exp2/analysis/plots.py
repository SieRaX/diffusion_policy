"""Figures for Experiment 2 (pure matplotlib Agg, consume the exp2.npz dict)."""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_PHASE_COLORS = {0: 'C0', 1: 'C1', 2: 'C2', 3: 'C3'}


def _episode_mask(data, ep):
    return data['query_episodes'] == int(ep)


def _shade_phases(ax, ts, phase, names):
    i, n = 0, len(ts)
    seen = set()
    while i < n:
        j = i
        while j + 1 < n and phase[j + 1] == phase[i]:
            j += 1
        p = int(phase[i])
        lbl = str(names[p]) if p not in seen else None
        seen.add(p)
        ax.axvspan(ts[i], ts[j], color=_PHASE_COLORS.get(p, 'gray'), alpha=0.12, label=lbl)
        i = j + 1


def plot_timeline(data, out_dir, yscale='log'):
    ep = int(data['timeline_episode'])
    m = _episode_mask(data, ep)
    order = np.argsort(data['query_timesteps'][m])
    ts = data['query_timesteps'][m][order]
    S = data['Svel_scalar'][m][order]
    Sf = data['Svel_first'][m][order]
    phase = data['query_phase'][m][order]
    path = os.path.join(out_dir, f'timeline_ep{ep}.png')
    fig, ax = plt.subplots(figsize=(12, 5))
    _shade_phases(ax, ts, phase, data['phase_names'])
    clip = (lambda a: np.clip(a, 1e-9, None)) if yscale == 'log' else (lambda a: a)
    ax.plot(ts, clip(S), marker='.', color='k', label='S_vel (full chunk)')
    ax.plot(ts, clip(Sf), marker='.', color='C4', alpha=0.8, label='S_vel first-executed')
    for d, c in (('temporal', 'C5'), ('config', 'C6')):
        key = f'Svel_scalar_{d}'
        if key in data and str(data['directions']) == 'both':
            ax.plot(ts, clip(data[key][m][order]), lw=1, alpha=0.6, color=c, label=f'S_vel {d}')
    if yscale == 'log':
        ax.set_yscale('log')
    ax.set_xlabel('episode timestep t'); ax.set_ylabel('S_vel (norm)')
    ax.set_title(f"S_vel timeline — {data['task_name']} / {data['obs_variant']} ep{ep}")
    ax.legend(fontsize=8, ncol=2); ax.grid(True, which='both', alpha=0.2)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)
    return path


def plot_perindex_heatmap(data, out_dir):
    ep = int(data['timeline_episode'])
    m = _episode_mask(data, ep)
    order = np.argsort(data['query_timesteps'][m])
    ts = data['query_timesteps'][m][order]
    pi = data['Svel_per_index'][m][order]                 # (T, H)
    path = os.path.join(out_dir, f'heatmap_ep{ep}.png')
    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(np.log10(np.clip(pi.T, 1e-30, None)), aspect='auto', origin='lower',
                   cmap='viridis', extent=[ts[0], ts[-1], -0.5, pi.shape[1] - 0.5])
    ax.set_xlabel('episode timestep t'); ax.set_ylabel('chunk index k')
    ax.set_title(f"S_vel(t,k) log10 — {data['task_name']} ep{ep}")
    fig.colorbar(im, ax=ax, label='log10 S_vel(t,k)')
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)
    return path


def plot_histogram(data, out_dir):
    path = os.path.join(out_dir, 'hist_by_phase.png')
    S = data['Svel_scalar']; phase = data['query_phase']; names = data['phase_names']
    fig, ax = plt.subplots(figsize=(9, 5))
    for p in np.unique(phase):
        v = S[phase == p]
        v = v[np.isfinite(v) & (v > 0)]
        if len(v):
            ax.hist(np.log10(v), bins=30, alpha=0.5, label=str(names[int(p)]))
    ax.set_xlabel('log10 S_vel'); ax.set_ylabel('count')
    ax.set_title('S_vel distribution by phase'); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)
    return path


def plot_flagged(data, out_dir):
    path = os.path.join(out_dir, 'flagged_by_phase.png')
    flagged = data['flagged']; phase = data['query_phase']; names = data['phase_names']
    ps = np.unique(phase)
    frac = [float(np.mean(flagged[phase == p])) if np.any(phase == p) else 0.0 for p in ps]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar([str(names[int(p)]) for p in ps], frac, color='C3', alpha=0.8)
    ax.set_ylabel('flagged fraction'); ax.set_ylim(0, 1)
    ax.set_title(f"Flagged states by phase (overall {float(data['flagged_fraction']):.1%})")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)
    return path


def plot_prelim_overlay(data, prelim, out_dir):
    """Overlay the prelim coupled-endpoint S(t) with this experiment's S_end over the
    timeline episode (different perturbation sources: sim-state vs data-neighbor)."""
    ep = int(data['timeline_episode'])
    if int(prelim['demo_index']) != ep:
        raise ValueError(f"compare_prelim_npz is demo {int(prelim['demo_index'])} but "
                         f"timeline_episode is {ep}; they must match.")
    m = _episode_mask(data, ep)
    order = np.argsort(data['query_timesteps'][m])
    ts = data['query_timesteps'][m][order]
    key = 'Send_scalar_norm' if 'Send_scalar_norm' in data else 'Send_scalar_raw'
    send = data[key][m][order]
    path = os.path.join(out_dir, f'compare_prelim_ep{ep}.png')
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(prelim['timesteps'], np.clip(prelim['S_norm'], 1e-9, None), color='C0',
            label='prelim S(t) (sim-state perturb)')
    finite = np.isfinite(send)
    ax.plot(ts[finite], np.clip(send[finite], 1e-9, None), 'x', color='C3',
            label='exp2 S_end (data-neighbor)')
    ax.set_yscale('log'); ax.set_xlabel('episode timestep t'); ax.set_ylabel('sensitivity (norm)')
    ax.set_title(f'Perturbation-source comparison — ep{ep}')
    ax.legend(fontsize=8); ax.grid(True, which='both', alpha=0.2)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)
    return path
