"""Standalone analysis for Experiment 2: consumes exp2.npz (+ optional prelim npz), writes
correlations + figures + summary.md. Re-runnable on any past run's outputs.

The headline result is the Spearman correlation S_vel vs S_end (norm space, on the S_end
subset) — it licenses the cheap velocity estimator. The summary LEADS with the
checkpoint-derived run identity so ablation outputs from different checkpoints can't be
silently mixed.

Usage:
    python -m diffusion_policy.experiments.spatial_attention_exp2.analysis.run_analysis \
      npz=<run>/exp2.npz output_dir=<run> [compare_prelim_npz=<prelim>/perturb.npz]
"""
import os
import pathlib

import hydra
import numpy as np
from omegaconf import OmegaConf
from scipy.stats import spearmanr

from diffusion_policy.experiments.spatial_attention_exp2.analysis import plots
from diffusion_policy.experiments.spatial_attention_exp2.util import validate_timeline_episode


def _load(npz_path):
    d = np.load(npz_path, allow_pickle=True)
    return {k: d[k] for k in d.files}


def _spear(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return float('nan'), int(m.sum())
    return float(spearmanr(a[m], b[m]).correlation), int(m.sum())


def analyze(npz_path, output_dir, yscale='log', compare_prelim_npz=None):
    os.makedirs(output_dir, exist_ok=True)
    data = _load(npz_path)
    validate_timeline_episode(data['timeline_episode'], list(data['eval_episodes']))
    spaces = [str(s) for s in data['distance_spaces']]
    dirs = ['temporal', 'config'] if str(data['directions']) == 'both' else [str(data['directions'])]

    figs = [plots.plot_timeline(data, output_dir, yscale=yscale),
            plots.plot_perindex_heatmap(data, output_dir),
            plots.plot_histogram(data, output_dir),
            plots.plot_flagged(data, output_dir)]
    if compare_prelim_npz:
        figs.append(plots.plot_prelim_overlay(data, _load(compare_prelim_npz), output_dir))

    lines = [f"# Exp2 S(o) — {data['task_name']} / {data['obs_variant']} "
             f"(abs_action={bool(data['abs_action'])})\n",
             f"- checkpoint: `{str(data['checkpoint'])}`",
             f"- probe_type: {data['probe_type']}, similarity_space: {data['similarity_space']}, "
             f"directions: {data['directions']}, include_gripper: {bool(data['include_gripper'])}",
             f"- K={int(data['K'])}, N={int(data['N'])}, M={int(data['M'])}, "
             f"num_tz_draws={int(data['num_tz_draws'])}, stride={int(data['stride'])}",
             f"- probe pool: {len(data['pool_episodes'])} states over {int(data['n_demos'])} demos; "
             f"query: {len(data['query_episodes'])} states in {list(data['eval_episodes'])}; "
             f"S_end on {len(data['endpoint_subset'])} states",
             f"- flagged fraction: **{float(data['flagged_fraction']):.1%}** "
             f"(fallback={data['fallback']}), index_reused={bool(data['index_reused'])}", ""]

    # (a) S_vel vs S_end correlation on the S_end subset — headline = norm space
    lines.append("## S_vel vs S_end (Spearman, on the S_end subset)")
    lines.append("| space | state-scalar | S_first | per-index (mean over k) |")
    lines.append("|---|---|---|---|")
    for sp in spaces:
        rs, n = _spear(data['Svel_scalar'], data[f'Send_scalar_{sp}'])
        rf, _ = _spear(data['Svel_first'], data[f'Send_first_{sp}'])
        pk = [_spear(data['Svel_per_index'][:, k], data[f'Send_per_index_{sp}'][:, k])[0]
              for k in range(int(data['horizon']))]
        pk_mean = float(np.nanmean(pk)) if len(pk) else float('nan')
        tag = ' (HEADLINE)' if sp == 'norm' else ''
        lines.append(f"| {sp}{tag} | {rs:.3f} (n={n}) | {rf:.3f} | {pk_mean:.3f} |")
    lines.append("")

    # (d) directions
    if str(data['directions']) == 'both':
        rho, _ = _spear(data['Svel_scalar_temporal'], data['Svel_scalar_config'])
        lines.append(f"## Directions\n- Spearman(S_temporal, S_config) over query states: **{rho:.3f}**")
    names = list(data['phase_names'])
    lines.append("\n## Per-phase mean S_vel")
    lines.append("| phase | mean S_vel | flagged frac | n |")
    lines.append("|---|---|---|---|")
    for p in np.unique(data['query_phase']):
        mask = data['query_phase'] == p
        v = data['Svel_scalar'][mask]
        lines.append(f"| {names[int(p)]} | {np.nanmean(v):.3e} | "
                     f"{float(np.mean(data['flagged'][mask])):.1%} | {int(mask.sum())} |")

    lines.append("\n## Figures")
    for f in figs:
        lines.append(f"- {os.path.basename(f)}")
    md = os.path.join(output_dir, 'summary.md')
    with open(md, 'w') as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"[exp2:analysis] wrote {md} and {len(figs)} figures to {output_dir}")
    return md


@hydra.main(version_base=None,
            config_path=str(pathlib.Path(__file__).parent.parent.joinpath('config')),
            config_name='analysis')
def main(cfg):
    OmegaConf.resolve(cfg)
    analyze(cfg.npz, cfg.output_dir, yscale=cfg.get('yscale', 'log'),
            compare_prelim_npz=cfg.get('compare_prelim_npz', None))


if __name__ == '__main__':
    main()
