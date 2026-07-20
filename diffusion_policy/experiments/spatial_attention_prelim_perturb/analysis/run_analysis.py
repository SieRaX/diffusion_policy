"""Analysis entry point (standalone from the npz). Produces the timeline panels
(per distance space), the raw-vs-norm rank scatter, the S(t,k) heatmaps, and a
markdown summary.

Usage:
    python -m diffusion_policy.experiments.spatial_attention_prelim_perturb.analysis.run_analysis \
      npz=outputs/prelim_perturb_<task>_<obs>/perturb.npz output_dir=outputs/prelim_perturb_<task>_<obs>
"""
import os
import pathlib

import hydra
import numpy as np
from omegaconf import OmegaConf

from diffusion_policy.experiments.spatial_attention_prelim_perturb.analysis import plots


def _load(npz_path):
    d = np.load(npz_path, allow_pickle=True)
    return {k: d[k] for k in d.files}


def _grasp_masks(data):
    gf = data['grasp_flags']
    grasped = gf.any(axis=1) if gf.ndim == 2 and gf.shape[1] > 0 else np.zeros(len(data['timesteps']), bool)
    return grasped, ~grasped


def analyze(npz_path, output_dir, yscale='log'):
    os.makedirs(output_dir, exist_ok=True)
    data = _load(npz_path)
    spaces = [str(s) for s in data['distance_spaces']]
    backend = str(data.get('perturbation_backend', 'sim_state'))  # old npz -> sim_state

    figs = []
    for sp in spaces:
        figs.append(plots.plot_timeline(data, sp, output_dir, yscale=yscale))
        figs.append(plots.plot_heatmap(data, sp, output_dir))
    rank_fig = plots.plot_rank_comparison(data, output_dir)
    if rank_fig:
        figs.append(rank_fig)

    grasped, nongrasped = _grasp_masks(data)

    def _mean(a, mask):
        a = np.asarray(a, dtype=float)[mask]
        return float(np.mean(a)) if a.size else float('nan')

    lines = [f"# Perturbation sensitivity — {data['task_name']} / {data['obs_variant']} "
             f"(abs_action={bool(data['abs_action'])})\n"]
    lines.append(f"- demo {int(data['demo_index'])}, episode length {int(data['episode_length'])}, "
                 f"evaluated {len(data['timesteps'])} timesteps (stride implied)")
    lines.append(f"- backend: {backend}, history_mode: {data['history_mode']}, distance spaces: {spaces}")
    lines.append(f"- K={int(data['K'])}, N={int(data['N'])}, M={int(data['M'])}, "
                 f"executed_start={int(data['executed_start'])}, H={int(data['horizon'])}, D={int(data['action_dim'])}")
    if backend == 'obs_noise':
        nm = str(data.get('noise_mode', 'per_dim_std'))
        detail = (f"noise_scale={float(data['noise_scale'])}" if nm == 'per_dim_std'
                  else f"sigma_abs={float(data['sigma_abs'])}")
        mask = np.asarray(data['noise_dim_mask'], dtype=bool) if 'noise_dim_mask' in data else None
        ndims = f"{int(mask.sum())}/{mask.size}" if mask is not None else "all"
        lines.append(f"- obs_noise: mode={nm}, {detail}, noised {ndims} obs dims; "
                     f"delta seed={int(data.get('noise_seed', -1))}, eps0 seed={int(data['seed_crn'])}")
    else:
        lines.append(f"- perturb_targets: {list(data['perturb_targets'])}; "
                     f"sigmas: pos(eef={data['sigma_pos_eef']}, obj={data['sigma_pos_object']}) "
                     f"rot(eef={data['sigma_rot_eef']}, obj={data['sigma_rot_object']}); "
                     f"seeds perturb={int(data['seed_perturb'])} crn={int(data['seed_crn'])}")
    lines.append(f"- bodies: {list(data['body_names'])}; grasped in {int(grasped.sum())}/"
                 f"{len(grasped)} evaluated steps")
    if backend != 'obs_noise' and list(data['perturb_targets']) == ['objects']:
        lines.append("- **perturb_targets fallback**: objects-only (no EEF/arm perturbation; EEF obs held "
                     "fixed — grasped-body perturbation is not a full physical grasp motion).")
    lines.append("")

    lines.append("## Mean sensitivity: grasped vs non-grasped intervals")
    lines.append("| space | S grasped | S non-grasped | S_first grasped | S_first non-grasped | control (max) |")
    lines.append("|---|---|---|---|---|---|")
    for sp in spaces:
        S, Sf = data[f'S_{sp}'], data[f'S_first_{sp}']
        ctrl = data[f'control_{sp}']
        cmax = np.nanmax(ctrl) if np.isfinite(ctrl).any() else float('nan')
        lines.append(f"| {sp} | {_mean(S,grasped):.3e} | {_mean(S,nongrasped):.3e} | "
                     f"{_mean(Sf,grasped):.3e} | {_mean(Sf,nongrasped):.3e} | {cmax:.2e} |")
    lines.append("")

    if 'S_raw' in data and 'S_norm' in data:
        from scipy.stats import spearmanr
        rho, _ = spearmanr(data['S_raw'], data['S_norm'])
        lines.append(f"## Raw vs norm\n- Spearman rank correlation of S_raw vs S_norm over timesteps: "
                     f"**{rho:.3f}** (do the two spaces reorder the peaks?)\n")

    lines.append("## Figures")
    for f in figs:
        lines.append(f"- {os.path.basename(f)}")

    md = os.path.join(output_dir, 'summary.md')
    with open(md, 'w') as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"[analysis] wrote {md} and {len(figs)} figures to {output_dir}")
    return md


@hydra.main(version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath('config')),
    config_name='analysis')
def main(cfg):
    OmegaConf.resolve(cfg)
    analyze(cfg.npz, cfg.output_dir, yscale=cfg.get('yscale', 'log'))


if __name__ == '__main__':
    main()
