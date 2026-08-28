"""
MoH pilot evaluation sweep (phase 2).

For a finished MoH training run dir:
  1. On the best checkpoint, probe scale_ratio r (10 episodes each) and pick /
     bisect the r whose average executed horizon hits the target (default 32,
     the SA paper's T_avg for Square/Tool Hang).
  2. Evaluate the top-k checkpoints with (a) dynamic inference at r*, and
     (b) fixed prefix Ta=target, n_final episodes each.
  3. Write per-eval jsons + a summary (mean/std over checkpoints).

Usage:
  python scripts/eval_moh_sweep.py --run_dir <train run dir> --device cuda:0
"""
import sys, os, re, json, pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent))
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)

import click
import numpy as np
import torch
import dill
import hydra
from omegaconf import OmegaConf

from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.env_runner.robomimic_lowdim_moh_runner import RobomimicLowdimMoHRunner

OmegaConf.register_new_resolver("eval", eval, replace=True)


def load_policy(ckpt, device):
    payload = torch.load(open(ckpt, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg, output_dir='/tmp/moh_eval_ws')
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    policy = workspace.ema_model if cfg.training.use_ema else workspace.model
    policy.to(torch.device(device))
    policy.eval()
    return policy, cfg


@click.command()
@click.option('--run_dir', required=True)
@click.option('--output_dir', default=None)
@click.option('-d', '--device', default='cuda:0')
@click.option('--target', default=32.0, help='target average executed horizon')
@click.option('--tol', default=0.5)
@click.option('--n_probe', default=10)
@click.option('--n_final', default=50)
@click.option('--top_k', default=5)
@click.option('--min_active_horizons', default=1)
@click.option('--min_replan_steps', default=4)
def main(run_dir, output_dir, device, target, tol, n_probe, n_final, top_k,
         min_active_horizons, min_replan_steps):
    run_dir = pathlib.Path(run_dir)
    if output_dir is None:
        output_dir = run_dir / 'moh_eval'
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # rank checkpoints by score in filename
    ckpts = []
    for p in (run_dir / 'checkpoints').glob('epoch=*.ckpt'):
        m = re.search(r'test_mean_score=([\d.]+?)\.ckpt', p.name)
        if m:
            ckpts.append((float(m.group(1)), p))
    ckpts.sort(key=lambda t: -t[0])
    ckpts = ckpts[:top_k]
    assert len(ckpts) > 0, f"no scored checkpoints in {run_dir}"
    best_ckpt = ckpts[0][1]
    print(f"[sweep] {len(ckpts)} checkpoints, best: {best_ckpt.name}")

    policy, cfg = load_policy(best_ckpt, device)
    policy.min_active_horizons = min_active_horizons
    policy.min_replan_steps = min_replan_steps
    runner = RobomimicLowdimMoHRunner(
        output_dir=str(output_dir),
        dataset_path=cfg.task.dataset.dataset_path,
        obs_keys=list(cfg.task.dataset.obs_keys),
        n_test=n_final,
        test_start_seed=100000,
        max_steps=cfg.task.env_runner.max_steps,
        n_obs_steps=cfg.n_obs_steps,
        abs_action=cfg.task.dataset.abs_action,
    )

    # ---- 1) probe r on best checkpoint ----
    def probe(r):
        policy.scale_ratio = r
        log = runner.run(policy, mode='dynamic', n_episodes=n_probe)
        h = log['test/avg_exec_horizon']
        print(f"[probe] r={r:.3f} -> avg_exec_horizon={h:.2f} "
              f"(score {log['test/mean_score']:.2f})")
        return h

    grid = [1.2, 1.5, 2.0, 3.0, 4.5, 6.0]
    probes = [(r, probe(r)) for r in grid]
    r_star = None
    for r, h in probes:
        if abs(h - target) <= tol:
            r_star = r
            break
    if r_star is None:
        # bisect between the bracketing pair (avg horizon increases with r)
        below = [(r, h) for r, h in probes if h < target]
        above = [(r, h) for r, h in probes if h > target]
        if not above:
            r_lo = max(r for r, _ in probes)
            r_hi = r_lo * 3
        elif not below:
            r_hi = min(r for r, _ in probes)
            r_lo = r_hi / 3
        else:
            r_lo = max(below, key=lambda t: t[1])[0]
            r_hi = min(above, key=lambda t: -t[1])[0]
        for _ in range(5):
            r_mid = 0.5 * (r_lo + r_hi)
            h = probe(r_mid)
            probes.append((r_mid, h))
            if abs(h - target) <= tol:
                r_star = r_mid
                break
            if h < target:
                r_lo = r_mid
            else:
                r_hi = r_mid
        if r_star is None:
            r_star = min(probes, key=lambda t: abs(t[1] - target))[0]
    print(f"[sweep] selected r* = {r_star}")

    # ---- 2) final evals on top-k checkpoints ----
    results = {'dynamic': [], 'fixed': []}
    for score, ckpt in ckpts:
        policy, _ = load_policy(ckpt, device)
        policy.min_active_horizons = min_active_horizons
        policy.min_replan_steps = min_replan_steps
        policy.scale_ratio = r_star

        log_d = runner.run(policy, mode='dynamic')
        log_d.update(checkpoint=str(ckpt), scale_ratio=r_star)
        json.dump(log_d, open(output_dir / f'dynamic_{ckpt.stem}.json', 'w'),
                  indent=2, sort_keys=True)
        results['dynamic'].append(log_d)
        print(f"[final] {ckpt.name} dynamic  score {log_d['test/mean_score']:.3f} "
              f"avgTa {log_d['test/avg_exec_horizon']:.1f}")

        log_f = runner.run(policy, mode='fixed', n_action_steps=int(target))
        log_f.update(checkpoint=str(ckpt))
        json.dump(log_f, open(output_dir / f'fixed_{ckpt.stem}.json', 'w'),
                  indent=2, sort_keys=True)
        results['fixed'].append(log_f)
        print(f"[final] {ckpt.name} fixed    score {log_f['test/mean_score']:.3f}")

    # ---- 3) summary ----
    summary = {'r_star': r_star, 'target': target, 'n_final': n_final,
               'min_active_horizons': min_active_horizons,
               'min_replan_steps': min_replan_steps,
               'probes': probes,
               'checkpoints': [str(c) for _, c in ckpts]}
    for mode in ('dynamic', 'fixed'):
        scores = [l['test/mean_score'] for l in results[mode]]
        summary[mode] = {
            'mean_score': float(np.mean(scores)),
            'std_score': float(np.std(scores)),
            'scores': scores,
            'avg_exec_horizon': float(np.mean(
                [l['test/avg_exec_horizon'] for l in results[mode]])),
        }
    json.dump(summary, open(output_dir / 'summary.json', 'w'),
              indent=2, sort_keys=True)
    print(f"[sweep] DONE  dynamic {summary['dynamic']['mean_score']:.3f}"
          f"±{summary['dynamic']['std_score']:.3f} (avgTa "
          f"{summary['dynamic']['avg_exec_horizon']:.1f})  |  fixed "
          f"{summary['fixed']['mean_score']:.3f}±{summary['fixed']['std_score']:.3f}")


if __name__ == '__main__':
    main()
