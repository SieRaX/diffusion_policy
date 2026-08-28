"""
Strictly re-match MoH dynamic inference's average executed horizon to a target
window (default 32 +/- 0.5) on a finished MoH training run.

The original sweep tuned r on the best checkpoint with 10-episode probes; the
final 5-checkpoint average drifted (tool_hang: 29.7). Here each bisection step
measures the ENSEMBLE average: every top-k checkpoint runs n_probe episodes and
the avgTa is averaged over all of them — the same quantity the final report
uses. After bisection, full finals (n_final per ckpt) are run and the achieved
ensemble avgTa is verified against the window; if it lands outside, one
corrective bisection step is applied using the full-eval measurement.

Usage:
  python scripts/rematch_moh_r.py --run_dir <moh run dir> \
      --r_lo 1.64 --r_hi 2.0
"""
import sys, re, json, pathlib
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
    workspace: BaseWorkspace = cls(cfg, output_dir='/tmp/moh_rematch_ws')
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    policy = workspace.ema_model if cfg.training.use_ema else workspace.model
    policy.to(torch.device(device))
    policy.eval()
    return policy, cfg


@click.command()
@click.option('--run_dir', required=True)
@click.option('-d', '--device', default='cuda:0')
@click.option('--target', default=32.0)
@click.option('--tol', default=0.5)
@click.option('--r_lo', default=1.64, help='bracket low (avgTa below target)')
@click.option('--r_hi', default=2.0, help='bracket high (avgTa above target)')
@click.option('--n_probe', default=10, help='episodes per ckpt per bisect step')
@click.option('--n_final', default=50)
@click.option('--top_k', default=5)
@click.option('--min_active_horizons', default=1)
@click.option('--min_replan_steps', default=4)
@click.option('--max_bisect', default=6)
def main(run_dir, device, target, tol, r_lo, r_hi, n_probe, n_final, top_k,
         min_active_horizons, min_replan_steps, max_bisect):
    run_dir = pathlib.Path(run_dir)
    output_dir = run_dir / 'moh_eval'
    output_dir.mkdir(parents=True, exist_ok=True)

    ckpts = []
    for p in (run_dir / 'checkpoints').glob('epoch=*.ckpt'):
        m = re.search(r'test_mean_score=([\d.]+?)\.ckpt', p.name)
        if m:
            ckpts.append((float(m.group(1)), p))
    ckpts.sort(key=lambda t: -t[0])
    ckpts = ckpts[:top_k]
    print(f"[rematch] {len(ckpts)} checkpoints")

    policies = []
    cfg = None
    for _, p in ckpts:
        pol, cfg = load_policy(p, device)
        pol.min_active_horizons = min_active_horizons
        pol.min_replan_steps = min_replan_steps
        policies.append(pol)

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

    def ensemble_avg_ta(r, n_eps):
        tas = []
        for pol in policies:
            pol.scale_ratio = r
            log = runner.run(pol, mode='dynamic', n_episodes=n_eps)
            tas.append(log['test/avg_exec_horizon'])
        h = float(np.mean(tas))
        print(f"[bisect] r={r:.4f} -> ensemble avgTa={h:.2f} "
              f"(per-ckpt {[round(t,1) for t in tas]})")
        return h

    # bisect on the ensemble measurement
    r_star, h_star = None, None
    lo, hi = r_lo, r_hi
    for _ in range(max_bisect):
        r_mid = 0.5 * (lo + hi)
        h = ensemble_avg_ta(r_mid, n_probe)
        if abs(h - target) <= tol * 0.6:  # tighter than the window for margin
            r_star, h_star = r_mid, h
            break
        if h < target:
            lo = r_mid
        else:
            hi = r_mid
        r_star, h_star = r_mid, h
    print(f"[rematch] selected r = {r_star} (probe ensemble avgTa {h_star:.2f})")

    def run_finals(r):
        results = []
        for (score, p), pol in zip(ckpts, policies):
            pol.scale_ratio = r
            log = runner.run(pol, mode='dynamic')
            log.update(checkpoint=str(p), scale_ratio=r)
            json.dump(log, open(
                output_dir / f'dynamic_rematched_{p.stem}.json', 'w'),
                indent=2, sort_keys=True)
            results.append(log)
            print(f"[final] {p.name} dynamic score "
                  f"{log['test/mean_score']:.3f} "
                  f"avgTa {log['test/avg_exec_horizon']:.1f}")
        return results

    results = run_finals(r_star)
    achieved = float(np.mean([l['test/avg_exec_horizon'] for l in results]))
    corrected = False
    if abs(achieved - target) > tol:
        # one corrective step: linear adjust using the two measurements
        print(f"[rematch] finals avgTa {achieved:.2f} outside window, correcting")
        slope_ref = (achieved - target)
        r_star = r_star + (0.15 if slope_ref < 0 else -0.15)
        h = ensemble_avg_ta(r_star, n_probe)
        results = run_finals(r_star)
        achieved = float(np.mean([l['test/avg_exec_horizon'] for l in results]))
        corrected = True

    scores = [l['test/mean_score'] for l in results]
    summary = {
        'r_rematched': r_star,
        'target': target, 'tol': tol, 'n_final': n_final,
        'corrected': corrected,
        'mean_score': float(np.mean(scores)),
        'std_score': float(np.std(scores)),
        'scores': scores,
        'avg_exec_horizon': achieved,
        'checkpoints': [str(p) for _, p in ckpts],
    }
    json.dump(summary, open(output_dir / 'summary_rematched.json', 'w'),
              indent=2, sort_keys=True)
    print(f"[rematch] DONE dynamic {summary['mean_score']:.3f}"
          f"±{summary['std_score']:.3f} avgTa {achieved:.2f} "
          f"(in window: {abs(achieved - target) <= tol})")


if __name__ == '__main__':
    main()
