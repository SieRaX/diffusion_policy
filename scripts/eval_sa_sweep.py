"""
+SA evaluation sweep on a plain DP (DDIM) training run, matching the MoH pilot
protocol (sequential seeded episodes, average executed horizon matched to a
target, top-k checkpoints x n_final episodes).

  1. On the best checkpoint, probe c_att (10 episodes each) and bisect until
     the average executed horizon hits the target (default 32).
  2. Evaluate the top-k checkpoints with (a) +SA at c_att*, and (b) fixed
     prefix Ta=target (the plain-DDIM baseline), n_final episodes each.
  3. Write per-eval jsons + summary.json.

Usage:
  python scripts/eval_sa_sweep.py --run_dir <ddim run dir> \
      --sa_dir data/sa_artifacts/square/attention_estimator_horizon_48
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
from diffusion_policy.model.transformer import Seq2SeqTransformer
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.env_runner.robomimic_lowdim_moh_runner import RobomimicLowdimMoHRunner

OmegaConf.register_new_resolver("eval", eval, replace=True)


def load_policy(ckpt, device):
    payload = torch.load(open(ckpt, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg, output_dir='/tmp/sa_eval_ws')
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    policy = workspace.ema_model if cfg.training.use_ema else workspace.model
    policy.to(torch.device(device))
    policy.eval()
    return policy, cfg


@click.command()
@click.option('--run_dir', required=True)
@click.option('--sa_dir', required=True,
              help='dir with seq2seq_attention_estimator.pth + normalizer.pth')
@click.option('--output_dir', default=None)
@click.option('-d', '--device', default='cuda:0')
@click.option('--target', default=32.0)
@click.option('--tol', default=0.5)
@click.option('--n_probe', default=10)
@click.option('--n_final', default=50)
@click.option('--top_k', default=5)
@click.option('--attention_exponent', default=1.0)
@click.option('--min_n_action_steps', default=2)
def main(run_dir, sa_dir, output_dir, device, target, tol, n_probe, n_final,
         top_k, attention_exponent, min_n_action_steps):
    run_dir = pathlib.Path(run_dir)
    if output_dir is None:
        output_dir = run_dir / 'sa_eval'
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ckpts = []
    for p in (run_dir / 'checkpoints').glob('epoch=*.ckpt'):
        m = re.search(r'test_mean_score=([\d.]+?)\.ckpt', p.name)
        if m:
            ckpts.append((float(m.group(1)), p))
    ckpts.sort(key=lambda t: -t[0])
    ckpts = ckpts[:top_k]
    assert len(ckpts) > 0, f"no scored checkpoints in {run_dir}"
    best_ckpt = ckpts[0][1]
    print(f"[sa-sweep] {len(ckpts)} checkpoints, best: {best_ckpt.name}")

    policy, cfg = load_policy(best_ckpt, device)

    # attention estimator + its normalizer (policy-independent, trained on
    # demo data with CGE-derived SA labels; seq_len must equal policy horizon)
    obs_dim = cfg.task.obs_dim
    estimator = Seq2SeqTransformer(
        obs_dim=obs_dim * cfg.n_obs_steps,
        action_dim=cfg.task.action_dim,
        seq_len=cfg.policy.horizon)
    estimator.load_state_dict(torch.load(
        os.path.join(sa_dir, 'seq2seq_attention_estimator.pth'),
        map_location='cpu', weights_only=False))
    estimator.to(torch.device(device)).eval()
    sa_normalizer = LinearNormalizer()
    sa_normalizer.load_state_dict(torch.load(
        os.path.join(sa_dir, 'normalizer.pth'),
        map_location='cpu', weights_only=False))
    sa_normalizer.to(torch.device(device))

    def sa_cfg(c_att):
        return dict(estimator=estimator, normalizer=sa_normalizer,
                    c_att=c_att, attention_exponent=attention_exponent,
                    min_n_action_steps=min_n_action_steps)

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

    # ---- 1) probe c_att on best checkpoint ----
    def probe(c):
        log = runner.run(policy, mode='sa', n_episodes=n_probe, sa=sa_cfg(c))
        h = log['test/avg_exec_horizon']
        print(f"[probe] c_att={c:.2f} -> avg_exec_horizon={h:.2f} "
              f"(score {log['test/mean_score']:.2f})")
        return h

    grid = [20.0, 50.0, 100.0, 200.0, 400.0, 800.0]
    probes = [(c, probe(c)) for c in grid]
    c_star = None
    for c, h in probes:
        if abs(h - target) <= tol:
            c_star = c
            break
    if c_star is None:
        below = [(c, h) for c, h in probes if h < target]
        above = [(c, h) for c, h in probes if h > target]
        if not above:
            c_lo = max(c for c, _ in probes); c_hi = c_lo * 3
        elif not below:
            c_hi = min(c for c, _ in probes); c_lo = c_hi / 3
        else:
            c_lo = max(below, key=lambda t: t[1])[0]
            c_hi = min(above, key=lambda t: -t[1])[0]
        for _ in range(6):
            c_mid = 0.5 * (c_lo + c_hi)
            h = probe(c_mid)
            probes.append((c_mid, h))
            if abs(h - target) <= tol:
                c_star = c_mid
                break
            if h < target:
                c_lo = c_mid
            else:
                c_hi = c_mid
        if c_star is None:
            c_star = min(probes, key=lambda t: abs(t[1] - target))[0]
    print(f"[sa-sweep] selected c_att* = {c_star}")

    # ---- 2) final evals on top-k checkpoints ----
    results = {'sa': [], 'fixed': []}
    for score, ckpt in ckpts:
        policy, _ = load_policy(ckpt, device)

        log_s = runner.run(policy, mode='sa', sa=sa_cfg(c_star))
        log_s.update(checkpoint=str(ckpt), c_att=c_star)
        json.dump(log_s, open(output_dir / f'sa_{ckpt.stem}.json', 'w'),
                  indent=2, sort_keys=True)
        results['sa'].append(log_s)
        print(f"[final] {ckpt.name} +SA    score {log_s['test/mean_score']:.3f} "
              f"avgTa {log_s['test/avg_exec_horizon']:.1f}")

        log_f = runner.run(policy, mode='fixed', n_action_steps=int(target))
        log_f.update(checkpoint=str(ckpt))
        json.dump(log_f, open(output_dir / f'fixed_{ckpt.stem}.json', 'w'),
                  indent=2, sort_keys=True)
        results['fixed'].append(log_f)
        print(f"[final] {ckpt.name} fixed  score {log_f['test/mean_score']:.3f}")

    # ---- 3) summary ----
    summary = {'c_att_star': c_star, 'target': target, 'n_final': n_final,
               'attention_exponent': attention_exponent,
               'min_n_action_steps': min_n_action_steps,
               'sa_dir': str(sa_dir), 'probes': probes,
               'checkpoints': [str(c) for _, c in ckpts]}
    for mode in ('sa', 'fixed'):
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
    print(f"[sa-sweep] DONE  +SA {summary['sa']['mean_score']:.3f}"
          f"±{summary['sa']['std_score']:.3f} (avgTa "
          f"{summary['sa']['avg_exec_horizon']:.1f})  |  fixed "
          f"{summary['fixed']['mean_score']:.3f}±{summary['fixed']['std_score']:.3f}")


if __name__ == '__main__':
    main()
