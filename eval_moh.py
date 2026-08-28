"""
Evaluate a trained MoH-DP checkpoint with fixed-prefix or dynamic (cross-horizon
consensus) inference, following the SA paper's sequential seeded-episode protocol.

Usage (fixed prefix, Ta=32):
  python eval_moh.py -c <ckpt> -o <out_dir> --mode fixed --n_action_steps 32

Usage (dynamic; sweep scale_ratio to hit a target average executed horizon):
  python eval_moh.py -c <ckpt> -o <out_dir> --mode dynamic --scale_ratio 1.2
"""
import sys
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import os
import json
import pathlib
import click
import torch
import dill
import hydra
from omegaconf import OmegaConf

from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.env_runner.robomimic_lowdim_moh_runner import RobomimicLowdimMoHRunner

OmegaConf.register_new_resolver("eval", eval, replace=True)


@click.command()
@click.option('-c', '--checkpoint', required=True)
@click.option('-o', '--output_dir', required=True)
@click.option('-m', '--mode', type=click.Choice(['fixed', 'dynamic']), default='dynamic')
@click.option('-a', '--n_action_steps', default=8, help='prefix length for fixed mode')
@click.option('-r', '--scale_ratio', default=1.1, help='consensus threshold ratio for dynamic mode')
@click.option('--min_replan_steps', default=None, type=int)
@click.option('--min_active_horizons', default=None, type=int)
@click.option('-n', '--n_test', default=50)
@click.option('-s', '--test_start_seed', default=100000)
@click.option('--max_steps', default=None, type=int)
@click.option('-d', '--device', default='cuda:0')
@click.option('--tag', default='', help='label appended to the result filename')
def main(checkpoint, output_dir, mode, n_action_steps, scale_ratio,
         min_replan_steps, min_active_horizons, n_test, test_start_seed,
         max_steps, device, tag):
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg, output_dir=output_dir)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model
    policy.to(torch.device(device))
    policy.eval()

    policy.scale_ratio = scale_ratio
    if min_replan_steps is not None:
        policy.min_replan_steps = min_replan_steps
    if min_active_horizons is not None:
        policy.min_active_horizons = min_active_horizons

    runner = RobomimicLowdimMoHRunner(
        output_dir=output_dir,
        dataset_path=cfg.task.dataset.dataset_path,
        obs_keys=list(cfg.task.dataset.obs_keys),
        n_test=n_test,
        test_start_seed=test_start_seed,
        max_steps=max_steps if max_steps is not None
            else cfg.task.env_runner.max_steps,
        n_obs_steps=cfg.n_obs_steps,
        abs_action=cfg.task.dataset.abs_action,
    )

    log = runner.run(policy, mode=mode, n_action_steps=n_action_steps)
    log['checkpoint'] = checkpoint
    log['mode'] = mode
    log['n_action_steps'] = n_action_steps
    log['scale_ratio'] = scale_ratio
    log['min_replan_steps'] = policy.min_replan_steps
    log['min_active_horizons'] = policy.min_active_horizons
    log['n_test'] = n_test

    name = f"eval_{mode}"
    name += f"_Ta{n_action_steps}" if mode == 'fixed' else f"_r{scale_ratio}"
    if tag:
        name += f"_{tag}"
    out_path = os.path.join(output_dir, name + '.json')
    json.dump(log, open(out_path, 'w'), indent=2, sort_keys=True)
    print(f"[eval_moh] mode={mode} "
          f"{'Ta=' + str(n_action_steps) if mode == 'fixed' else 'r=' + str(scale_ratio)} "
          f"score={log['test/mean_score']:.3f} "
          f"avg_exec_horizon={log['test/avg_exec_horizon']:.2f} -> {out_path}")


if __name__ == '__main__':
    main()
