"""
Usage:
    python eval_entropy_by_seed.py \
        -c /path/to/checkpoint.ckpt \
        -o /path/to/output_dir \
        -d cuda:0 \
        -v 4 \
        -n 64 \
        -t 400
"""

import sys
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import os
import shutil
import pathlib
import click
import hydra
import torch
import dill
import wandb
import json
from diffusion_policy.workspace.base_workspace import BaseWorkspace

from omegaconf import OmegaConf

# Prevent deadlock in fork-based multiprocessing with Mujoco/EGL contexts.
import multiprocessing as mp
mp.set_start_method("spawn", force=True)


@click.command()
@click.option('-c', '--checkpoint', required=True,
              help='Path to training checkpoint (.ckpt)')
@click.option('-o', '--output_dir', required=True,
              help='Directory to write eval results and videos')
@click.option('-d', '--device', default='cuda:0',
              help='Torch device string')
@click.option('-v', '--n_test_vis', default=4, type=int,
              help='Number of test episodes to render video for')
@click.option('-n', '--n_action_samples', default=64, type=int,
              help='Number of action samples for entropy estimation')
@click.option('-t', '--max_steps', default=400, type=int,
              help='Maximum steps per episode')
@click.option('-a', '--n_action_steps', default=None, type=int,
              help='Override policy n_action_steps (default: horizon - n_obs_steps)')
@click.option('-r', '--env_runner', default=(
    'diffusion_policy.env_runner.'
    'robomimic_lowdim_entropy_avg_record_runner.'
    'RobomimicLowdimEntropyAvgRecordRunner'),
              help='Fully-qualified env runner class')
def main(checkpoint, output_dir, device, n_test_vis, n_action_samples,
         max_steps, n_action_steps, env_runner):
    if os.path.exists(output_dir):
        click.confirm(
            f"Output path {output_dir} already exists! Overwrite?",
            abort=True)
        shutil.rmtree(output_dir)
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    # ---- load checkpoint --------------------------------------------------
    payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
    cfg = payload['cfg']

    # override policy n_action_steps
    if n_action_steps is not None:
        cfg.policy.n_action_steps = n_action_steps
    else:
        # use full prediction horizon so entropy covers all h
        cfg.policy.n_action_steps = cfg.policy.horizon - cfg.policy.n_obs_steps

    # override env_runner target to our entropy recording runner
    cfg.task.env_runner._target_ = env_runner
    cfg.task.env_runner.n_test = n_test_vis
    cfg.task.env_runner.n_test_vis = n_test_vis
    cfg.task.env_runner.max_steps = max_steps

    # inject n_action_samples (may not exist in original config)
    OmegaConf.set_struct(cfg.task.env_runner, False)
    cfg.task.env_runner.n_action_samples = n_action_samples
    OmegaConf.set_struct(cfg.task.env_runner, True)

    # Change the noise_scheduler to DDPM (ensures stochastic sampling).
    # DDIM-trained checkpoints carry DDIM-only ctor kwargs; drop them so the
    # DDPMScheduler swap doesn't raise TypeError.
    cfg.policy.noise_scheduler._target_ = \
        'diffusion_policy.schedulers.scheduling_ddpm.DDPMScheduler'
    OmegaConf.set_struct(cfg.policy.noise_scheduler, False)
    for ddim_only_key in ('set_alpha_to_one', 'steps_offset'):
        cfg.policy.noise_scheduler.pop(ddim_only_key, None)
    OmegaConf.set_struct(cfg.policy.noise_scheduler, True)

    # ---- build workspace & load weights -----------------------------------
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, output_dir=output_dir)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    # get policy from workspace
    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model

    device = torch.device(device)
    policy.to(device)
    policy.eval()

    # ---- instantiate env runner -------------------------------------------
    env_runner_instance = hydra.utils.instantiate(
        cfg.task.env_runner,
        output_dir=output_dir)

    # ---- run evaluation ---------------------------------------------------
    runner_log = env_runner_instance.run(policy)

    # ---- dump log to json -------------------------------------------------
    json_log = dict()
    for key, value in runner_log.items():
        if isinstance(value, wandb.sdk.data_types.video.Video):
            json_log[key] = value._path
        else:
            json_log[key] = value
    json_log['command'] = ' '.join(sys.argv)
    out_path = os.path.join(output_dir, 'eval_log.json')
    json.dump(json_log, open(out_path, 'w'), indent=2, sort_keys=True)

    print(f"\n✓ Evaluation complete. Results saved to {output_dir}")
    print(f"  - eval_log.json")
    print(f"  - entropy_histories.pickle")
    print(f"  - media/*.mp4 (videos with entropy graphs)")


if __name__ == '__main__':
    main()
