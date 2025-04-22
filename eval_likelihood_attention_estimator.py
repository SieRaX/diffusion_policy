"""
Usage:
python eval.py --checkpoint data/image/pusht/diffusion_policy_cnn/train_0/checkpoints/latest.ckpt -o data/pusht_eval_output
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
from diffusion_policy.model.transformer import Seq2SeqTransformer
from diffusion_policy.model.common.normalizer import LinearNormalizer

from omegaconf import OmegaConf

@click.command()
@click.option('-c', '--checkpoint', required=True)
@click.option('-o', '--output_dir', required=True)
@click.option('-d', '--device', default='cuda:0')
@click.option('-n', '--n_action_steps', default=8)
@click.option('-v', '--n_test_vis', default=4)
@click.option('-t', '--max_steps', default=70)
@click.option('-r', '--env_runner', default='diffusion_policy.env_runner.robomimic_lowdim_likelihood_disturbance_runner.RobomimicLowdimLikelihoodDisturbanceRunner')
def main(checkpoint, output_dir, device, n_action_steps, n_test_vis, max_steps, env_runner):
    if os.path.exists(output_dir):
        click.confirm(f"Output path {output_dir} already exists! Overwrite?", abort=True)
        shutil.rmtree(output_dir)
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # load checkpoint
    payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    cfg.policy.n_action_steps = n_action_steps
    cfg.task.env_runner.n_test = n_test_vis
    cfg.task.env_runner.n_test_vis = n_test_vis
    cfg.task.env_runner.n_train = 10
    cfg.task.env_runner.n_train_vis = 10
    # Change the noise_scheduler to ours
    cfg.policy.noise_scheduler._target_ = 'diffusion_policy.schedulers.scheduling_ddpm.DDPMScheduler'
    # Change the env runner to ours
    # cfg.task.env_runner._target_ = 'diffusion_policy.env_runner.pusht_keypoints_likelihood_runner.PushTKeypointsLikelihoodRunner'
    cfg.task.env_runner._target_ = env_runner
    cfg.task.env_runner.max_steps = max_steps
    # cfg.task.env_runner.test_start_seed = 100050
    
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, output_dir=output_dir)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    
    # get policy from workspace
    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model
        
    normalizer = LinearNormalizer()
    normalizer.load_state_dict(torch.load("data/outputs/lift_lowdim_ph_reproduction/horizon_16/2025.03.11/10.57.22_train_diffusion_unet_lowdim_lift_lowdim_transformer_128/checkpoints/normalizer.pth"))
    
    attention_estimator = Seq2SeqTransformer()
    attention_estimator.load_state_dict(torch.load("data/outputs/lift_lowdim_ph_reproduction/horizon_16/2025.03.11/10.57.22_train_diffusion_unet_lowdim_lift_lowdim_transformer_128/checkpoints/epoch=0200-transformer_attention_estimator.pth"))
    
    device = torch.device(device)
    policy.to(device)
    policy.eval()
    attention_estimator.to(device)
    attention_estimator.eval()
    
    # run eval
    env_runner = hydra.utils.instantiate(
        cfg.task.env_runner,
        output_dir=output_dir
    )
    runner_log = env_runner.run(policy, attention_estimator, normalizer)
    
    # dump log to json
    json_log = dict()
    for key, value in runner_log.items():
        if isinstance(value, wandb.sdk.data_types.video.Video):
            json_log[key] = value._path
        else:
            json_log[key] = value
    json_log['command'] = ' '.join(sys.argv)
    out_path = os.path.join(output_dir, 'eval_log.json')
    json.dump(json_log, open(out_path, 'w'), indent=2, sort_keys=True)

if __name__ == '__main__':
    main()
