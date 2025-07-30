"""
Usage:
python eval.py --checkpoint data/image/pusht/diffusion_policy_cnn/train_0/checkpoints/latest.ckpt -o data/pusht_eval_output
"""

import sys
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import hydra
import hydra.utils
from omegaconf import OmegaConf
import pathlib
from diffusion_policy.workspace.base_workspace import BaseWorkspace
import os
import shutil
import pathlib
import click
import time
import hydra
import torch
import dill
import wandb
import json
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.model.transformer import Seq2SeqTransformer
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.diffusion_unet_lowdim_policy import DiffusionUnetLowdimPolicy
from omegaconf import OmegaConf

# allows arbitrary python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        'diffusion_policy','config', 'eval_config'))
)

def main(cfg: OmegaConf):
    # resolve immediately so all the ${now:} resolvers
    # will use the same time.
    OmegaConf.resolve(cfg)
    
    checkpoint = cfg.checkpoint
    output_dir = cfg.output_dir
    normalizer_dir = cfg.normalizer_dir
    attention_estimator_dir = cfg.attention_estimator_dir
    device = cfg.device
    disturbance_cfg = cfg.disturbance
    init_catt = cfg.init_catt
    init_dcatt = cfg.init_dcatt

    # start_log.json works as an indicator that the evaluation is under process.
    # This will prevent other process to work on the same directory, or else, it might overwrite the exsiting logs.
    if os.path.exists(os.path.join(output_dir, "start_log.json")):
        click.confirm(f"Output path {output_dir} already exists! Overwrite?", abort=True)
    start_log = dict()
    start_log['command'] = ' '.join(sys.argv)
    json.dump(start_log, open(os.path.join(output_dir, "start_log.json"), "w"), indent=2, sort_keys=True)
    
    # load checkpoint
    payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
    checkpoint_cfg = payload['cfg']
    # checkpoint_cfg.policy.n_action_steps = checkpoint_cfg.policy.horizon - checkpoint_cfg.policy.n_obs_steps
    # checkpoint_cfg.task.env_runner.n_train = 2
    # checkpoint_cfg.task.env_runner.n_train_vis = 1
    # Change the noise_scheduler to ours
    # checkpoint_cfg.policy.noise_scheduler._target_ = 'diffusion_policy.schedulers.scheduling_ddpm.DDPMScheduler'
    # Change the env runner to ours
    # checkpoint_cfg.task.env_runner._target_ = 'diffusion_policy.env_runner.pusht_keypoints_likelihood_runner.PushTKeypointsLikelihoodRunner' # Have to change to pust ahc runner...
    # checkpoint_cfg.task.env_runner._target_ = 'diffusion_policy.env_runner.robomimic_lowdim_AHC_runner_by_seed.RobomimicLowdimAHCRunner'
    cls = hydra.utils.get_class(checkpoint_cfg._target_)
    workspace = cls(checkpoint_cfg, output_dir=output_dir)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    
    # get policy from workspace
    policy = workspace.model
    if checkpoint_cfg.training.use_ema:
        policy = workspace.ema_model

    normalizer = LinearNormalizer()
    normalizer.load_state_dict(torch.load(normalizer_dir, weights_only=False))
    
    attention_estimator = Seq2SeqTransformer(obs_dim=checkpoint_cfg.obs_dim*2, action_dim=checkpoint_cfg.action_dim, seq_len=checkpoint_cfg.policy.horizon)
    attention_estimator.load_state_dict(torch.load(attention_estimator_dir, weights_only=False))
    
    device = torch.device(device)
    policy.to(device)
    policy.eval()
    attention_estimator.to(device)
    attention_estimator.eval()
    
    assert type(policy) == DiffusionUnetLowdimPolicy
    assert type(attention_estimator) == Seq2SeqTransformer
    
    disturbance_generator = hydra.utils.instantiate(
        disturbance_cfg.generator)
    env_runner_cfg = checkpoint_cfg.task.env_runner
    OmegaConf.set_struct(env_runner_cfg, False)
    env_runner_cfg.update(cfg.env_runner)
    env_runner = hydra.utils.instantiate(
        env_runner_cfg,
        output_dir=output_dir,
        disturbance_generator=disturbance_generator
        )
    start_time = time.time()
    runner_log = env_runner.run(policy, attention_estimator, normalizer, init_catt=init_catt, init_dcatt=init_dcatt)
    elapsed_time = time.time() - start_time
    # dump log to json
    json_log = dict()
    for key, value in runner_log.items():
        if isinstance(value, wandb.sdk.data_types.video.Video):
            json_log[key] = value._path
        else:
            json_log[key] = value
    json_log['command'] = ' '.join(sys.argv)
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = int(elapsed_time % 60)
    json_log['elapsed_time'] = f"{hours:02d}h {minutes:02d}m {seconds:02d}s"
    out_path = os.path.join(output_dir, 'eval_log.json')
    json.dump(json_log, open(out_path, 'w'), indent=2, sort_keys=True)

if __name__ == '__main__':
    main()
