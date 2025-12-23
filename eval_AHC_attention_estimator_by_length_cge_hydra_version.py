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
from diffusion_policy.model.transformer import Seq2SeqTransformer, Seq2SeqTransformerWithVAE
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.diffusion_unet_lowdim_policy import DiffusionUnetLowdimPolicy
from diffusion_policy.policy.diffusion_unet_hybrid_image_policy import DiffusionUnetHybridImagePolicy
from omegaconf import OmegaConf

# This process is needed to prevent the deadlock caused when try Image based env runner.
'''
# This is the anser from chat gpt
AsyncVectorEnv uses multiprocessing (fork-based on Unix)
It creates subprocesses that inherit the parent’s memory state.
But Mujoco’s internal C/OpenGL contexts cannot be safely inherited. When a child process attempts to initialize a new MjSim (via robosuite.make()), it often deadlocks or hangs in the call to mujoco_py.MjSim(...).

EGL or OpenGL context initialization happens per process
Robosuite tries to create an offscreen renderer (EGL) or OpenGL context. In forked processes, this often causes the context creation call to block forever.

egl_probe and GPU device selection make it worse
The block might happen at egl_probe.get_available_devices() (or inside EGL initialization), especially if each process tries to access the same GPU EGL context.
'''

import multiprocessing as mp
mp.set_start_method("spawn", force=True)

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
    # checkpoint_cfg.task.env_runner.n_train = 20
    # checkpoint_cfg.task.env_runner.n_train_vis = 20
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
    policy.n_action_steps = checkpoint_cfg.policy.horizon - checkpoint_cfg.policy.n_obs_steps
    # Make sure that policy samples maximum length of action chunk.
    # In ADP runner, Attention module decides the length of action chunk.

    normalizer = LinearNormalizer()
    normalizer.load_state_dict(torch.load(normalizer_dir, weights_only=False))

    if 'obs_dim' in checkpoint_cfg:
        obs_dim = checkpoint_cfg.obs_dim
    else:
        OmegaConf.set_struct(checkpoint_cfg, False)
        from omegaconf import open_dict
        with open_dict(checkpoint_cfg):
            if 'image_feature_dim' not in checkpoint_cfg.policy:
                checkpoint_cfg.policy.image_feature_dim = 5

            if 'action_dim' not in checkpoint_cfg:
                checkpoint_cfg.action_dim = checkpoint_cfg.task.shape_meta.action.shape[0]
            obs_dim = checkpoint_cfg.policy.image_feature_dim * 2 + 9 ## This is only for robomimic tasks
    try:
        attention_estimator = Seq2SeqTransformer(obs_dim=obs_dim*2, action_dim=checkpoint_cfg.action_dim, seq_len=checkpoint_cfg.policy.horizon)
        attention_estimator.load_state_dict(torch.load(attention_estimator_dir, weights_only=False))
    except Exception as e:
        print("\033[33mError loading attention estimator, Switching to Seq2SeqTransformerWithVisionEncoder\033[0m")
        attention_estimator = Seq2SeqTransformerWithVAE(obs_dim=obs_dim*2, action_dim=checkpoint_cfg.action_dim, seq_len=checkpoint_cfg.policy.horizon)
        attention_estimator.load_state_dict(torch.load(attention_estimator_dir, weights_only=False))
    
    device = torch.device(device)
    policy.to(device)
    policy.eval()
    attention_estimator.to(device)
    attention_estimator.eval()
    
    assert type(policy) in [DiffusionUnetLowdimPolicy, DiffusionUnetHybridImagePolicy]
    assert type(attention_estimator) == Seq2SeqTransformer or type(attention_estimator) == Seq2SeqTransformerWithVAE
    
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
