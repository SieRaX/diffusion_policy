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
import numpy as np
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.model.transformer import Seq2SeqTransformer, Seq2SeqTransformerWithVAE
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.diffusion_unet_lowdim_policy import DiffusionUnetLowdimPolicy
from diffusion_policy.policy.diffusion_unet_hybrid_image_policy import DiffusionUnetHybridImagePolicy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.policy.diffusion_unet_hybrid_image_policy import DiffusionUnetHybridImagePolicy
from omegaconf import OmegaConf

import time


def main():
    
    """
    Loading the modelss
    """
    checkpoint_path = "outputs_HDD3/lift_lowdim_ph_reproduction/train_by_seed_ddpm/seed_0_2025.07.30-23.38.46_train_diffusion_unet_lowdim_lift_lowdim_cnn_32/checkpoints/epoch=0100-test_mean_score=1.000.ckpt"
    attention_estimator_dir = "outputs_HDD3/cge/lift_lowdim_abs_ph/train_by_seed/seed_0_2025.08.25_18.54.37_train_conditional_gradient_lowdim_lift_lowdim_abs/dataset_with_spatial_attention_attention_at_info_time_0.1/seq2seq_attention_estimator.pth"
    normalizer_dir = "outputs_HDD3/cge/lift_lowdim_abs_ph/train_by_seed/seed_0_2025.08.25_18.54.37_train_conditional_gradient_lowdim_lift_lowdim_abs/dataset_with_spatial_attention_attention_at_info_time_0.1/normalizer.pth"
    device = "cuda:0"
    attention_exponent = 3.0
    c_att = 1000.0
    
    # Load the policy and attention estimator
    payload = torch.load(open(checkpoint_path, 'rb'), pickle_module=dill)
    checkpoint_cfg = payload['cfg']
    cls = hydra.utils.get_class(checkpoint_cfg._target_)
    workspace = cls(checkpoint_cfg)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    
    policy = workspace.model
    if checkpoint_cfg.training.use_ema:
        policy = workspace.ema_model
    policy.n_action_steps = checkpoint_cfg.policy.horizon - checkpoint_cfg.policy.n_obs_steps
    # Make sure that policy samples maximum length of action chunk.
    
    attention_normalizer = LinearNormalizer()
    attention_normalizer.load_state_dict(torch.load(normalizer_dir, weights_only=False))

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
        attention_estimator.load_state_dict(torch.load(attention_estimator_dir, weights_only=False, map_location="cpu"))
    
    device = torch.device(device)
    policy.to(device)
    policy.eval()
    attention_estimator.to(device)
    attention_estimator.eval()
    
    assert type(policy) in [DiffusionUnetLowdimPolicy, DiffusionUnetHybridImagePolicy], f"Policy type: {type(policy)} it should be in [DiffusionUnetLowdimPolicy, DiffusionUnetHybridImagePolicy]"
    assert type(attention_estimator) == Seq2SeqTransformer or type(attention_estimator) == Seq2SeqTransformerWithVAE
    
    while True:
        """
        Sampling Action Chunk
        """
        obs = [[ # obs_dim=19 (object : 10 (abs pos, abs quat, relative pos endeffector), eef_pos:3 , eef_quat:4, gripper_qpos:2)
            [1.4783e-02, -3.4025e-03,  8.2118e-01,  0.0000e+00,  0.0000e+00,
             8.9584e-01, -4.4437e-01,  1.6002e-02,  3.3919e-03,  4.1289e-02,
             3.0785e-02, -1.0596e-05,  8.6247e-01,  
             9.8911e-01, -1.4342e-01, 3.3065e-02, -1.7123e-03,  
             3.9458e-02, -3.9427e-02],
            [1.4783e-02, -3.4025e-03,  8.2118e-01,  0.0000e+00,  0.0000e+00,
             8.9584e-01, -4.4437e-01,  1.6002e-02,  3.3919e-03,  4.1289e-02,
             3.0785e-02, -1.0596e-05,  8.6247e-01,  
             9.8911e-01, -1.4342e-01,3.3065e-02, -1.7123e-03, 
             3.9458e-02, -3.9427e-02]
        ]]
        obs = np.array(obs).astype(np.float32) 
        
        np_obs_dict = {
            'obs': obs # obs_dim = (batch, obs_dim, obs_steps)
        }
        
        # device transfer
        obs_dict = dict_apply(np_obs_dict, 
            lambda x: torch.from_numpy(x).to(
                device=device))

        # run policy
        with torch.no_grad():
            action_dict = policy.predict_action(obs_dict)
        
        # device_transfer
        np_action_dict = dict_apply(action_dict,
            lambda x: x.detach().to('cpu').numpy())
        
        # extract action
        np_action = np_action_dict['action']
        print(f"action_shape: {np_action.shape}")
        
        
        """
        Estimating horizon length
        """
        with torch.no_grad():
            # obs_dict_with_dummy = torch.cat([obs_dict['obs'], torch.zeros_like(obs_dict['obs'][..., -1:]).to(device=device)], dim=-1)
            nobs = attention_normalizer['obs'].normalize(obs_dict['obs']).to(device=device)
            naction = attention_normalizer['action'].normalize(action_dict['action_pred']).to(device=device)
            output = attention_estimator(nobs.reshape(nobs.shape[0], -1), naction)
            attention_pred = attention_normalizer['spatial_attention'].unnormalize(output).detach().cpu().squeeze(-1)
            attention_pred = torch.pow(attention_pred, attention_exponent)
            attention_pred_cumsum = torch.cumsum(attention_pred, dim=-1)
            
            # Create a mask where a > b
            mask = attention_pred_cumsum > c_att

            # Use torch.argmax on the mask to get the first index where condition is True
            # But be careful: if no element is > b, argmax will return 0, which may be incorrect
            # So we mask out invalid rows later
            horizon_idx = mask.float().cumsum(dim=1).eq(1).float().argmax(dim=1).item()
        
        print(f"horizon_idx: {horizon_idx}")
        print(f"final action shape: {np_action[:, :horizon_idx].shape}")
        print()
        
        time.sleep(1)
    
    

if __name__ == "__main__":
    main()