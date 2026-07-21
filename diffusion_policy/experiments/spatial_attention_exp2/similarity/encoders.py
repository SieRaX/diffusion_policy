"""Encoders that turn an observation into a similarity-space latent vector.

- PolicyEncoder (default): the trained policy's conditioning vector
  `policy.fm_global_cond(obs)` (the exact global_cond the UNet consumes).
- VAEEncoder (optional): a separately trained conv image VAE
  (`conditional_gradient_estimator/VAE/vae.py`), deterministic mu of the current frame.
  Requires a checkpoint path; lazily imported (it lives outside the diffusion_policy package).
"""
import numpy as np
import torch

from diffusion_policy.common.pytorch_util import dict_apply


class PolicyEncoder:
    def __init__(self, policy):
        self.policy = policy

    @torch.no_grad()
    def encode(self, obs_dict):
        gc = self.policy.fm_global_cond(
            dict_apply(obs_dict, lambda x: x.to(self.policy.device)))   # (1, gdim)
        return gc[0].detach().cpu().numpy().reshape(-1)


class VAEEncoder:
    def __init__(self, vae, rgb_key, device):
        self.vae = vae
        self.rgb_key = rgb_key
        self.device = device

    @torch.no_grad()
    def encode(self, obs_dict):
        # current + previous frame stacked (VAE expects in_channels = 2*RGB); use last To frames
        img = obs_dict[self.rgb_key][0]                # (To, C, H, W)
        x = img.reshape(1, -1, img.shape[-2], img.shape[-1]).to(self.device).float()
        mu = self.vae.encoder(x)[0]                    # (1, latent_dim)
        return mu[0].detach().cpu().numpy().reshape(-1)


def make_encoder(policy, sim_cfg, rgb_key=None):
    enc = str(sim_cfg.get('encoder', 'policy_obs_encoder'))
    if enc == 'policy_obs_encoder':
        return PolicyEncoder(policy)
    if enc == 'vae':
        vcfg = sim_cfg.get('vae', {}) or {}
        ckpt = vcfg.get('checkpoint', None)
        if not ckpt:
            raise ValueError("similarity_space.latent encoder=vae requires vae.checkpoint "
                             "(TODO: no VAE checkpoint provided).")
        from conditional_gradient_estimator.VAE.vae import VAE  # lazy: outside the package
        vae = VAE()
        vae.load_state_dict(torch.load(ckpt, map_location='cpu'))
        vae.to(policy.device).eval()
        return VAEEncoder(vae, rgb_key, policy.device)
    raise ValueError(f"unknown similarity encoder {enc!r} (policy_obs_encoder | vae)")
