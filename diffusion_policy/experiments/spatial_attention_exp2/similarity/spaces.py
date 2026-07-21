"""Similarity space: map each probe-pool state to a vector, and assert the active space.

- low_dim  : the observation vector itself (state ≡ latent; the similarity_space group is
             inert). This is also the model's INPUT space, so noise probes (o+δ) are valid.
- latent   : the policy obs-encoder (or VAE) latent of the observation.
- pixel    : the current RGB frame, downsampled and flattened.

`metric=cosine` is realized by L2-normalizing the vectors upstream (Index stays Euclidean);
`default_radius` is then in the normalized space. TODO: native cosine in Index if needed.
"""
import numpy as np
import torch
import torch.nn.functional as F

from diffusion_policy.experiments.spatial_attention_exp2.similarity import encoders


def _first_rgb_key(shape_meta):
    for k, a in shape_meta['obs'].items():
        if a.get('type', 'low_dim') == 'rgb':
            return k
    return 'agentview_image'


def make_vectorizer(obs_source, policy, variant, sim_cfg, shape_meta=None):
    """Return a callable (episode, timestep) -> similarity vector (1D np array), and the
    resolved space name. Asserts the active space matches the checkpoint variant."""
    name = str(sim_cfg['name'])
    if variant == 'lowdim':
        # state ≡ latent; the similarity_space group is inert for low_dim.
        return (lambda ep, t: obs_source.frame_vector(ep, t)), 'state'
    # image
    if name == 'latent':
        rgb_key = _first_rgb_key(shape_meta)
        enc = encoders.make_encoder(policy, sim_cfg, rgb_key=rgb_key)
        return (lambda ep, t: enc.encode(obs_source.obs_at(ep, t))), 'latent'
    if name == 'pixel':
        rgb_key = _first_rgb_key(shape_meta)
        d = int(sim_cfg.get('downsample', 16))

        def vec(ep, t):
            img = obs_source.obs_at(ep, t)[rgb_key][0, -1]           # (C,H,W) current frame
            small = F.interpolate(img.unsqueeze(0).float(), size=(d, d),
                                  mode='bilinear', align_corners=False)
            return small.reshape(-1).cpu().numpy()
        return vec, 'pixel'
    raise ValueError(f"unknown similarity_space {name!r} for image variant (latent | pixel)")


def build_vectors(obs_source, policy, pool, variant, sim_cfg, shape_meta=None):
    """Compute the RAW (P, dim) similarity matrix over the probe pool. One-time; cache
    upstream. Cosine normalization (if requested) is applied by the index builder."""
    vec, space = make_vectorizer(obs_source, policy, variant, sim_cfg, shape_meta)
    rows = [np.asarray(vec(ep, t), dtype=np.float64).reshape(-1) for (ep, t) in pool]
    return np.stack(rows, axis=0), space
