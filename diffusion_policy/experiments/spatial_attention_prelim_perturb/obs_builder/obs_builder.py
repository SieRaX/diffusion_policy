"""Build a single (non-vectorized) robomimic env from checkpoint-derived config,
reset it to a stored demo state (optionally perturbed), and extract the policy
input — with symmetric history construction.

HARD REQUIREMENT: the nominal and perturbed inputs must be built by the SAME
history rule. `build_input` is called with the same `history_mode` for both sides;
the only difference is the per-frame `applier` (a perturbation for the perturbed
side, nominal for the nominal side). See `history_mode` handling below.

history_mode:
  - tile_perturbed (DEFAULT): extract ONE frame at t (nominal or perturbed) and tile
    it across n_obs_steps (velocity content zero on both sides -> cancels).
  - consistent_perturbed: apply the same perturbation to the last n_obs_steps frames
    (grasp rule re-evaluated per frame); t< n_obs_steps-1 is front-clamped/tiled.
  - current_frame_only: history frames NOMINAL, only the last frame gets the applier
    (ablation that injects apparent object velocity).
"""
import collections
import numpy as np
import torch

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.obs_utils as ObsUtils

from diffusion_policy.env.robomimic.robomimic_lowdim_wrapper import RobomimicLowdimWrapper
from diffusion_policy.env.robomimic.robomimic_image_wrapper import RobomimicImageWrapper
from diffusion_policy.experiments.spatial_attention_prelim_perturb.perturbation.grasp import is_grasped
from diffusion_policy.experiments.spatial_attention_prelim_perturb.perturbation.state_perturb import (
    apply_realization,
)

HISTORY_MODES = ('tile_perturbed', 'consistent_perturbed', 'current_frame_only')


# ---------------------------------------------------------------- env build
def _first_rgb_key(shape_meta):
    for key, attr in shape_meta['obs'].items():
        if attr.get('type', 'low_dim') == 'rgb':
            return key
    return 'agentview_image'


def build_env(dataset_path, variant, obs_keys=None, shape_meta=None, abs_action=False):
    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path)
    if abs_action:
        env_meta['env_kwargs']['controller_configs']['control_delta'] = False
    if variant == 'lowdim':
        assert obs_keys is not None
        ObsUtils.initialize_obs_modality_mapping_from_dict({'low_dim': list(obs_keys)})
        renv = EnvUtils.create_env_from_metadata(
            env_meta=env_meta, render=False, render_offscreen=False, use_image_obs=False)
        wrapper = RobomimicLowdimWrapper(env=renv, obs_keys=list(obs_keys))
    else:
        assert shape_meta is not None
        env_meta['env_kwargs']['use_object_obs'] = False
        modality = collections.defaultdict(list)
        for key, attr in shape_meta['obs'].items():
            modality[attr.get('type', 'low_dim')].append(key)
        ObsUtils.initialize_obs_modality_mapping_from_dict(dict(modality))
        renv = EnvUtils.create_env_from_metadata(
            env_meta=env_meta, render=False, render_offscreen=False, use_image_obs=True)
        renv.env.hard_reset = False
        wrapper = RobomimicImageWrapper(env=renv, shape_meta=shape_meta,
                                        render_obs_key=_first_rgb_key(shape_meta))
        wrapper.env.reset()  # one-time full reset so rendering is correct before reset_to
    return wrapper


# ---------------------------------------------------------------- obs helpers
def _obs_to_torch(obs):
    """wrapper.get_observation() output -> torch float32. lowdim: (obs_dim,);
    image: dict {key: (C,H,W)/(dim,)}."""
    if isinstance(obs, dict):
        return {k: torch.as_tensor(np.asarray(v), dtype=torch.float32) for k, v in obs.items()}
    return torch.as_tensor(np.asarray(obs), dtype=torch.float32)


def _is_dict(x):
    return isinstance(x, dict)


def _tile(frame, n):
    if _is_dict(frame):
        return {k: torch.stack([v] * n, dim=0) for k, v in frame.items()}  # (n, ...)
    return torch.stack([frame] * n, dim=0)


def _stack(frames):
    if _is_dict(frames[0]):
        return {k: torch.stack([f[k] for f in frames], dim=0) for k in frames[0]}
    return torch.stack(frames, dim=0)


def _batch(stacked):
    if _is_dict(stacked):
        return {k: v.unsqueeze(0) for k, v in stacked.items()}
    return stacked.unsqueeze(0)


def _to_obs_dict(batched, variant):
    """predict_action-ready obs_dict. lowdim -> {'obs': (1,To,obs_dim)}; image -> the dict."""
    if variant == 'lowdim':
        return {'obs': batched}
    return batched


# ---------------------------------------------------------------- perturbation applier
def make_perturb_applier(bodies, realization, grasp_qpos_threshold,
                         perturb_targets, settle_steps):
    """A callable (sim, rs_env) -> grasp_flags that detects grasp per body and
    applies `realization`. `realization=None` => nominal (forward only)."""
    def apply(sim, rs_env):
        if realization is None:
            sim.forward()
            return None
        flags = [is_grasped(rs_env, sim, b, grasp_qpos_threshold) for b in bodies]
        apply_realization(sim, bodies, flags, realization, rs_env=rs_env,
                          perturb_targets=perturb_targets, settle_steps=settle_steps)
        return flags
    return apply


def detect_grasp_flags(wrapper, states, t, bodies, grasp_qpos_threshold):
    """Nominal per-body grasp flags at timestep t (for recording / shading)."""
    wrapper.env.reset_to({'states': states[t]})
    sim, rs_env = wrapper.env.env.sim, wrapper.env.env
    sim.forward()
    return [is_grasped(rs_env, sim, b, grasp_qpos_threshold) for b in bodies]


def lowdim_obs_key_ranges(wrapper):
    """Map each low_dim obs key to its ``[start, end)`` index range in the
    concatenated observation vector (concatenation order = ``wrapper.obs_keys``).
    Used to resolve obs-key names in the obs_noise backend's ``noise_dim_mask``."""
    raw = wrapper.env.get_observation()
    ranges = {}
    cur = 0
    for key in wrapper.obs_keys:
        dim = int(np.asarray(raw[key]).reshape(-1).shape[0])
        ranges[key] = (cur, cur + dim)
        cur += dim
    return ranges


# ---------------------------------------------------------------- input builder
def _extract_frame(wrapper, states, idx, applier):
    wrapper.env.reset_to({'states': states[idx]})
    sim, rs_env = wrapper.env.env.sim, wrapper.env.env
    if applier is None:
        sim.forward()
    else:
        applier(sim, rs_env)
    return _obs_to_torch(wrapper.get_observation())


def build_input(wrapper, variant, n_obs_steps, history_mode, states, t, applier):
    """Return a predict_action-ready obs_dict (batched to 1) for one input.
    `applier=None` builds a pure-nominal input; a perturb applier builds a
    perturbed input. Both sides call this with the SAME history_mode."""
    assert history_mode in HISTORY_MODES, history_mode
    To = int(n_obs_steps)

    if history_mode == 'tile_perturbed':
        frame = _extract_frame(wrapper, states, t, applier)
        stacked = _tile(frame, To)

    elif history_mode == 'consistent_perturbed':
        idxs = [max(0, t - (To - 1) + i) for i in range(To)]  # front-clamped
        frames = [_extract_frame(wrapper, states, i, applier) for i in idxs]
        stacked = _stack(frames)

    elif history_mode == 'current_frame_only':
        # history frames NOMINAL, last frame gets the applier (ablation)
        idxs = [max(0, t - (To - 1) + i) for i in range(To)]
        frames = []
        for i, idx in enumerate(idxs):
            frame_applier = applier if i == (To - 1) else None
            frames.append(_extract_frame(wrapper, states, idx, frame_applier))
        stacked = _stack(frames)

    return _to_obs_dict(_batch(stacked), variant)
