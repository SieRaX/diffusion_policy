"""Dataset-driven state sourcing (NO rollout), mirroring the prelim experiment.

- `enumerate_dataset` lists (episode, timestep) over the FULL dataset (probe pool) or a
  subset of episodes (query set), with stride.
- `ObsSource` builds the predict_action-ready observation at any (episode, timestep) via
  the prelim `obs_builder` (env.reset_to → forward → get_observation → history_mode), for
  low_dim and image alike. It also exposes the single-frame low_dim obs vector used as the
  similarity-space vector.
- `GTActions` provides the query's demonstrator action chunk aligned to the policy's chunk
  indexing (chunk index To-1 == env timestep t) in NORMALIZED action space, from the
  checkpoint dataset's replay buffer (rotation-transformed actions for all demos).
"""
import h5py
import numpy as np
import torch

from diffusion_policy.experiments.spatial_attention_prelim_perturb.obs_builder import obs_builder


def enumerate_dataset(dataset_path, stride, episodes=None):
    """Return (list of (episode, timestep), n_demos). episodes=None -> all demos."""
    with h5py.File(dataset_path, 'r') as f:
        n = len(f['data'])
        eps = range(n) if episodes is None else list(episodes)
        pool = []
        for e in eps:
            if e < 0 or e >= n:
                raise ValueError(f"episode {e} out of range [0,{n})")
            L = f[f'data/demo_{int(e)}/states'].shape[0]
            for t in range(0, L, int(stride)):
                pool.append((int(e), int(t)))
    return pool, n


class ObsSource:
    def __init__(self, dataset_path, variant, obs_keys, shape_meta, abs_action,
                 history_mode, n_obs_steps):
        self.wrapper = obs_builder.build_env(
            dataset_path, variant, obs_keys=obs_keys, shape_meta=shape_meta,
            abs_action=abs_action)
        self.variant = variant
        self.history_mode = history_mode
        self.To = int(n_obs_steps)
        self._path = dataset_path
        self._states = {}

    def states(self, episode):
        e = int(episode)
        if e not in self._states:
            with h5py.File(self._path, 'r') as f:
                self._states[e] = f[f'data/demo_{e}/states'][:]
        return self._states[e]

    def obs_at(self, episode, timestep):
        """predict_action-ready obs_dict (batched to 1) with the configured history_mode."""
        return obs_builder.build_input(
            self.wrapper, self.variant, self.To, self.history_mode,
            self.states(episode), int(timestep), applier=None)

    def frame_vector(self, episode, timestep):
        """Single-frame low_dim observation vector at (episode, timestep) — the similarity
        vector for low_dim runs (state ≡ latent)."""
        assert self.variant == 'lowdim', "frame_vector is low_dim only"
        frame = obs_builder._extract_frame(self.wrapper, self.states(episode), int(timestep), None)
        return frame.numpy().reshape(-1)


class GTActions:
    """Normalized demonstrator action chunk a_GT_norm(episode, timestep) from the dataset
    replay buffer, aligned so chunk index To-1 == env timestep t (edge-clamped at episode
    boundaries, mirroring the SequenceSampler pad_before=To-1 convention)."""
    def __init__(self, replay_action, episode_ends, n_obs_steps, horizon, action_normalizer):
        self.actions = np.asarray(replay_action)              # (total, Da)
        self.episode_ends = np.asarray(episode_ends, dtype=np.int64)
        self.ep_starts = np.concatenate([[0], self.episode_ends[:-1]]).astype(np.int64)
        self.To = int(n_obs_steps)
        self.H = int(horizon)
        self.normalizer = action_normalizer                   # policy.normalizer['action']

    def raw_chunk(self, episode, timestep):
        s = int(self.ep_starts[episode]); e = int(self.episode_ends[episode])
        base = s + int(timestep) - (self.To - 1)
        idxs = np.clip(np.arange(base, base + self.H), s, e - 1)
        return self.actions[idxs]                             # (H, Da)

    def norm_chunk(self, episode, timestep):
        raw = torch.as_tensor(self.raw_chunk(episode, timestep), dtype=torch.float32)
        return self.normalizer.normalize(raw)                 # (H, Da) normalized


def build_gt_actions(train_cfg, dataset_path_override, n_obs_steps, horizon, action_normalizer):
    """Instantiate the checkpoint's dataset (hydra) to obtain the rotation-transformed
    action array + episode boundaries, then wrap in GTActions. Reuse over reimplementation
    of the abs-action rotation transform."""
    import hydra
    from omegaconf import open_dict
    ds_cfg = train_cfg.task.dataset
    if dataset_path_override:
        with open_dict(ds_cfg):
            ds_cfg.dataset_path = dataset_path_override
    dataset = hydra.utils.instantiate(ds_cfg)
    rb = dataset.replay_buffer
    return GTActions(rb['action'][:], rb.episode_ends[:], n_obs_steps, horizon, action_normalizer)
