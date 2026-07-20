"""Perturbation backends: pluggable strategies that turn a stored demo state at
timestep ``t`` into a nominal policy input and ``K`` perturbed inputs. Everything
DOWNSTREAM of these (coupled endpoint distance, CRN, figures, npz) is
backend-agnostic, so a run differing only in the backend is directly overlayable
(same checkpoint / demo / stride / eps0 seed).

Two backends, selected by the ``perturbation_backend`` hydra config group:

- ``SimStateBackend`` (``sim_state``, DEFAULT): grasp-aware SE(3) perturbation of
  the MuJoCo simulator state. Factored VERBATIM out of the original runner loop —
  same RNG stream (``np.random.default_rng([seed_perturb, t])`` then ``K``
  sequential ``sample_realization`` draws) — so its output is bitwise-unchanged
  (see ``tests/test_backend.py::test_sim_state_backend_parity``).

- ``ObsNoiseBackend`` (``obs_noise``, NEW, LOW_DIM only): additive Gaussian noise
  on the observation vector, ``o'_k = o + delta_k`` with ``delta_k ~ N(0, diag(sigma^2))``.
  No ``reset_to`` for the perturbed side, no grasp logic, no SE(3) machinery — so
  it is strictly cheaper than ``sim_state``. The ``K`` draws ``{delta_k}`` are
  FIXED (one fixed seed) and reused across ALL timesteps (CRN over perturbations,
  matching ``sim_state``'s per-timestep determinism).
"""
import numpy as np
import torch

from diffusion_policy.experiments.spatial_attention_prelim_perturb.obs_builder import obs_builder
from diffusion_policy.experiments.spatial_attention_prelim_perturb.perturbation.state_perturb import (
    sample_realization,
)


class PerturbationBackend:
    """Base interface. The nominal input is identical for every backend (it is
    unperturbed), so it lives here; subclasses only supply ``build_perturbed``."""
    name = 'base'

    def __init__(self, variant, n_obs_steps, history_mode):
        self.variant = variant
        self.To = int(n_obs_steps)
        self.history_mode = history_mode

    def build_nominal(self, wrapper, states, t):
        """predict_action-ready nominal obs_dict at timestep ``t`` (no perturbation)."""
        return obs_builder.build_input(
            wrapper, self.variant, self.To, self.history_mode, states, t, applier=None)

    def build_perturbed(self, wrapper, states, t, K):
        """List of ``K`` predict_action-ready perturbed obs_dicts at timestep ``t``."""
        raise NotImplementedError

    def metadata(self):
        """Backend-specific scalars/arrays merged into the saved npz."""
        return {}


class SimStateBackend(PerturbationBackend):
    """Grasp-aware SE(3) state perturbation (original behavior). ``build_perturbed``
    reproduces the pre-refactor runner loop exactly, including the RNG seeding and
    draw order, so results are bitwise-identical."""
    name = 'sim_state'

    def __init__(self, variant, n_obs_steps, history_mode, *, bodies,
                 sigma_pos_eef, sigma_rot_eef, sigma_pos_object, sigma_rot_object,
                 per_body_sigma, sigma_qpos, n_arm_joints, perturb_targets,
                 grasp_qpos_threshold, settle_steps, seed_perturb):
        super().__init__(variant, n_obs_steps, history_mode)
        self.bodies = bodies
        self.spe, self.sre = float(sigma_pos_eef), float(sigma_rot_eef)
        self.spo, self.sro = float(sigma_pos_object), float(sigma_rot_object)
        self.per_body_sigma = per_body_sigma
        self.sigma_qpos = float(sigma_qpos)
        self.n_arm = n_arm_joints
        self.perturb_targets = list(perturb_targets)
        self.grasp_threshold = float(grasp_qpos_threshold)
        self.settle_steps = int(settle_steps)
        self.seed_perturb = int(seed_perturb)

    def build_perturbed(self, wrapper, states, t, K):
        # EXACT original stream: fresh rng seeded [seed_perturb, t], then K
        # sequential sample_realization draws, one build_input each.
        rng_t = np.random.default_rng([self.seed_perturb, int(t)])
        out = []
        for _k in range(K):
            realization = sample_realization(
                self.bodies, rng_t, self.spe, self.sre, self.spo, self.sro,
                per_body_sigma=self.per_body_sigma,
                sigma_qpos=self.sigma_qpos, n_arm_joints=self.n_arm)
            applier = obs_builder.make_perturb_applier(
                self.bodies, realization, self.grasp_threshold,
                self.perturb_targets, self.settle_steps)
            out.append(obs_builder.build_input(
                wrapper, self.variant, self.To, self.history_mode,
                states, t, applier=applier))
        return out

    def metadata(self):
        return dict(
            perturb_targets=np.asarray(self.perturb_targets, dtype=object),
            sigma_pos_eef=self.spe, sigma_rot_eef=self.sre,
            sigma_pos_object=self.spo, sigma_rot_object=self.sro,
        )


def _resolve_dim_mask(noise_dim_mask, obs_key_ranges, obs_dim):
    """Bool ``(obs_dim,)`` mask of dims to noise. Entries may be int obs-dim
    indices or obs-key names (resolved to their ``[start,end)`` range via the
    checkpoint's obs metadata). ``None`` -> all dims."""
    if noise_dim_mask is None:
        return np.ones(obs_dim, dtype=bool)
    mask = np.zeros(obs_dim, dtype=bool)
    for entry in noise_dim_mask:
        is_int = isinstance(entry, (int, np.integer)) or (
            isinstance(entry, str) and entry.lstrip('-').isdigit())
        if is_int:
            idx = int(entry)
            if not (0 <= idx < obs_dim):
                raise ValueError(f"noise_dim_mask index {idx} out of range [0,{obs_dim}).")
            mask[idx] = True
        elif isinstance(entry, str):
            if entry not in obs_key_ranges:
                raise ValueError(f"noise_dim_mask key {entry!r} is not an obs key; "
                                 f"known keys: {list(obs_key_ranges)}.")
            s, e = obs_key_ranges[entry]
            mask[s:e] = True
        else:
            raise TypeError(f"noise_dim_mask entry {entry!r} must be an int index or key name.")
    if not mask.any():
        raise ValueError("noise_dim_mask resolved to an empty set of dims.")
    return mask


class ObsNoiseBackend(PerturbationBackend):
    """Additive Gaussian obs-vector noise (LOW_DIM only). The nominal obs is
    extracted exactly as the shared pipeline does (``build_nominal`` /
    ``_extract_frame``); the perturbed side just adds a fixed ``delta_k`` — no
    ``reset_to`` of a perturbed state, no grasp, no SE(3)."""
    name = 'obs_noise'

    def __init__(self, variant, n_obs_steps, history_mode, *, obs_dim, obs_std,
                 obs_key_ranges, K, noise_mode, noise_scale, sigma_abs,
                 noise_dim_mask, seed):
        super().__init__(variant, n_obs_steps, history_mode)
        if variant != 'lowdim':
            raise ValueError(
                "perturbation_backend=obs_noise supports LOW_DIM checkpoints only "
                f"(got variant={variant!r}). Use perturbation_backend=sim_state for image.")
        self.obs_dim = int(obs_dim)
        self.K = int(K)
        self.noise_mode = str(noise_mode)
        self.noise_scale = float(noise_scale)
        self.sigma_abs = float(sigma_abs)
        self.obs_std = np.asarray(obs_std, dtype=np.float64).reshape(-1)
        assert self.obs_std.shape[0] == self.obs_dim, \
            f"obs_std length {self.obs_std.shape[0]} != obs_dim {self.obs_dim}"

        if self.noise_mode == 'per_dim_std':
            sigma = self.noise_scale * self.obs_std        # sigma_d = noise_scale * std_d
        elif self.noise_mode == 'isotropic':
            sigma = np.full(self.obs_dim, self.sigma_abs, dtype=np.float64)
        else:
            raise ValueError(f"unknown noise_mode {self.noise_mode!r} (per_dim_std | isotropic).")

        self.mask = _resolve_dim_mask(noise_dim_mask, obs_key_ranges, self.obs_dim)
        self.sigma = sigma * self.mask                     # zero outside the mask
        # FIXED {delta_k}: drawn ONCE, reused across every timestep (CRN over perturbations).
        self.seed = int(seed)
        z = np.random.default_rng(self.seed).standard_normal((self.K, self.obs_dim))
        self.deltas = (z * self.sigma[None, :]).astype(np.float32)   # (K, obs_dim)

    def _nominal_frames(self, wrapper, states, t):
        """Nominal frame tensor(s) + per-slot flag of which slots receive delta,
        matching ``history_mode``. Extracted ONCE and reused across all K."""
        To = self.To
        if self.history_mode == 'tile_perturbed':
            # single frame, tiled To times; the SAME delta noises every slot
            return [obs_builder._extract_frame(wrapper, states, t, None)], [True]
        idxs = [max(0, t - (To - 1) + i) for i in range(To)]  # front-clamped
        frames = [obs_builder._extract_frame(wrapper, states, idx, None) for idx in idxs]
        if self.history_mode == 'consistent_perturbed':
            slots = [True] * To                                # same delta on every frame
        elif self.history_mode == 'current_frame_only':
            slots = [i == (To - 1) for i in range(To)]         # only the last frame
        else:
            raise ValueError(self.history_mode)
        return frames, slots

    def build_perturbed(self, wrapper, states, t, K):
        assert K == self.K, f"K mismatch: backend built for K={self.K}, got {K}."
        base_frames, slots = self._nominal_frames(wrapper, states, t)
        out = []
        for k in range(K):
            delta = torch.from_numpy(self.deltas[k])           # (obs_dim,)
            frames = [f + delta if slot else f for f, slot in zip(base_frames, slots)]
            if len(frames) > 1:
                stacked = obs_builder._stack(frames)
            else:
                stacked = obs_builder._tile(frames[0], self.To)  # tile the SAME delta'd frame
            out.append(obs_builder._to_obs_dict(obs_builder._batch(stacked), 'lowdim'))
        return out

    def metadata(self):
        return dict(
            noise_mode=str(self.noise_mode),
            noise_scale=np.float64(self.noise_scale),
            sigma_abs=np.float64(self.sigma_abs),
            noise_seed=np.int64(self.seed),
            obs_std=self.obs_std.astype(np.float64),       # per-dim std cache artifact
            noise_sigma=self.sigma.astype(np.float64),     # effective per-dim sigma (masked)
            noise_dim_mask=self.mask.astype(bool),
        )


def make_backend(cfg, r, policy, wrapper, bodies, rs_env):
    """Construct the perturbation backend selected by ``cfg.perturbation_backend``.
    Shared measurement params (K, history_mode, seeds, grasp threshold) come from
    the experiment level; backend-specific params from the sub-config."""
    bcfg = cfg.perturbation_backend
    name = str(bcfg.name)
    variant, To, hist = r['variant'], r['n_obs_steps'], cfg.history_mode
    K = int(cfg.K)

    if name == 'sim_state':
        per_body_sigma = None
        if bcfg.get('per_body_sigma', None) is not None:
            per_body_sigma = {k: (float(v['sigma_pos']), float(v['sigma_rot']))
                              for k, v in bcfg.per_body_sigma.items()}
        perturb_targets = list(bcfg.perturb_targets)
        n_arm = (len(rs_env.robots[0]._ref_joint_pos_indexes)
                 if 'eef' in perturb_targets else None)
        return SimStateBackend(
            variant, To, hist, bodies=bodies,
            sigma_pos_eef=bcfg.sigma_pos.eef, sigma_rot_eef=bcfg.sigma_rot.eef,
            sigma_pos_object=bcfg.sigma_pos.object, sigma_rot_object=bcfg.sigma_rot.object,
            per_body_sigma=per_body_sigma, sigma_qpos=bcfg.get('sigma_qpos', 0.0),
            n_arm_joints=n_arm, perturb_targets=perturb_targets,
            grasp_qpos_threshold=float(cfg.grasp_qpos_threshold),
            settle_steps=int(bcfg.get('settle_steps', 0)),
            seed_perturb=int(cfg.seeds.perturb))

    if name == 'obs_noise':
        if variant != 'lowdim':
            raise ValueError(
                "perturbation_backend=obs_noise supports LOW_DIM checkpoints only "
                f"(got variant={variant!r}). Use perturbation_backend=sim_state for image.")
        obs_std = (policy.normalizer['obs'].get_input_stats()['std']
                   .detach().cpu().numpy().reshape(-1))
        obs_dim = int(obs_std.shape[0])
        key_ranges = obs_builder.lowdim_obs_key_ranges(wrapper)
        ndm = bcfg.get('noise_dim_mask', None)
        ndm = list(ndm) if ndm is not None else None
        return ObsNoiseBackend(
            variant, To, hist, obs_dim=obs_dim, obs_std=obs_std,
            obs_key_ranges=key_ranges, K=K,
            noise_mode=str(bcfg.get('noise_mode', 'per_dim_std')),
            noise_scale=float(bcfg.get('noise_scale', 0.1)),
            sigma_abs=float(bcfg.get('sigma_abs', 0.01)),
            noise_dim_mask=ndm,
            seed=int(bcfg.get('seed', cfg.seeds.perturb)))

    raise ValueError(f"unknown perturbation_backend {name!r} (sim_state | obs_noise).")
