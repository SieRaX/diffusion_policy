"""Perturbation-backend tests (no robosuite/GPU).

Covers: obs_noise is low_dim-only; the fixed {delta_k} draw is reused across
timesteps; the SAME delta_k tiles every history slot; per_dim_std / isotropic /
masked sigma against hand-computed values; sim_state stays bitwise-identical after
the refactor (parity vs a copy of the original inline loop); and a single override
(perturbation_backend=obs_noise) composes correctly.
"""
import pathlib

import numpy as np
import pytest
import torch

from diffusion_policy.experiments.spatial_attention_prelim_perturb.obs_builder import obs_builder
from diffusion_policy.experiments.spatial_attention_prelim_perturb.perturbation.backend import (
    ObsNoiseBackend, SimStateBackend,
)
from diffusion_policy.experiments.spatial_attention_prelim_perturb.perturbation.bodies import PerturbBody
from diffusion_policy.experiments.spatial_attention_prelim_perturb.perturbation.state_perturb import (
    sample_realization,
)

RANGES = {'k0': (0, 2), 'k1': (2, 4)}
STD = np.array([1.0, 2.0, 0.5, 4.0])


# ------------------------------------------------------------------ fake wrapper (obs_noise)
# get_observation for state s is a KNOWN vector [s, s+1, s+2, s+3], so
# (perturbed - nominal) == delta_k exactly, independent of the state.
class _NSim:
    def forward(self): pass


class _NInner:
    def __init__(self): self.sim = _NSim()


class _NEnv:
    def __init__(self, obs_dim):
        self.env = _NInner()
        self.obs_dim = obs_dim
        self.half = obs_dim // 2
        self.cur = None

    def reset_to(self, d):
        self.cur = float(np.asarray(d['states']).reshape(-1)[0])

    def get_observation(self):
        base = self.cur + np.arange(self.obs_dim, dtype=np.float32)
        return {'k0': base[:self.half], 'k1': base[self.half:]}


class _NWrapper:
    def __init__(self, obs_dim):
        self.env = _NEnv(obs_dim)
        self.obs_keys = ['k0', 'k1']

    def get_observation(self):
        raw = self.env.get_observation()
        return np.concatenate([raw[k] for k in self.obs_keys], axis=0)


def _obs_noise_backend(To=1, history_mode='tile_perturbed', K=4, seed=0,
                       noise_mode='isotropic', noise_scale=0.1, sigma_abs=0.5,
                       noise_dim_mask=None, obs_dim=4, ranges=RANGES, variant='lowdim'):
    return ObsNoiseBackend(
        variant, To, history_mode, obs_dim=obs_dim, obs_std=STD[:obs_dim],
        obs_key_ranges=ranges, K=K, noise_mode=noise_mode, noise_scale=noise_scale,
        sigma_abs=sigma_abs, noise_dim_mask=noise_dim_mask, seed=seed)


# ------------------------------------------------------------------ tests: obs_noise
def test_obs_noise_is_lowdim_only():
    with pytest.raises(ValueError):
        _obs_noise_backend(variant='image')


def test_lowdim_obs_key_ranges():
    w = _NWrapper(4)
    w.env.reset_to({'states': [0.0]})
    assert obs_builder.lowdim_obs_key_ranges(w) == {'k0': (0, 2), 'k1': (2, 4)}


def test_deltas_reused_across_timesteps():
    # perturbed = nominal + the SAME fixed delta_k at every timestep (compared via
    # exact reconstruction to avoid float32 cancellation in (f+d)-f at large |f|).
    be = _obs_noise_backend(To=1, K=4, seed=0)
    w = _NWrapper(4)
    states = np.arange(10.0)
    snap = be.deltas.copy()
    for t in (3, 7):
        nom = be.build_nominal(w, states, t)['obs']            # (1,1,4)
        pert = be.build_perturbed(w, states, t, 4)
        for k in range(4):
            d = torch.from_numpy(be.deltas[k])
            assert torch.equal(pert[k]['obs'], nom + d)        # same delta_k added at every t
    assert np.array_equal(be.deltas, snap)                     # {delta_k} never redrawn


def test_same_delta_tiled_across_history_slots():
    be = _obs_noise_backend(To=2, history_mode='tile_perturbed', K=3, seed=1)
    w = _NWrapper(4)
    states = np.arange(10.0)
    nom = be.build_nominal(w, states, 5)['obs']                # (1,2,4)
    pert = be.build_perturbed(w, states, 5, 3)
    for k in range(3):
        d = torch.from_numpy(be.deltas[k])
        obs = pert[k]['obs']                                   # (1,2,4)
        assert torch.equal(obs[0, 0], obs[0, 1])               # SAME delta tiled in both slots
        assert torch.equal(obs, nom + d)                       # each slot = nominal + delta


def test_current_frame_only_noises_last_slot_only():
    be = _obs_noise_backend(To=2, history_mode='current_frame_only', K=2, seed=2)
    w = _NWrapper(4)
    states = np.arange(10.0)
    nom = be.build_nominal(w, states, 5)['obs']                # (1,2,4)
    pert = be.build_perturbed(w, states, 5, 2)
    for k in range(2):
        d = torch.from_numpy(be.deltas[k])
        obs = pert[k]['obs']
        assert torch.equal(obs[0, 0], nom[0, 0])               # history frame untouched
        assert torch.equal(obs[0, 1], nom[0, 1] + d)           # only the last frame noised


def test_per_dim_std_scaling():
    be = _obs_noise_backend(noise_mode='per_dim_std', noise_scale=0.1, K=5, seed=123)
    expected_sigma = 0.1 * STD
    assert np.allclose(be.sigma, expected_sigma)
    z = np.random.default_rng(123).standard_normal((5, 4))
    assert np.allclose(be.deltas, (z * expected_sigma[None, :]).astype(np.float32))


def test_isotropic_sigma():
    be = _obs_noise_backend(noise_mode='isotropic', sigma_abs=0.05, K=2)
    assert np.allclose(be.sigma, np.full(4, 0.05))


def test_noise_dim_mask_by_index_and_key():
    bi = _obs_noise_backend(noise_mode='isotropic', sigma_abs=0.05, noise_dim_mask=[0, 2])
    assert np.allclose(bi.sigma, [0.05, 0.0, 0.05, 0.0])

    bk = _obs_noise_backend(noise_mode='isotropic', sigma_abs=0.05, noise_dim_mask=['k1'])
    assert np.allclose(bk.sigma, [0.0, 0.0, 0.05, 0.05])
    assert np.allclose(bk.deltas[:, 0], 0.0) and np.allclose(bk.deltas[:, 1], 0.0)  # masked dims -> 0


def test_obs_noise_metadata_has_std_cache():
    be = _obs_noise_backend(noise_mode='per_dim_std')
    md = be.metadata()
    assert np.allclose(md['obs_std'], STD)                     # per-dim std cache artifact
    assert md['noise_mode'] == 'per_dim_std'
    assert 'noise_sigma' in md and 'noise_dim_mask' in md


# ------------------------------------------------------------------ fake sim (sim_state parity)
class _PData:
    def __init__(self):
        self._q = {}
        self.qvel = np.ones(64)

    def get_joint_qpos(self, j): return self._q[j].copy()

    def set_joint_qpos(self, j, v): self._q[j] = np.asarray(v, dtype=np.float64).copy()


class _PModel:
    def __init__(self, addr): self._addr = addr

    def get_joint_qvel_addr(self, j): return self._addr[j]


class _PSim:
    def __init__(self, addr):
        self.data = _PData()
        self.model = _PModel(addr)

    def forward(self): pass

    def step(self): pass


class _PRS:  # rs_env
    def __init__(self, sim): self.sim = sim


class _PEnv:  # wrapper.env
    def __init__(self, bodies, addr):
        self.sim = _PSim(addr)
        self.env = _PRS(self.sim)
        self.bodies = bodies

    def reset_to(self, d):
        s = float(np.asarray(d['states']).reshape(-1)[0])
        for i, b in enumerate(self.bodies):
            self.sim.data.set_joint_qpos(b.joint, np.array([s + i, 0.1 * i, 0.2, 1, 0, 0, 0], float))


class _PWrapper:
    def __init__(self, bodies, addr):
        self.env = _PEnv(bodies, addr)
        self.obs_keys = [b.name for b in bodies]

    def get_observation(self):
        raw = {b.name: self.env.sim.data.get_joint_qpos(b.joint) for b in self.env.bodies}
        return np.concatenate([raw[k] for k in self.obs_keys], axis=0)


def _reference_perturbed(wrapper, bodies, To, hist, states, t, K, seed, spe, sre, spo, sro):
    """Verbatim copy of the pre-refactor runner's per-timestep perturbation loop."""
    rng_t = np.random.default_rng([seed, int(t)])
    out = []
    for _k in range(K):
        realization = sample_realization(bodies, rng_t, spe, sre, spo, sro,
                                         per_body_sigma=None, sigma_qpos=0.0, n_arm_joints=None)
        applier = obs_builder.make_perturb_applier(bodies, realization, 0.06, ['objects'], 0)
        out.append(obs_builder.build_input(wrapper, 'lowdim', To, hist, states, t, applier=applier))
    return out


def test_sim_state_backend_parity(monkeypatch):
    # grasp detection needs robosuite; force non-grasped so BOTH paths use the
    # independent-noise branch (the parity target is the RNG stream + call order).
    monkeypatch.setattr(obs_builder, 'is_grasped', lambda *a, **k: False)
    bodies = [PerturbBody(name='cube', joint='cube_joint', root_body='cube_main', grasp_handle=None)]
    addr = {'cube_joint': (0, 6)}
    states = np.arange(8.0)
    spe, sre, spo, sro = 0.005, 0.02, 0.005, 0.02

    for hist in ('tile_perturbed', 'consistent_perturbed', 'current_frame_only'):
        for To in (1, 2):
            w = _PWrapper(bodies, addr)
            ref = _reference_perturbed(w, bodies, To, hist, states, t=4, K=3, seed=7,
                                       spe=spe, sre=sre, spo=spo, sro=sro)
            be = SimStateBackend(
                'lowdim', To, hist, bodies=bodies,
                sigma_pos_eef=spe, sigma_rot_eef=sre, sigma_pos_object=spo, sigma_rot_object=sro,
                per_body_sigma=None, sigma_qpos=0.0, n_arm_joints=None, perturb_targets=['objects'],
                grasp_qpos_threshold=0.06, settle_steps=0, seed_perturb=7)
            got = be.build_perturbed(w, states, t=4, K=3)
            assert len(got) == len(ref) == 3
            for g, rr in zip(got, ref):
                assert torch.equal(g['obs'], rr['obs']), (hist, To)


# ------------------------------------------------------------------ config composition
def test_single_override_backend_switch():
    from hydra import compose, initialize_config_dir
    cfg_dir = str(pathlib.Path(__file__).resolve().parents[1] / 'config')

    with initialize_config_dir(config_dir=cfg_dir, version_base=None):
        cfg = compose(config_name='perturb',
                      overrides=['checkpoint=/x.ckpt', 'perturbation_backend=obs_noise'])
    assert cfg.perturbation_backend.name == 'obs_noise'
    assert cfg.perturbation_backend.noise_mode == 'per_dim_std'
    # shared params untouched by the single override
    assert int(cfg.K) == 8 and str(cfg.history_mode) == 'tile_perturbed'
    assert int(cfg.seeds.crn) == 0

    with initialize_config_dir(config_dir=cfg_dir, version_base=None):
        cfg2 = compose(config_name='perturb', overrides=['checkpoint=/x.ckpt'])
    assert cfg2.perturbation_backend.name == 'sim_state'       # DEFAULT
    assert float(cfg2.perturbation_backend.sigma_pos.eef) == 0.005
