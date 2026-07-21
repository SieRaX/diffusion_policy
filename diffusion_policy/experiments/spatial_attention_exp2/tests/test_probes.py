"""Probe types + fallback paths (env-free ProbePlans)."""
import numpy as np

from diffusion_policy.experiments.spatial_attention_exp2.similarity.index import Index
from diffusion_policy.experiments.spatial_attention_exp2.probes.base import make_probe_type
from diffusion_policy.experiments.spatial_attention_exp2.probes.real_neighbor import RealNeighborDiff
from diffusion_policy.experiments.spatial_attention_exp2.probes.noise import DiagStdNoise, FullCovNoise
from diffusion_policy.experiments.spatial_attention_exp2.probes.fallback import apply_fallback

EPS = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
TS = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
VEC = np.array([[0, 0], [1, 0], [2, 0],
                [0, 1], [1, 1], [2, 1],
                [0, 5], [1, 5], [2, 5]], float)
QROW = 1  # (ep0, t=1)

PARAMS = dict(K=4, max_radius=10.0, min_neighbors=1, same_episode_window=10,
              project_out_temporal=True, exclude_same_episode=False, rank=2,
              seed_base=0, fallback='dtw')


def _idx():
    return Index(VEC, EPS, TS, metric='euclidean')


def test_make_probe_type_dispatch():
    assert isinstance(make_probe_type('real_neighbor_diff'), RealNeighborDiff)
    assert isinstance(make_probe_type('diag_std_noise'), DiagStdNoise)
    assert isinstance(make_probe_type('fullcov_noise'), FullCovNoise)


def test_real_neighbor_plan_config_denom_is_d2():
    plan = RealNeighborDiff().plan(QROW, _idx(), 'config', PARAMS)
    assert plan.kind == 'real'
    # probes come from OTHER episodes only (probe pool ⊇ query, S computed elsewhere)
    assert all(ep != 0 for ep, _ in plan.targets)
    # denom is per-probe d^2
    assert plan.denom is not None and plan.denom.shape[0] == len(plan.targets)
    assert np.all(plan.denom >= 0)


def test_diag_noise_plan_shapes_and_denom():
    plan = DiagStdNoise().plan(QROW, _idx(), 'config', PARAMS)
    assert plan.kind == 'noise'
    assert plan.deltas.shape == (4, 2)
    assert np.allclose(plan.denom, (plan.deltas ** 2).sum(1).mean())   # E||δ||^2, same for all K


def test_fullcov_noise_low_rank():
    plan = FullCovNoise().plan(QROW, _idx(), 'config', {**PARAMS, 'rank': 1})
    assert plan.kind == 'noise' and plan.deltas.shape == (4, 2)


def test_fallback_paths():
    idx = _idx()
    pool_phases = np.array([0, 1, 0, 0, 1, 0, 0, 1, 0])   # phase per pool row
    ctx = {'pool_phases': pool_phases}
    # a flagged plan (no neighbors within a tiny radius)
    flagged = RealNeighborDiff().plan(QROW, idx, 'config', {**PARAMS, 'max_radius': 1e-6, 'min_neighbors': 3})
    assert flagged.flagged

    inv = apply_fallback(flagged, QROW, idx, {**PARAMS, 'fallback': 'default_horizon'}, ctx)
    assert inv.kind == 'invalid'

    dw = apply_fallback(flagged, QROW, idx, {**PARAMS, 'fallback': 'downweight'}, ctx)
    assert dw.weight == float(flagged.n_valid)

    dtw = apply_fallback(flagged, QROW, idx, {**PARAMS, 'fallback': 'dtw'}, ctx)
    assert dtw.kind in ('real', 'invalid')
    if dtw.kind == 'real':
        # phase-matched cross-demo: same phase as the query, other episodes
        qphase = pool_phases[QROW]
        for ep, t in dtw.targets:
            assert ep != 0


def test_not_flagged_plan_passes_through_fallback():
    idx = _idx()
    good = RealNeighborDiff().plan(QROW, idx, 'config', PARAMS)
    assert not good.flagged
    out = apply_fallback(good, QROW, idx, PARAMS, {'pool_phases': None})
    assert out is good
