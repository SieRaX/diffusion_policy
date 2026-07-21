"""S_end estimator (reuses the coupled-endpoint metric): coupled CRN identity, /denom,
per-index sum == scalar, raw==4*norm (distance-space consistency), include_gripper, and
the small-query S_end subset rule."""
import numpy as np
import torch

from diffusion_policy.experiments.spatial_attention_exp1.mse_metric.crn import CRNManager
from diffusion_policy.experiments.spatial_attention_prelim_perturb.metric.endpoint_distance import (
    CoupledEndpointDistance,
)
from diffusion_policy.experiments.spatial_attention_exp2.estimators.endpoint import (
    EndpointSensitivity, stratified_subset,
)
from diffusion_policy.experiments.spatial_attention_exp2.tests.toy import DummyEndpointPolicy

H, D, To = 4, 3, 2


def _endpoint(action_dims=None, spaces='both'):
    crn = CRNManager(k_s=4, horizon=H, action_dim=D, eps0_seed=0)
    metric = CoupledEndpointDistance(crn, ode_steps=5, distance_space=spaces,
                                     max_batch=64, action_dims=action_dims)
    return EndpointSensitivity(metric)


def test_identical_input_zero():
    ep = _endpoint()
    nom = {'obs': torch.zeros(1, To, 5)}
    res = ep.compute(DummyEndpointPolicy(H, D, To), nom, [nom], denom=np.ones(1))
    for sp in ('raw', 'norm'):
        assert res[sp]['S'] < 1e-12


def test_denom_and_space_consistency_and_perindex_sum():
    ep = _endpoint()
    p = DummyEndpointPolicy(H, D, To)
    nom = {'obs': torch.zeros(1, To, 5)}
    probes = [{'obs': torch.full((1, To, 5), 0.3)}]
    r1 = ep.compute(p, nom, probes, denom=np.array([1.0]))
    r2 = ep.compute(p, nom, probes, denom=np.array([2.0]))
    for sp in ('raw', 'norm'):
        assert abs(r1[sp]['per_index'].sum() - r1[sp]['S']) < 1e-9
        assert abs(r2[sp]['S'] - r1[sp]['S'] / 2.0) < 1e-9          # /denom always on
    assert abs(r1['raw']['S'] - 4.0 * r1['norm']['S']) < 1e-6      # raw = (2*norm)^2


def test_include_gripper_drops_last_dim():
    p = DummyEndpointPolicy(H, D, To)
    nom = {'obs': torch.zeros(1, To, 5)}
    probes = [{'obs': torch.full((1, To, 5), 0.3)}]
    s_all = _endpoint(action_dims=None).compute(p, nom, probes, np.ones(1))['norm']['S']
    s_arm = _endpoint(action_dims=list(range(D - 1))).compute(p, nom, probes, np.ones(1))['norm']['S']
    assert abs(s_arm - s_all * (D - 1) / D) < 1e-6


def test_small_query_subset_rule():
    # n_query <= endpoint_subset_min -> all query states
    idx = stratified_subset(50, np.arange(50.0), np.zeros(50, int), frac=0.5, min_all=300, seed=0)
    assert np.array_equal(idx, np.arange(50))
    # n_query > min -> a strict subset, roughly frac of them
    big = stratified_subset(1000, np.random.default_rng(0).random(1000),
                            (np.arange(1000) % 4), frac=0.5, min_all=300, seed=0)
    assert 0 < len(big) < 1000
