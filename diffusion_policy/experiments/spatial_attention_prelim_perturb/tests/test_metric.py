"""Coupled endpoint-distance metric via a toy deterministic policy."""
import numpy as np
import torch

from diffusion_policy.experiments.spatial_attention_prelim_perturb.metric.crn import CRNManager
from diffusion_policy.experiments.spatial_attention_prelim_perturb.metric.endpoint_distance import (
    CoupledEndpointDistance,
)

H, D, To = 4, 3, 2


class DummyPerturbPolicy:
    """Deterministic given _init_noise: norm = eps0 + mean(obs); raw = 2*norm + 1."""
    def __init__(self):
        self.action_dim = D
        self.horizon = H
        self.n_obs_steps = To
        self.oa_step_convention = True
        self.num_inference_steps = 99
        self.dtype = torch.float32
        self.device = torch.device('cpu')
        self._init_noise = None

    def eval(self):
        return self

    def predict_action(self, obs_dict):
        eps = self._init_noise                       # (B, H, D)
        g = torch.stack([v.float().mean() for v in obs_dict.values()]).mean()
        norm = eps + g
        raw = norm * 2.0 + 1.0
        start = self.n_obs_steps - 1
        return {'naction_pred': norm, 'action_pred': raw,
                'naction': norm[:, start:start + 1], 'action': raw[:, start:start + 1]}


def _metric():
    crn = CRNManager(k_s=8, horizon=H, action_dim=D, eps0_seed=0)
    return CoupledEndpointDistance(crn, ode_steps=5, distance_space='both', max_batch=64)


def test_crn_identity_zero_for_same_input():
    m = _metric(); p = DummyPerturbPolicy()
    nom = {'obs': torch.zeros(1, To, 5)}
    res = m.compute(p, nom, [nom, nom])       # perturbed == nominal
    for sp in ('raw', 'norm'):
        assert res[sp]['S'] < 1e-12
        assert np.all(res[sp]['D_k'] < 1e-12)


def test_control_is_zero_and_M_restored():
    m = _metric(); p = DummyPerturbPolicy()
    nom = {'obs': torch.zeros(1, To, 5)}
    pert = [{'obs': torch.full((1, To, 5), 0.1 * (k + 1))} for k in range(3)]
    res = m.compute(p, nom, pert, compute_control=True)
    for sp in ('raw', 'norm'):
        assert res[sp]['control'] < 1e-12       # nominal-vs-nominal control ~0
    assert p.num_inference_steps == 99          # forced-M restored


def test_per_index_sum_equals_scalar_and_space_relation():
    m = _metric(); p = DummyPerturbPolicy()
    nom = {'obs': torch.zeros(1, To, 5)}
    pert = [{'obs': torch.full((1, To, 5), 0.2 * (k + 1))} for k in range(4)]
    res = m.compute(p, nom, pert)
    for sp in ('raw', 'norm'):
        assert abs(res[sp]['per_index'].sum() - res[sp]['S']) < 1e-8
        # S_first is the per-index value at the executed-slice start
        assert abs(res[sp]['S_first'] - res[sp]['per_index'][To - 1]) < 1e-12
    # raw diff = 2 * norm diff  =>  squared distance 4x
    assert abs(res['raw']['S'] - 4.0 * res['norm']['S']) < 1e-6
    assert res['norm']['S'] > 0.0               # sensitive to the input change


def test_action_dims_subset_excludes_gripper():
    p = DummyPerturbPolicy()
    nom = {'obs': torch.zeros(1, To, 5)}
    pert = [{'obs': torch.full((1, To, 5), 0.3)}]
    crn = CRNManager(k_s=8, horizon=H, action_dim=D, eps0_seed=0)
    full = CoupledEndpointDistance(crn, 5, 'norm').compute(p, nom, pert)
    sub = CoupledEndpointDistance(crn, 5, 'norm', action_dims=list(range(D - 1))).compute(p, nom, pert)
    # the dummy shifts every action dim equally, so dropping 1 of D dims scales S by (D-1)/D
    assert abs(sub['norm']['S'] - full['norm']['S'] * (D - 1) / D) < 1e-6
    assert sub['norm']['S'] < full['norm']['S']
    assert sub['norm']['per_index'].shape == (H,)          # still summed per horizon index
    assert abs(sub['norm']['per_index'].sum() - sub['norm']['S']) < 1e-8


def test_both_matches_single_space_runs():
    p = DummyPerturbPolicy()
    nom = {'obs': torch.zeros(1, To, 5)}
    pert = [{'obs': torch.full((1, To, 5), 0.3)}]
    crn = CRNManager(k_s=8, horizon=H, action_dim=D, eps0_seed=0)
    both = CoupledEndpointDistance(crn, 5, 'both').compute(p, nom, pert)
    raw_only = CoupledEndpointDistance(crn, 5, 'raw').compute(p, nom, pert)
    assert abs(both['raw']['S'] - raw_only['raw']['S']) < 1e-12
    assert 'norm' not in raw_only
