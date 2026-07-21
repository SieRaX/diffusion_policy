"""S_vel estimator: CRN identity, /denom, per-index sum == scalar, S_first alignment,
include_gripper drops the last action dim."""
import numpy as np
import torch

from diffusion_policy.experiments.spatial_attention_exp1.mse_metric.crn import CRNManager
from diffusion_policy.experiments.spatial_attention_exp2.estimators.velocity import VelocitySensitivity
from diffusion_policy.experiments.spatial_attention_exp2.tests.toy import DummyVelPolicy

H, Da, To, Do = 4, 3, 2, 5


def _setup(action_dims=None, num_fm=8):
    p = DummyVelPolicy(H, Da, To, Do)
    crn = CRNManager(k_s=1, horizon=H, action_dim=Da, num_fm=num_fm, fm_seed=0)
    vel = VelocitySensitivity(crn, action_dims=action_dims, max_batch=64)
    nom = {'obs': torch.zeros(1, To, Do)}
    a_gt = torch.zeros(H, Da)
    return p, vel, nom, a_gt


def test_crn_identity_zero_for_same_obs():
    p, vel, nom, a_gt = _setup()
    res = vel.compute(p, nom, [nom, nom], a_gt, denom=np.ones(2))   # probe == nominal
    assert res['scalar'] < 1e-12
    assert np.all(np.abs(res['per_index']) < 1e-12)


def test_per_index_sum_equals_scalar_and_first_alignment():
    p, vel, nom, a_gt = _setup()
    probes = [{'obs': torch.full((1, To, Do), 0.2 * (k + 1))} for k in range(3)]
    res = vel.compute(p, nom, probes, a_gt, denom=np.ones(3))
    assert abs(res['per_index'].sum() - res['scalar']) < 1e-9
    assert abs(res['first'] - res['per_index'][To - 1]) < 1e-12
    assert res['scalar'] > 0


def test_distance_normalization_scales_inversely():
    p, vel, nom, a_gt = _setup()
    probes = [{'obs': torch.full((1, To, Do), 0.3)}]
    s1 = vel.compute(p, nom, probes, a_gt, denom=np.array([1.0]))['scalar']
    s2 = vel.compute(p, nom, probes, a_gt, denom=np.array([2.0]))['scalar']
    assert abs(s2 - s1 / 2.0) < 1e-9      # /denom is always on


def test_include_gripper_drops_last_dim():
    probes = [{'obs': torch.full((1, To, Do), 0.3)}]
    p, vel_all, nom, a_gt = _setup(action_dims=None)
    p2, vel_arm, nom2, a_gt2 = _setup(action_dims=list(range(Da - 1)))
    s_all = vel_all.compute(p, nom, probes, a_gt, denom=np.ones(1))['scalar']
    s_arm = vel_arm.compute(p2, nom2, probes, a_gt2, denom=np.ones(1))['scalar']
    # the toy contributes equally per action dim, so dropping 1 of Da scales by (Da-1)/Da
    assert abs(s_arm - s_all * (Da - 1) / Da) < 1e-5      # float32 velocity forward
    assert s_arm < s_all
