"""Directions-combine (never pooled), a_GT alignment, index-cache reuse, timeline-episode
validation, phase segmentation, CRN identity, and loud checkpoint-config resolution."""
import os
import tempfile

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from diffusion_policy.experiments.spatial_attention_exp1.mse_metric.crn import CRNManager
from diffusion_policy.experiments.spatial_attention_exp2.runner import _combine, _directions
from diffusion_policy.experiments.spatial_attention_exp2.obs_source import GTActions
from diffusion_policy.experiments.spatial_attention_exp2.similarity.build import build_or_load_index
from diffusion_policy.experiments.spatial_attention_exp2.util import validate_timeline_episode
from diffusion_policy.experiments.spatial_attention_exp2.segmentation import phases as seg
from diffusion_policy.experiments.spatial_attention_prelim_perturb.runner import _resolve_from_checkpoint


def test_directions_combine_is_weighted_sum_not_pooled():
    Q, H = 3, 2
    maps = {
        'temporal': {'scalar': np.array([1., 2., 3.]), 'first': np.array([1., 1., 1.]),
                     'per_index': np.ones((Q, H))},
        'config': {'scalar': np.array([10., 20., 30.]), 'first': np.array([2., 2., 2.]),
                   'per_index': 2 * np.ones((Q, H))},
    }
    out = _combine(maps, ['temporal', 'config'], [0.25, 0.75], H, Q)
    assert np.allclose(out['scalar'], 0.25 * maps['temporal']['scalar'] + 0.75 * maps['config']['scalar'])
    assert np.allclose(out['per_index'], 0.25 * 1 + 0.75 * 2)
    # single direction passes through unchanged (the two classes are never merged into one)
    assert _combine(maps, ['config'], [0.5, 0.5], H, Q) is maps['config']
    assert _directions('both') == ['temporal', 'config']


def test_gt_action_alignment():
    actions = np.arange(30, dtype=np.float32).reshape(30, 1)   # 3 episodes of 10
    ends = np.array([10, 20, 30])
    gt = GTActions(actions, ends, n_obs_steps=2, horizon=4, action_normalizer=None)
    # chunk index To-1 == env timestep t; mid-episode -> no clamping
    assert np.array_equal(gt.raw_chunk(1, 3)[:, 0], [12, 13, 14, 15])
    # episode start clamps the pre-pad
    assert np.array_equal(gt.raw_chunk(0, 0)[:, 0], [0, 0, 1, 2])


def test_index_cache_reuse():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, 'cache.npz')
        pool_e = np.array([0, 0, 1]); pool_t = np.array([0, 1, 0])
        calls = {'n': 0}

        def vectors_fn():
            calls['n'] += 1
            return np.array([[0., 0.], [1., 0.], [0., 1.]])

        idx1, reused1 = build_or_load_index(path, pool_e, pool_t, vectors_fn)
        idx2, reused2 = build_or_load_index(path, pool_e, pool_t, vectors_fn)
        assert reused1 is False and reused2 is True
        assert calls['n'] == 1                    # second invocation loaded, did NOT rebuild
        assert np.array_equal(idx1.vectors, idx2.vectors)


def test_validate_timeline_episode():
    assert validate_timeline_episode(2, [0, 1, 2]) == 2
    with pytest.raises(ValueError):
        validate_timeline_episode(3, [0, 1, 2])


def test_segment_episode_phases():
    grasped = np.array([0, 0, 1, 1, 1, 1, 1, 0], bool)
    gopen = ~grasped
    lab = seg.segment_episode(gopen, grasped, grasp_window=1, place_window=1)
    assert lab[0] == seg.PHASE_ID['APPROACH']
    assert lab[2] == seg.PHASE_ID['GRASP']       # onset
    assert lab[6] == seg.PHASE_ID['PLACE']       # near release
    assert lab[4] == seg.PHASE_ID['TRANSPORT']   # middle
    assert lab[7] == seg.PHASE_ID['APPROACH']    # released


def test_crn_identity_across_instances():
    a = CRNManager(k_s=8, horizon=4, action_dim=3, num_fm=8, eps0_seed=0, fm_seed=1)
    b = CRNManager(k_s=8, horizon=4, action_dim=3, num_fm=8, eps0_seed=0, fm_seed=1)
    assert torch.equal(a.eps0, b.eps0)
    assert torch.equal(a.taus, b.taus) and torch.equal(a.eps_fm, b.eps_fm)


def _lowdim_cfg():
    FM = 'diffusion_policy.policy.flow_matching_unet_lowdim_policy.FlowMatchingUnetLowdimPolicy'
    return OmegaConf.create({
        'task': {'name': 'lift_lowdim', 'task_name': 'lift', 'obs_keys': ['object'],
                 'abs_action': True, 'dataset': {'dataset_path': '/data/x.hdf5'}},
        'n_obs_steps': 2, 'horizon': 16, 'policy': {'_target_': FM}})


def test_checkpoint_resolution_fails_loudly():
    cfg = _lowdim_cfg()
    del cfg['n_obs_steps']
    with pytest.raises(KeyError):
        _resolve_from_checkpoint(cfg, None)
