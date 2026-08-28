import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

import torch
import pytest
from diffusers.schedulers.scheduling_ddim import DDIMScheduler

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.model.diffusion.moh_conditional_unet1d import MoHConditionalUnet1D
from diffusion_policy.policy.moh_diffusion_unet_lowdim_policy import MoHDiffusionUnetLowdimPolicy

OBS_DIM = 3
ACTION_DIM = 2
HORIZON = 8
N_OBS_STEPS = 2


def make_policy(horizons=(4, 8), min_active_horizons=1, **kwargs):
    model = MoHConditionalUnet1D(
        input_dim=ACTION_DIM,
        global_cond_dim=OBS_DIM * N_OBS_STEPS,
        diffusion_step_embed_dim=16,
        down_dims=[8, 16],
        kernel_size=3,
        n_groups=4,
        cond_predict_scale=True)
    scheduler = DDIMScheduler(
        num_train_timesteps=10,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule='squaredcos_cap_v2',
        clip_sample=True,
        set_alpha_to_one=True,
        steps_offset=0,
        prediction_type='epsilon')
    policy = MoHDiffusionUnetLowdimPolicy(
        model=model,
        noise_scheduler=scheduler,
        horizon=HORIZON,
        horizons=list(horizons),
        obs_dim=OBS_DIM,
        action_dim=ACTION_DIM,
        n_action_steps=3,
        n_obs_steps=N_OBS_STEPS,
        num_inference_steps=4,
        min_replan_steps=2,
        min_active_horizons=min_active_horizons,
        scale_ratio=1.1,
        downsample_factor=2,
        **kwargs)
    normalizer = LinearNormalizer()
    normalizer.fit({
        'obs': torch.randn(100, OBS_DIM),
        'action': torch.randn(100, ACTION_DIM),
    })
    policy.set_normalizer(normalizer)
    return policy


def make_batch(B=5):
    return {
        'obs': torch.randn(B, HORIZON, OBS_DIM),
        'action': torch.randn(B, HORIZON, ACTION_DIM),
    }


def test_horizon_validation():
    with pytest.raises(AssertionError):
        make_policy(horizons=(4, 6))  # 6 not divisible by downsample factor 2? 6%2==0 -> max!=horizon fails
    with pytest.raises(AssertionError):
        make_policy(horizons=(3, 8))  # 3 not a multiple of 2


def test_gate_weights_mask_and_normalization():
    policy = make_policy()
    B = 4
    feats = [torch.randn(B, h, policy.model.feature_dim) for h in policy.horizons]
    alpha = policy._gate_weights(feats, training=False)
    assert alpha.shape == (B, HORIZON, len(policy.horizons))
    # invalid (step k >= h) entries carry zero weight
    for i, h in enumerate(policy.horizons):
        assert torch.all(alpha[:, h:, i] == 0)
    # weights over valid horizons sum to 1 at every step
    assert torch.allclose(alpha.sum(dim=-1), torch.ones(B, HORIZON), atol=1e-5)


def test_fused_equals_single_horizon():
    policy = make_policy(horizons=(HORIZON,), min_active_horizons=0)
    B = 3
    preds = [torch.randn(B, HORIZON, ACTION_DIM)]
    feats = [torch.randn(B, HORIZON, policy.model.feature_dim)]
    alpha = policy._gate_weights(feats, training=False)
    fused = policy._fuse(preds, alpha)
    assert torch.allclose(fused, preds[0], atol=1e-6)


def test_balance_loss_zero_for_uniform_usage():
    policy = make_policy()
    B = 4
    # uniform weights over valid horizons at every step
    valid = policy.valid_mask.float()  # (T, N)
    alpha = (valid / valid.sum(dim=-1, keepdim=True)).unsqueeze(0).expand(B, -1, -1)
    bal = policy._balance_loss(alpha)
    assert bal.item() == pytest.approx(0.0, abs=1e-10)


def test_compute_loss_finite_and_backward():
    policy = make_policy()
    policy.train()
    loss = policy.compute_loss(make_batch())
    assert torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in policy.parameters() if p.grad is not None]
    assert len(grads) > 0
    assert all(torch.isfinite(g).all() for g in grads)
    assert set(policy.last_loss_info) == {'loss_mix', 'loss_ind', 'loss_bal'}


def test_predict_action_fixed_mode():
    policy = make_policy()
    policy.eval()
    with torch.no_grad():
        result = policy.predict_action({'obs': torch.randn(2, HORIZON, OBS_DIM)})
    # oa_step_convention: start = To-1 = 1
    assert result['action'].shape == (2, policy.n_action_steps, ACTION_DIM)
    assert result['action_pred'].shape == (2, HORIZON, ACTION_DIM)


def test_consensus_exec_steps():
    policy = make_policy()  # horizons (4,8), n=2, m=1, start will be passed
    B, start = 2, 1
    # zero disagreement: extension limited by active-horizon count
    # n_active: steps 0-3 -> 2 horizons, steps 4-7 -> 1 horizon (= m, stops)
    d = torch.zeros(B, HORIZON)
    n_exec = policy._consensus_exec_steps(d, start=start)
    # j=2 (global 3, 2 active) ok; j=3 (global 4, 1 active) stops -> 3
    assert torch.all(n_exec == 3)
    # disagreement spike right after the threshold window stops extension
    d = torch.zeros(B, HORIZON)
    d[:, start + 2] = 10.0
    n_exec = policy._consensus_exec_steps(d, start=start)
    assert torch.all(n_exec == policy.min_replan_steps)


def test_dynamic_single_horizon_runs_full_chunk():
    # one candidate horizon -> zero disagreement -> execute the whole chunk
    policy = make_policy(horizons=(HORIZON,), min_active_horizons=0)
    policy.eval()
    with torch.no_grad():
        result = policy.predict_action(
            {'obs': torch.randn(2, HORIZON, OBS_DIM)}, use_dynamic=True)
    start = N_OBS_STEPS - 1
    max_exec = HORIZON - start
    assert torch.all(result['replan_steps'] == max_exec)
    assert result['action'].shape == (2, max_exec, ACTION_DIM)
    assert torch.all(result['disagreement'] == 0)
