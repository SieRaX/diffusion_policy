from typing import Dict, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import reduce

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy
from diffusion_policy.model.diffusion.moh_conditional_unet1d import MoHConditionalUnet1D


class MoHDiffusionUnetLowdimPolicy(BaseLowdimPolicy):
    """
    Mixture-of-Horizons (Jing et al., ICML 2026) baseline on top of
    DiffusionUnetLowdimPolicy.

    Differences from the original MoH (full-attention action transformer on a
    VLA), documented for the paper:
    - The shared backbone is ConditionalUnet1D (CNN). A CNN has no attention
      mask, so the candidate horizons cannot be batched with per-horizon masks;
      instead each horizon h runs a separate forward over the truncated noisy
      chunk x_t[:, :h]. This is mathematically equivalent to MoH's masked
      parallel forward but costs len(horizons) UNet passes.
    - The UNet downsamples twice (stride 2), so every candidate horizon must be
      a multiple of 4 (MoH's stride-3 set {3,6,...} is replaced by stride-4).
    - The gate head is a linear layer on the UNet's final per-step feature
      (analogous to MoH's linear gate on the action transformer hidden state).
    - The sampler is DDPM/DDIM instead of a flow-matching ODE. Gate fusion is
      applied to the epsilon prediction at every denoise step. The dynamic
      inference disagreement is accumulated per denoise step in prev-sample
      space via the (unclipped) DDIM linearization: |c2(t)| * l1(eps_h - eps_fused),
      mirroring MoH's per-ODE-step |indiv_iter - fused_iter| accumulation.

    Only obs_as_global_cond=True is supported: MoH operates on pure action
    chunks (observation enters as context), matching global conditioning.
    """
    def __init__(self,
            model: MoHConditionalUnet1D,
            noise_scheduler,
            horizon,
            horizons: List[int],
            obs_dim,
            action_dim,
            n_action_steps,
            n_obs_steps,
            num_inference_steps=None,
            obs_as_global_cond=True,
            oa_step_convention=True,
            # MoH training (paper Eq. 17)
            lambda_ind=1.0,
            lambda_bal=1.0e-3,
            use_gate_noise=True,
            # MoH dynamic inference (paper Alg. 1)
            min_replan_steps=4,
            min_active_horizons=2,
            scale_ratio=1.1,
            downsample_factor=4,
            # parameters passed to scheduler.step
            **kwargs):
        super().__init__()
        assert obs_as_global_cond, \
            "MoH baseline only supports obs_as_global_cond=True"
        horizons = sorted(int(h) for h in horizons)
        assert len(horizons) >= 1
        assert horizons[-1] == horizon, \
            f"max candidate horizon {horizons[-1]} must equal prediction horizon {horizon}"
        assert all(h % downsample_factor == 0 for h in horizons), \
            f"all horizons must be multiples of {downsample_factor} (UNet downsampling), got {horizons}"
        assert len(set(horizons)) == len(horizons)

        self.model = model
        self.noise_scheduler = noise_scheduler
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.horizons = horizons
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.oa_step_convention = oa_step_convention
        self.lambda_ind = lambda_ind
        self.lambda_bal = lambda_bal
        self.use_gate_noise = use_gate_noise
        self.min_replan_steps = min_replan_steps
        self.min_active_horizons = min_active_horizons
        self.scale_ratio = scale_ratio
        self.kwargs = kwargs

        # gate head on per-step UNet features -> one logit per (step, horizon
        # branch). Zero-init so gating starts exactly uniform.
        self.gate_head = nn.Linear(model.feature_dim, 1)
        nn.init.zeros_(self.gate_head.weight)
        nn.init.zeros_(self.gate_head.bias)
        # learnable noise std for noisy gating (MoH use_gate_noise)
        self.gate_noise_head = nn.Linear(model.feature_dim, 1)
        nn.init.zeros_(self.gate_noise_head.weight)
        nn.init.zeros_(self.gate_noise_head.bias)

        # validity mask: valid[k, i] = step k is covered by horizons[i]
        step_idx = torch.arange(horizon).unsqueeze(1)          # (T, 1)
        horizon_t = torch.tensor(horizons).unsqueeze(0)        # (1, N)
        self.register_buffer('valid_mask', step_idx < horizon_t, persistent=False)
        # number of active horizons per step
        self.register_buffer(
            'n_active', self.valid_mask.sum(dim=-1), persistent=False)

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

        self.last_loss_info = dict()

    # ========= MoH internals ============
    def _forward_all_horizons(self, trajectory, timesteps, global_cond):
        """Run the shared UNet once per candidate horizon on the truncated
        chunk. Returns (preds, feats): lists indexed like self.horizons with
        preds[i]: (B, h_i, Da), feats[i]: (B, h_i, C)."""
        preds, feats = [], []
        for h in self.horizons:
            pred, feat = self.model(
                trajectory[:, :h], timesteps,
                global_cond=global_cond, return_features=True)
            preds.append(pred)
            feats.append(feat)
        return preds, feats

    def _gate_weights(self, feats, training):
        """Per-step softmax over valid horizons (MoH Eq. 12).
        feats: list of (B, h_i, C). Returns alpha (B, T, N)."""
        B = feats[0].shape[0]
        T = self.horizon
        N = len(self.horizons)
        device = feats[0].device
        dtype = feats[0].dtype
        neg_inf = torch.finfo(dtype).min
        logits = torch.full((B, T, N), neg_inf, device=device, dtype=dtype)
        for i, h in enumerate(self.horizons):
            g = self.gate_head(feats[i]).squeeze(-1)  # (B, h)
            if training and self.use_gate_noise:
                std = F.softplus(self.gate_noise_head(feats[i]).squeeze(-1))
                g = g + torch.randn_like(g) * std
            logits[:, :h, i] = g
        alpha = torch.softmax(logits, dim=-1)
        return alpha

    def _fuse(self, preds, alpha):
        """Gate-weighted fusion of per-horizon predictions (MoH Eq. 13).
        Returns (B, T, Da)."""
        B = preds[0].shape[0]
        T = self.horizon
        Da = preds[0].shape[-1]
        stacked = torch.zeros(
            (B, T, len(self.horizons), Da),
            device=preds[0].device, dtype=preds[0].dtype)
        for i, h in enumerate(self.horizons):
            stacked[:, :h, i] = preds[i]
        return (alpha.unsqueeze(-1) * stacked).sum(dim=2)

    def _balance_loss(self, alpha):
        """MoH Eq. 14-16: mean squared coefficient of variation of average
        horizon usage over each interval (h_{i-1}, h_i] with >1 active horizon."""
        eps = 1e-8
        boundaries = [0] + self.horizons
        cv2_list = []
        for i in range(len(self.horizons)):
            lo, hi = boundaries[i], boundaries[i+1]
            # horizons active on steps [lo, hi): indices i..N-1
            active = alpha[:, lo:hi, i:]          # (B, S_i, N-i)
            if active.shape[-1] <= 1 or active.shape[1] == 0:
                continue
            avg_usage = active.mean(dim=(0, 1))    # (N-i,)
            cv2 = avg_usage.var(unbiased=False) / (avg_usage.mean()**2 + eps)
            cv2_list.append(cv2)
        if len(cv2_list) == 0:
            return alpha.new_zeros(())
        return torch.stack(cv2_list).mean()

    def _ddim_prev_eps_coeff(self, t):
        """|c2(t)| where x_{t-1} = c1(t) x_t + c2(t) eps under (eta=0, unclipped)
        DDIM. Used to scale epsilon-space disagreement into prev-sample space."""
        scheduler = self.noise_scheduler
        alphas_cumprod = scheduler.alphas_cumprod
        t = int(t)
        step_ratio = scheduler.config.num_train_timesteps // self.num_inference_steps
        prev_t = t - step_ratio
        a_t = alphas_cumprod[t]
        if prev_t >= 0:
            a_prev = alphas_cumprod[prev_t]
        else:
            a_prev = getattr(scheduler, 'final_alpha_cumprod',
                             torch.ones_like(a_t))
        c2 = (1 - a_prev).sqrt() - (a_prev * (1 - a_t) / a_t).sqrt()
        return c2.abs().item()

    # ========= inference ============
    def conditional_sample(self, cond_shape, global_cond,
            generator=None, accumulate_disagreement=False, **kwargs):
        model = self.model
        scheduler = self.noise_scheduler

        trajectory = torch.randn(
            size=cond_shape,
            dtype=self.dtype,
            device=self.device,
            generator=generator)

        scheduler.set_timesteps(self.num_inference_steps)

        disagreement = None
        if accumulate_disagreement:
            disagreement = torch.zeros(
                cond_shape[:2], device=self.device, dtype=self.dtype)

        for t in scheduler.timesteps:
            preds, feats = self._forward_all_horizons(trajectory, t, global_cond)
            alpha = self._gate_weights(feats, training=False)
            fused = self._fuse(preds, alpha)

            if accumulate_disagreement:
                # gate-weighted l1 between per-horizon and fused epsilon,
                # scaled by the DDIM prev-sample coefficient (cf. MoH's
                # per-ODE-step |indiv_iter - fused_iter| accumulation)
                c2 = self._ddim_prev_eps_coeff(t)
                for i, h in enumerate(self.horizons):
                    diff = (preds[i] - fused[:, :h]).abs().sum(dim=-1)  # (B, h)
                    disagreement[:, :h] += c2 * alpha[:, :h, i] * diff

            trajectory = scheduler.step(
                fused, t, trajectory,
                generator=generator,
                **kwargs).prev_sample

        return trajectory, disagreement

    def _consensus_exec_steps(self, disagreement, start):
        """MoH Alg. 1 adapted: given per-step accumulated disagreement (B, T)
        and the execution start index, return per-sample executable step counts
        (B,) in the executed frame."""
        B, T = disagreement.shape
        n = self.min_replan_steps
        m = self.min_active_horizons
        max_exec = T - start
        if max_exec <= n:
            return torch.full((B,), max_exec,
                dtype=torch.long, device=disagreement.device)

        d = disagreement[:, start:]                      # (B, max_exec)
        thres = d[:, :n].mean(dim=1) * self.scale_ratio  # (B,)
        # step j (executed frame) is extendable if disagreement stays below
        # threshold and enough horizons are still active at global index start+j
        active = self.n_active[start + n: start + max_exec]      # (max_exec-n,)
        ok = (d[:, n:] <= thres.unsqueeze(1)) \
            & (active.unsqueeze(0) > m)
        n_exec = n + ok.long().cumprod(dim=1).sum(dim=1)
        return n_exec

    def predict_action(self, obs_dict: Dict[str, torch.Tensor],
            use_dynamic=False) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key.
        With use_dynamic=True, also returns per-sample "replan_steps" and cuts
        "action" to the batch-min consensus prefix (MoH dynamic inference).
        """
        assert 'obs' in obs_dict
        assert 'past_action' not in obs_dict  # not implemented yet
        nobs = self.normalizer['obs'].normalize(obs_dict['obs'])
        B, _, Do = nobs.shape
        To = self.n_obs_steps
        assert Do == self.obs_dim
        T = self.horizon
        Da = self.action_dim

        global_cond = nobs[:, :To].reshape(nobs.shape[0], -1)

        nsample, disagreement = self.conditional_sample(
            (B, T, Da), global_cond,
            accumulate_disagreement=use_dynamic,
            **self.kwargs)

        action_pred = self.normalizer['action'].unnormalize(nsample)

        start = To
        if self.oa_step_convention:
            start = To - 1

        result = {'action_pred': action_pred}
        if use_dynamic:
            n_exec = self._consensus_exec_steps(disagreement, start)
            # vectorized envs step in lockstep -> execute the batch-min prefix;
            # single-env eval (B=1) recovers MoH's per-episode behavior
            n_exec_common = int(n_exec.min().item())
            result['action'] = action_pred[:, start:start + n_exec_common]
            result['replan_steps'] = n_exec
            result['disagreement'] = disagreement
        else:
            end = start + self.n_action_steps
            result['action'] = action_pred[:, start:end]
        return result

    # ========= training ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        # normalize input
        assert 'valid_mask' not in batch
        nbatch = self.normalizer.normalize(batch)
        obs = nbatch['obs']
        action = nbatch['action']
        trajectory = action
        assert trajectory.shape[1] == self.horizon

        global_cond = obs[:, :self.n_obs_steps, :].reshape(obs.shape[0], -1)

        # forward diffusion on the full-length chunk; every candidate horizon
        # sees the same noise/timestep on its truncated prefix (mirrors MoH
        # sharing one time variable across horizon branches)
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        bsz = trajectory.shape[0]
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (bsz,), device=trajectory.device
        ).long()
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise, timesteps)

        pred_type = self.noise_scheduler.config.prediction_type
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        preds, feats = self._forward_all_horizons(
            noisy_trajectory, timesteps, global_cond)

        # L_ind: sum of per-horizon losses (MoH L_ind = sum_h L^(h))
        loss_ind = trajectory.new_zeros(())
        for i, h in enumerate(self.horizons):
            loss_h = F.mse_loss(preds[i], target[:, :h], reduction='none')
            loss_ind = loss_ind + reduce(loss_h, 'b ... -> b (...)', 'mean').mean()

        # L_mix on gated fusion
        alpha = self._gate_weights(feats, training=self.training)
        fused = self._fuse(preds, alpha)
        loss_mix = F.mse_loss(fused, target, reduction='none')
        loss_mix = reduce(loss_mix, 'b ... -> b (...)', 'mean').mean()

        # L_bal
        loss_bal = self._balance_loss(alpha)

        loss = loss_mix + self.lambda_ind * loss_ind + self.lambda_bal * loss_bal
        self.last_loss_info = {
            'loss_mix': loss_mix.item(),
            'loss_ind': loss_ind.item(),
            'loss_bal': loss_bal.item(),
        }
        return loss
