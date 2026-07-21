"""Velocity-level sensitivity S_vel (primary, cheap).

    S_vel(o) = mean_{o'} (1/J) Σ_j || v_θ(x_τj, τj, o') − v_θ(x_τj, τj, o) ||² / denom(o,o')
    x_τj = (1−τj)·a_GT_norm(o) + τj·ε_j     (IDENTICAL x_τj on both sides — CRN)

The fixed {(τj, εj)} come from CRNManager.taus / .eps_fm (num_fm=num_tz_draws,
fm_seed=seeds.tz_draws); a_GT is the query's demonstrator action chunk in NORMALIZED
action space (the velocity field lives there — no raw variant). Reductions: scalar (sum
over H and selected action dims), per chunk index k, and S_first at the executed-slice
start (_executed_start). include_gripper → drop the last action dim from the residual.

Reductions divide by denom PER PROBE before averaging over probes, so Σ_k per_index == S.
Reuses `policy.model` + `policy.fm_global_cond` (Exp1 FM subclass) and Exp1's
`_executed_start`.
"""
import numpy as np
import torch

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.experiments.spatial_attention_exp1.mse_metric.metric import _executed_start


def cat_obs(obs_dicts):
    """Concatenate a list of batch-1 predict_action obs_dicts along the batch dim.
    lowdim: {'obs': (M,To,Do)}; image: {key: (M,To,...)}."""
    keys = obs_dicts[0].keys()
    return {k: torch.cat([o[k] for o in obs_dicts], dim=0) for k in keys}


class VelocitySensitivity:
    def __init__(self, crn_vel, action_dims=None, max_batch=64):
        self.crn = crn_vel                       # CRNManager with .taus (J,), .eps_fm (J,H,D)
        self.action_dims = action_dims           # include_gripper subset (None = all)
        self.max_batch = int(max_batch)

    @torch.no_grad()
    def compute(self, policy, nominal_obs, probe_obs_list, a_gt_norm, denom):
        """One query state. probe_obs_list: K batch-1 obs_dicts. a_gt_norm: (H,Da)
        NORMALIZED GT action chunk. denom: (K,). Returns {'scalar','per_index'(H,),'first'}."""
        device, dtype = policy.device, policy.dtype
        H = int(self.crn.horizon)
        start = _executed_start(policy)
        K = len(probe_obs_list)
        assert K > 0, "no probes for this query state"
        taus = self.crn.taus.to(device=device, dtype=dtype)          # (J,)
        eps_fm = self.crn.eps_fm.to(device=device, dtype=dtype)      # (J,H,D)
        J = int(self.crn.num_fm)

        gc = policy.fm_global_cond(dict_apply(cat_obs([nominal_obs] + probe_obs_list),
                                              lambda x: x.to(device)))   # (K+1, gdim)
        a = a_gt_norm.to(device=device, dtype=dtype).unsqueeze(0)        # (1,H,Da)
        acc = torch.zeros(K, H, dtype=torch.float64, device=device)     # Σ_j sum_selD (Δv)^2
        for j in range(J):
            tau = taus[j]
            x = (1.0 - tau) * a + tau * eps_fm[j].unsqueeze(0)          # (1,H,Da)
            tval = float(tau) * getattr(policy, 'time_scale', 1.0)
            vn = policy.model(x, torch.full((1,), tval, device=device, dtype=dtype),
                              global_cond=gc[:1])                        # (1,H,Da)
            for c0 in range(0, K, self.max_batch):
                c1 = min(K, c0 + self.max_batch); cc = c1 - c0
                xr = x.expand(cc, *x.shape[1:])
                t = torch.full((cc,), tval, device=device, dtype=dtype)
                vp = policy.model(xr, t, global_cond=gc[1 + c0:1 + c1])  # (cc,H,Da)
                d2 = (vp - vn) ** 2                                      # (cc,H,Da)
                if self.action_dims is not None:
                    d2 = d2[..., self.action_dims]
                acc[c0:c1] += d2.sum(dim=2).double()                    # sum over selected dims -> (cc,H)
        acc /= J
        denom_t = torch.as_tensor(np.clip(np.asarray(denom, np.float64), 1e-12, None),
                                  device=device).unsqueeze(1)           # (K,1)
        per_index = (acc / denom_t).mean(dim=0).cpu().numpy()           # /denom per probe, mean over K
        return {'scalar': float(per_index.sum()), 'per_index': per_index,
                'first': float(per_index[start])}
