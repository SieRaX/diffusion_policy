"""Endpoint-level sensitivity S_end (validation, expensive).

    S_end(o) = mean_{o'} (1/N) Σ_j || f_θ(ε0_j, o') − f_θ(ε0_j, o) ||² / denom(o,o')

Reuses the prelim's coupled-endpoint metric: `CoupledEndpointDistance._sample` (full ODE
sampler with the fixed CRN ε0 injected on BOTH sides) and `_reduce` (sum-of-squares over
H and the selected action dims — include_gripper is already handled by `action_dims`). We
add the always-on /denom normalization PER PROBE before averaging, so Σ_k per_index == S,
in each configured distance space (raw / norm).
"""
import numpy as np

from diffusion_policy.experiments.spatial_attention_exp1.mse_metric.metric import _executed_start


class EndpointSensitivity:
    def __init__(self, metric):
        self.metric = metric                     # CoupledEndpointDistance (crn=eps0, action_dims set)

    def compute(self, policy, nominal_obs, probe_obs_list, denom):
        """One query state. Returns {space: {'S','S_first','per_index'(H,),'D_k'(K,)}}."""
        start = _executed_start(policy)
        spaces = self.metric.spaces
        H = int(self.metric.crn.horizon)
        K = len(probe_obs_list)
        samp_nom = self.metric._sample(policy, nominal_obs)             # {'raw','norm'} (N,H,D)
        acc = {sp: {'D_k': np.empty(K, np.float64), 'pi': np.zeros(H, np.float64)} for sp in spaces}
        for k, probe in enumerate(probe_obs_list):
            samp = self.metric._sample(policy, probe)
            dk = float(max(denom[k], 1e-12))
            for sp in spaces:
                d_scalar, pidx, _ = self.metric._reduce(samp[sp], samp_nom[sp], start)
                acc[sp]['D_k'][k] = d_scalar / dk
                acc[sp]['pi'] += pidx / dk
        res = {}
        for sp in spaces:
            per_index = acc[sp]['pi'] / max(K, 1)
            res[sp] = {'S': float(per_index.sum()), 'S_first': float(per_index[start]),
                       'per_index': per_index, 'D_k': acc[sp]['D_k']}
        return res


def stratified_subset(n_query, svel_scalar, phase_labels, frac, min_all, seed):
    """Select the S_end subset of query states: if n_query <= min_all, take ALL; else
    select frac of them stratified JOINTLY by phase label and S_vel quantile bin."""
    idx = np.arange(int(n_query))
    if n_query <= int(min_all):
        return idx
    rng = np.random.default_rng(int(seed))
    svel = np.asarray(svel_scalar, np.float64)
    # S_vel quantile bins (quartiles) x phase -> stratify
    ranks = np.argsort(np.argsort(svel))
    qbin = np.minimum((ranks * 4) // max(len(svel), 1), 3)
    strata = np.asarray(phase_labels, np.int64) * 4 + qbin
    target = int(round(float(frac) * n_query))
    picked = []
    for s in np.unique(strata):
        rows = idx[strata == s]
        take = max(1, int(round(len(rows) * frac)))
        picked.append(rng.choice(rows, size=min(take, len(rows)), replace=False))
    out = np.unique(np.concatenate(picked)) if picked else idx
    if len(out) > target:                       # trim to ~target deterministically
        out = np.sort(rng.choice(out, size=target, replace=False))
    return out
