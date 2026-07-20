"""Coupled endpoint-distance sensitivity metric (common random numbers).

For a timestep with nominal input `o` and perturbed inputs `{o'_k}`, using a FIXED
initial-noise set {eps0_j} (j=1..N) reused on both sides:

    D_k = (1/N) Σ_j || f(eps0_j, o'_k) − f(eps0_j, o) ||^2      (|| ||^2 = sum over H×D)
    S   = (1/K) Σ_k D_k

`f` is the full ODE sampler; identical eps0 on both sides makes the difference
reflect the input change, not sampling luck. All values are computed in BOTH the
raw (unnormalized) and normalized action spaces from a single sampler pass
(array names carry `_raw` / `_norm`).

Norm convention: sum-of-squares over the chunk (not mean), with means only over N
and K. This makes `Σ_k per_index[k] == S` (the per-index reduction consistency
check) hold exactly. `S_first` is the per-chunk-index value at the executed-slice
start index. Reuses Exp1's CRN tiling, `_executed_start`, and the forced-ODE-steps /
injected-noise context managers.

TODO: assumes a DETERMINISTIC sampler (flow matching: fixing eps0 fixes the run).
A stochastic sampler (DDPM) would require coupling the full per-step noise
realization, not just eps0.
"""
import numpy as np
import torch

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.experiments.spatial_attention_exp1.mse_metric.metric import (
    _executed_start, _forced_ode_steps, _injected_noise,
)

SPACES = ('raw', 'norm')


class CoupledEndpointDistance:
    def __init__(self, crn, ode_steps, distance_space='both', max_batch=64, action_dims=None):
        self.crn = crn
        self.ode_steps = int(ode_steps)
        assert distance_space in ('both', 'raw', 'norm')
        self.spaces = SPACES if distance_space == 'both' else (distance_space,)
        self.max_batch = int(max_batch)
        # optional subset of action dims the distance is measured over (None = all dims).
        # e.g. drop the gripper channel, whose near-binary switch otherwise dominates S.
        self.action_dims = list(action_dims) if action_dims is not None else None

    @torch.no_grad()
    def _sample(self, policy, obs_dict):
        """Sample N action chunks for a single (B=1) input under CRN eps0.
        Returns {'raw': (N,H,D), 'norm': (N,H,D)} on CPU float64."""
        assert hasattr(policy, '_init_noise'), \
            "policy must expose `_init_noise` (Exp1/Perturb subclass) for CRN injection"
        device = policy.device
        eps0 = self.crn.eps0.to(device=device, dtype=policy.dtype)   # (N, H, D)
        N = eps0.shape[0]
        raw_chunks, norm_chunks = [], []
        for s in range(0, N, self.max_batch):
            e = min(N, s + self.max_batch)
            c = e - s
            obs_rep = dict_apply(obs_dict, lambda x: x.repeat_interleave(c, dim=0).to(device))
            with _forced_ode_steps(policy, self.ode_steps), _injected_noise(policy, eps0[s:e]):
                out = policy.predict_action(obs_rep)
            raw_chunks.append(out['action_pred'].double().cpu())      # (c, H, D)
            norm_chunks.append(out['naction_pred'].double().cpu())    # (c, H, D)
        return {'raw': torch.cat(raw_chunks, 0), 'norm': torch.cat(norm_chunks, 0)}

    def _reduce(self, samp_pert, samp_nom, start):
        """Given two (N,H,D) sample sets, return (D_scalar, per_index (H,), first_scalar)
        using sum-of-squares over (H, selected D), mean over N."""
        diff2 = (samp_pert - samp_nom) ** 2                # (N, H, D)
        if self.action_dims is not None:
            diff2 = diff2[..., self.action_dims]           # measure only the selected action dims
        per_index = diff2.sum(dim=2).mean(dim=0).numpy()   # (H,)  sum over D, mean over N
        d_scalar = float(per_index.sum())                  # sum over H  == Σ_k per_index[k]
        first_scalar = float(per_index[start])             # executed-slice first action
        return d_scalar, per_index, first_scalar

    def compute(self, policy, nominal_obs, perturbed_obs_list, compute_control=False):
        """One timestep. Returns a dict keyed by space with:
        S, S_first, per_index (H,), D_k (K,), and (if compute_control) control."""
        start = _executed_start(policy)
        H = self.crn.horizon
        K = len(perturbed_obs_list)

        samp_nom = self._sample(policy, nominal_obs)
        samp_pert = [self._sample(policy, o) for o in perturbed_obs_list]

        result = {}
        for space in self.spaces:
            D_k = np.empty(K, dtype=np.float64)
            per_index_acc = np.zeros(H, dtype=np.float64)
            for k in range(K):
                d, pidx, _ = self._reduce(samp_pert[k][space], samp_nom[space], start)
                D_k[k] = d
                per_index_acc += pidx
            per_index = per_index_acc / max(K, 1)
            S = float(D_k.mean()) if K > 0 else float('nan')
            S_first = float(per_index[start])
            entry = {'S': S, 'S_first': S_first, 'per_index': per_index, 'D_k': D_k}
            result[space] = entry

        if compute_control:
            # nominal vs a second nominal pass, both under eps0 -> ~0 if CRN is wired
            samp_nom2 = self._sample(policy, nominal_obs)
            for space in self.spaces:
                c, _, _ = self._reduce(samp_nom2[space], samp_nom[space], start)
                result[space]['control'] = float(c)
        return result
