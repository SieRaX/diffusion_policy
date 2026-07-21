"""Noise-probe ablations (LOW_DIM only — the similarity space must equal the model's
observation space so that o + delta is a valid input). Fit a noise model from the query's
neighbor DIFFERENCES, then probe with o' = o + delta. The /d^2 normalization is replaced
by /E||delta||^2 (empirical mean over the K sampled deltas)."""
import numpy as np

from diffusion_policy.experiments.spatial_attention_exp2.probes.base import ProbeType, ProbePlan

_HUGE = 10 ** 9  # keep ALL valid neighbors for fitting (index.select caps at this)


class _NoiseBase(ProbeType):
    def plan(self, query_row, index, direction, params):
        ns = index.select(
            query_row, direction, params['max_radius'], params['min_neighbors'],
            _HUGE, params['same_episode_window'],
            params['project_out_temporal'], params['exclude_same_episode'])
        if ns.flagged or ns.n_valid < 2:            # need >=2 diffs to fit a noise model
            return ProbePlan('noise', deltas=np.zeros((0, index.dim)), denom=np.zeros(0),
                             flagged=True, n_valid=ns.n_valid)
        rng = np.random.default_rng([int(params['seed_base']), int(query_row)])
        delta = self._sample(ns.deltas, int(params['K']), rng, params)      # (K, dim)
        if direction == 'config' and params['project_out_temporal']:
            zhat = index.local_temporal_dir(query_row)
            if np.linalg.norm(zhat) > 0:
                delta = delta - np.outer(delta @ zhat, zhat)
        denom = np.full(int(params['K']), float((delta ** 2).sum(axis=1).mean()))  # E||delta||^2
        return ProbePlan('noise', deltas=delta, denom=denom, flagged=False, n_valid=ns.n_valid)

    def _sample(self, deltas, K, rng, params):
        raise NotImplementedError


class DiagStdNoise(_NoiseBase):
    name = 'diag_std_noise'

    def _sample(self, deltas, K, rng, params):
        sigma = deltas.std(axis=0)                  # per-dim std of the neighbor differences
        z = rng.standard_normal((K, deltas.shape[1]))
        return z * sigma[None, :]


class FullCovNoise(_NoiseBase):
    name = 'fullcov_noise'

    def _sample(self, deltas, K, rng, params):
        rank = int(params.get('rank', 16))
        X = deltas - deltas.mean(axis=0, keepdims=True)
        _, s, Vt = np.linalg.svd(X, full_matrices=False)
        m = X.shape[0]
        r = min(rank, Vt.shape[0])
        comp = Vt[:r]                               # (r, dim) principal directions
        scale = s[:r] / np.sqrt(max(m - 1, 1))      # per-component std
        z = rng.standard_normal((K, r))
        return (z * scale[None, :]) @ comp          # ~ N(0, low-rank Sigma)
