"""Fallback for FLAGGED query states (fewer than min_neighbors valid neighbors within
max_radius). Modes:
  - default_horizon : mark invalid, exclude from downstream;
  - downweight      : compute anyway, attach confidence weight = neighbor count;
  - dtw (DEFAULT)   : substitute probes from DTW-phase-matched cross-demo pairs (same
                      distance-normalized /d^2 formula).

The dtw path here is a SIMPLIFIED phase-match: the K nearest OTHER-episode states sharing
the query's phase label, ignoring max_radius. TODO: true DTW time-warp alignment on the
dtw.features (ee_pose, gripper) with per_phase_sigma; phase-match is a functional proxy.
"""
import numpy as np

from diffusion_policy.experiments.spatial_attention_exp2.probes.base import ProbePlan


def apply_fallback(plan, query_row, index, params, ctx):
    if not plan.flagged:
        return plan
    mode = params['fallback']
    if mode == 'default_horizon':
        return ProbePlan('invalid', flagged=True, n_valid=plan.n_valid)
    if mode == 'downweight':
        plan.weight = float(max(plan.n_valid, 0))   # confidence = neighbor count
        return plan                                  # still flagged; computed anyway
    if mode == 'dtw':
        return _dtw_phase_matched(query_row, index, params, ctx)
    raise ValueError(f"unknown fallback {mode!r} (dtw | default_horizon | downweight)")


def _dtw_phase_matched(query_row, index, params, ctx):
    pool_phases = ctx['pool_phases']                 # (P,) int phase label per pool row
    qphase = int(pool_phases[query_row])
    qep = int(index.episodes[query_row])
    cand = np.nonzero((index.episodes != qep) & (pool_phases == qphase))[0]
    if len(cand) == 0:
        return ProbePlan('invalid', flagged=True, n_valid=0)
    deltas = index.vectors[cand] - index.vectors[query_row]
    dists = np.linalg.norm(deltas, axis=1)
    order = np.argsort(dists)[:int(params['K'])]
    cand, dists = cand[order], dists[order]
    targets = [(int(index.episodes[r]), int(index.timesteps[r])) for r in cand]
    return ProbePlan('real', targets=targets, denom=dists ** 2, flagged=True, n_valid=int(len(cand)))
