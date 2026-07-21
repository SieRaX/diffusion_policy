"""Probe interface. A probe TYPE turns a query state (+ the neighbor index) into a
``ProbePlan`` — an env-free description of the K probes and their /denom values. The
runner materializes the actual observations (real neighbor env obs, or nominal + delta),
so all probe logic here is pure numpy and unit-testable.

Probe directions (temporal | config) are handled by ``Index.select``; ``both`` is a
map-level weighted sum in the runner (never pooled into one probe set).

Denominator (always-on distance normalization):
  - real_neighbor_diff : per-probe d(o, o')^2 (similarity-space distance);
  - diag/fullcov noise : E||delta||^2 (empirical mean over the K sampled deltas).
"""
import numpy as np


class ProbePlan:
    """kind:
      'real'    -> materialize probe obs from env at each (episode, timestep) in targets;
      'noise'   -> materialize probe obs as nominal_obs + delta (low_dim only);
      'invalid' -> flagged and excluded (default_horizon fallback)."""
    def __init__(self, kind, targets=None, deltas=None, denom=None,
                 flagged=False, n_valid=0, weight=1.0):
        self.kind = kind
        self.targets = list(targets) if targets is not None else []      # [(ep, t), ...] for 'real'
        self.deltas = np.asarray(deltas, dtype=np.float64) if deltas is not None else None  # (K, Do) 'noise'
        self.denom = np.asarray(denom, dtype=np.float64) if denom is not None else None      # (K,) per-probe /denom
        self.flagged = bool(flagged)
        self.n_valid = int(n_valid)
        self.weight = float(weight)      # downweight fallback attaches confidence = neighbor count

    @property
    def K(self):
        if self.kind == 'real':
            return len(self.targets)
        if self.kind == 'noise':
            return 0 if self.deltas is None else int(self.deltas.shape[0])
        return 0


class ProbeType:
    """Interface. Register new probe types by subclassing and dispatching on cfg name."""
    name = 'base'

    def plan(self, query_row, index, direction, cfg, ctx):
        """Return a ProbePlan for one query row along one direction.
        ``ctx`` carries run-level context (rng seed base, obs_dim, pool_phases, ...)."""
        raise NotImplementedError


def make_probe_type(name):
    from diffusion_policy.experiments.spatial_attention_exp2.probes.real_neighbor import RealNeighborDiff
    from diffusion_policy.experiments.spatial_attention_exp2.probes.noise import DiagStdNoise, FullCovNoise
    table = {
        'real_neighbor_diff': RealNeighborDiff,
        'diag_std_noise': DiagStdNoise,
        'fullcov_noise': FullCovNoise,
    }
    if name not in table:
        raise ValueError(f"unknown probe_type {name!r} ({sorted(table)})")
    return table[name]()
