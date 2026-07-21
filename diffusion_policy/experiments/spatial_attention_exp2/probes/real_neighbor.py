"""real_neighbor_diff probe (DEFAULT): probes are actual dataset observations o' from
the probe pool within max_radius of the query in the similarity space. Nothing synthetic
is injected; /d^2 uses the real similarity-space distance."""
import numpy as np

from diffusion_policy.experiments.spatial_attention_exp2.probes.base import ProbeType, ProbePlan


class RealNeighborDiff(ProbeType):
    name = 'real_neighbor_diff'

    def plan(self, query_row, index, direction, params):
        ns = index.select(
            query_row, direction, params['max_radius'], params['min_neighbors'],
            params['K'], params['same_episode_window'],
            params['project_out_temporal'], params['exclude_same_episode'])
        targets = [(int(index.episodes[r]), int(index.timesteps[r])) for r in ns.rows]
        denom = ns.dists ** 2                       # per-probe d(o, o')^2
        return ProbePlan('real', targets=targets, denom=denom,
                         flagged=ns.flagged, n_valid=ns.n_valid)
