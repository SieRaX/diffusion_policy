"""Full-dataset similarity index + neighbor selection with direction rules.

Pure numpy (no torch / robosuite) so it is fully unit-testable. A row is one
probe-pool state, identified by ``(episode, timestep)``; its ``vector`` is the
similarity-space representation (low_dim obs vector, image latent, or pixels).

Neighbor selection enforces the probe-direction rules RELATIVE TO THE QUERY
STATE'S EPISODE:
  - temporal : same episode, 0 < |Δt| <= same_episode_window;
  - config   : other episodes only, with the query's local temporal direction
               (central finite difference of the vector along its episode)
               projected out of every difference before measuring distance.
Distances are Euclidean in the similarity vector space (the space probes live
in); the caller asserts that. Candidates within ``max_radius`` are kept; if fewer
than ``min_neighbors`` remain the query is FLAGGED (fallback handles it).
"""
import numpy as np

DIRECTIONS = ('temporal', 'config', 'both')


class NeighborSet:
    """Selected neighbors for one query state along one direction."""
    def __init__(self, rows, deltas, dists, flagged, n_valid):
        self.rows = np.asarray(rows, dtype=np.int64)        # (k,) probe-pool row indices
        self.deltas = np.asarray(deltas, dtype=np.float64)  # (k, dim) difference vectors (temporal-projected for config)
        self.dists = np.asarray(dists, dtype=np.float64)    # (k,) euclidean norm of deltas (== d(o,o'))
        self.flagged = bool(flagged)                        # fewer than min_neighbors within radius
        self.n_valid = int(n_valid)                         # number of valid neighbors found (before capping to K)


class Index:
    def __init__(self, vectors, episodes, timesteps, metric='euclidean'):
        self.vectors = np.asarray(vectors, dtype=np.float64)     # (P, dim)
        self.episodes = np.asarray(episodes, dtype=np.int64)     # (P,)
        self.timesteps = np.asarray(timesteps, dtype=np.int64)   # (P,)
        assert self.vectors.ndim == 2
        assert len(self.episodes) == len(self.timesteps) == self.vectors.shape[0]
        assert metric in ('euclidean', 'cosine')
        self.metric = metric
        self.dim = self.vectors.shape[1]
        # per-episode row ordering by timestep (for the local temporal direction)
        self._by_ep = {}
        for ep in np.unique(self.episodes):
            rows = np.nonzero(self.episodes == ep)[0]
            self._by_ep[int(ep)] = rows[np.argsort(self.timesteps[rows])]

    # ---------------------------------------------------------------- lookups
    def row_of(self, episode, timestep):
        m = np.nonzero((self.episodes == int(episode)) & (self.timesteps == int(timestep)))[0]
        return int(m[0]) if len(m) else -1

    def query_rows(self, eval_episodes):
        """Rows whose episode is in eval_episodes, sorted by (episode, timestep)."""
        mask = np.isin(self.episodes, np.asarray(list(eval_episodes), dtype=np.int64))
        rows = np.nonzero(mask)[0]
        return rows[np.lexsort((self.timesteps[rows], self.episodes[rows]))]

    def local_temporal_dir(self, row):
        """Unit central-finite-difference of the vector along its episode at ``row``.
        Returns a zero vector if the neighbors are unavailable/degenerate."""
        ep = int(self.episodes[row])
        order = self._by_ep[ep]
        pos = int(np.nonzero(order == row)[0][0])
        lo = order[max(0, pos - 1)]
        hi = order[min(len(order) - 1, pos + 1)]
        d = self.vectors[hi] - self.vectors[lo]
        n = np.linalg.norm(d)
        return d / n if n > 0 else np.zeros(self.dim)

    # ---------------------------------------------------------------- neighbor selection
    def _candidate_rows(self, row, direction, same_episode_window, exclude_same_episode):
        ep = int(self.episodes[row]); t = int(self.timesteps[row])
        if direction == 'temporal':
            same = self.episodes == ep
            within = np.abs(self.timesteps - t) <= int(same_episode_window)
            cand = np.nonzero(same & within)[0]
            cand = cand[cand != row]
        elif direction == 'config':
            cand = np.nonzero(self.episodes != ep)[0]
        else:
            raise ValueError(f"direction must be temporal|config, got {direction!r}")
        if exclude_same_episode and direction != 'config':
            cand = cand[self.episodes[cand] != ep]
        return cand

    def select(self, row, direction, max_radius, min_neighbors, K,
               same_episode_window=10, project_out_temporal=True,
               exclude_same_episode=False):
        """NeighborSet for one query row along one direction (temporal | config)."""
        cand = self._candidate_rows(row, direction, same_episode_window, exclude_same_episode)
        if len(cand) == 0:
            return NeighborSet([], np.zeros((0, self.dim)), [], flagged=True, n_valid=0)
        deltas = self.vectors[cand] - self.vectors[row]              # (m, dim)
        if direction == 'config' and project_out_temporal:
            zhat = self.local_temporal_dir(row)                      # (dim,)
            if np.linalg.norm(zhat) > 0:
                deltas = deltas - np.outer(deltas @ zhat, zhat)      # remove along-task component
        dists = np.linalg.norm(deltas, axis=1)                       # (m,) euclidean == d(o,o')
        keep = dists <= float(max_radius)
        cand, deltas, dists = cand[keep], deltas[keep], dists[keep]
        n_valid = int(len(cand))
        flagged = n_valid < int(min_neighbors)
        if n_valid > int(K):                                         # keep the K nearest
            order = np.argsort(dists)[:int(K)]
            cand, deltas, dists = cand[order], deltas[order], dists[order]
        else:
            order = np.argsort(dists)
            cand, deltas, dists = cand[order], deltas[order], dists[order]
        return NeighborSet(cand, deltas, dists, flagged, n_valid)


def save_index(path, index, pool_episodes, pool_timesteps, meta):
    """Persist the index vectors + pool listing + metadata for cache reuse."""
    np.savez(path, vectors=index.vectors, episodes=index.episodes,
             timesteps=index.timesteps, metric=str(index.metric),
             pool_episodes=np.asarray(pool_episodes, dtype=np.int64),
             pool_timesteps=np.asarray(pool_timesteps, dtype=np.int64),
             **{f'meta_{k}': v for k, v in meta.items()})


def load_index(path):
    d = np.load(path, allow_pickle=True)
    idx = Index(d['vectors'], d['episodes'], d['timesteps'], metric=str(d['metric']))
    meta = {k[len('meta_'):]: d[k] for k in d.files if k.startswith('meta_')}
    return idx, meta
