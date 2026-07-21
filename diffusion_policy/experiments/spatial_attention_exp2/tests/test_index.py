"""Similarity index: probe-pool/query separation, direction rules (temporal window,
config other-episodes + temporal projection), active-space distances, flagged logic."""
import numpy as np

from diffusion_policy.experiments.spatial_attention_exp2.similarity.index import Index

# ep0 moves along +x; ep1 sits at x=3 and moves along +y.
EPS = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
TS = np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4])
VEC = np.array([[0, 0], [1, 0], [2, 0], [3, 0], [4, 0],
                [3, 0], [3, 1], [3, 2], [3, 3], [3, 4]], float)
QROW = 2  # query = (ep0, t=2), vector (2,0)


def _idx():
    return Index(VEC, EPS, TS, metric='euclidean')


def test_query_rows_only_eval_episodes():
    idx = _idx()
    q = idx.query_rows([0])
    assert set(idx.episodes[q]) == {0}
    assert len(q) == 5


def test_local_temporal_direction():
    idx = _idx()
    z = idx.local_temporal_dir(QROW)
    assert np.allclose(z, [1, 0])          # central diff of ep0 along +x


def test_temporal_neighbors_same_episode_window():
    idx = _idx()
    ns = idx.select(QROW, 'temporal', max_radius=10, min_neighbors=1, K=10,
                    same_episode_window=1, project_out_temporal=True)
    assert set(idx.episodes[ns.rows]) == {0}                 # same episode only
    assert set(idx.timesteps[ns.rows]) == {1, 3}             # |Δt| <= 1, excluding self
    assert np.allclose(ns.dists, np.linalg.norm(ns.deltas, axis=1))  # d == ||δ|| (active space)


def test_config_neighbors_other_episodes_and_projection():
    idx = _idx()
    ns = idx.select(QROW, 'config', max_radius=10, min_neighbors=1, K=10,
                    project_out_temporal=True)
    assert set(idx.episodes[ns.rows]) == {1}                 # other episodes only
    assert np.allclose(ns.deltas[:, 0], 0.0, atol=1e-9)      # temporal (+x) component removed
    assert np.allclose(ns.dists, np.abs(ns.deltas[:, 1]))    # remaining is the +y (config) part


def test_flagged_when_too_few_within_radius():
    idx = _idx()
    ns = idx.select(QROW, 'config', max_radius=0.5, min_neighbors=3, K=10,
                    project_out_temporal=True)
    assert ns.flagged and ns.n_valid < 3
