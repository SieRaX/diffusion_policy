"""Build-or-load the full-dataset similarity index, with a reusable on-disk cache.

The cache is keyed by checkpoint + similarity_space (name/encoder/metric) + stride and lives
NEXT TO THE CHECKPOINT (not in output_dir), so ablation cells that differ only by probe_type
/ output_dir reuse it. `build_or_load_index` returns (index, reused: bool); a second
invocation with a matching probe pool loads instead of rebuilding.
"""
import os
import numpy as np

from diffusion_policy.experiments.spatial_attention_exp2.similarity.index import (
    Index, save_index, load_index,
)


def index_cache_path(checkpoint, space_name, encoder, metric, stride):
    base = os.path.splitext(checkpoint)[0]
    tag = f"{space_name}_{encoder}_{metric}_stride{int(stride)}"
    return os.path.join(base, 'exp2_index_cache', tag + '.npz')


def build_or_load_index(cache_path, pool_episodes, pool_timesteps, vectors_fn,
                        metric='euclidean', meta=None):
    """vectors_fn() -> RAW (P, dim) float array (built lazily only on a cache miss)."""
    pool_episodes = np.asarray(pool_episodes, dtype=np.int64)
    pool_timesteps = np.asarray(pool_timesteps, dtype=np.int64)
    if os.path.exists(cache_path):
        idx, _ = load_index(cache_path)
        if (np.array_equal(idx.episodes, pool_episodes)
                and np.array_equal(idx.timesteps, pool_timesteps)):
            return idx, True                       # reuse — do NOT rebuild
    V = np.asarray(vectors_fn(), dtype=np.float64)
    if str(metric) == 'cosine':
        V = V / np.clip(np.linalg.norm(V, axis=1, keepdims=True), 1e-12, None)
    idx = Index(V, pool_episodes, pool_timesteps, metric='euclidean')  # cosine folded into V
    os.makedirs(os.path.dirname(cache_path) or '.', exist_ok=True)
    save_index(cache_path, idx, pool_episodes, pool_timesteps, meta or {})
    return idx, False
