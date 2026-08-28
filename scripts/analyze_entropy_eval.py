"""Assess AAC eval caveats from an eval_entropy_by_seed output dir.

Usage:
    python scripts/analyze_entropy_eval.py <output_dir> [<output_dir> ...]

For each dir, reads entropy_histories.pickle + eval_log.json and reports:
  - mean scores (train/test)
  - raw h* distribution (from the elbow rule, before the wrapper's max(8,.) floor)
  - fraction of decisions clipped by the floor of 8 (i.e. raw h* < 8)
  - fraction where h* hit the max executable length
  - NaN/Inf occurrences in avg_entropy / diffs
"""
import sys
import json
import pickle
import pathlib

import numpy as np


def analyze(output_dir):
    output_dir = pathlib.Path(output_dir)
    print(f"\n{'='*70}\n{output_dir}\n{'='*70}")

    log_path = output_dir / 'eval_log.json'
    if log_path.exists():
        log = json.load(open(log_path))
        print(f"test/mean_score: {log.get('test/mean_score')}   "
              f"train/mean_score: {log.get('train/mean_score')}")
    else:
        print("eval_log.json not found (run incomplete?)")

    pkl_path = output_dir / 'entropy_histories.pickle'
    if not pkl_path.exists():
        print("entropy_histories.pickle not found")
        return
    hists = pickle.load(open(pkl_path, 'rb'))
    hists = [h for h in hists if h is not None]

    all_h_star = []
    n_nan_avg = n_inf_avg = n_nan_diff = 0
    n_decisions = 0
    max_len = 0
    for h in hists:
        all_h_star.extend(h['h_star'])
        for a in h['avg_entropy']:
            a = np.asarray(a)
            max_len = max(max_len, len(a))
            n_nan_avg += int(np.isnan(a).any())
            n_inf_avg += int(np.isinf(a).any())
        for d in h['diffs']:
            n_nan_diff += int(~np.isfinite(np.asarray(d)).all())
        n_decisions += len(h['h_star'])

    hs = np.asarray(all_h_star)
    if len(hs) == 0:
        print("no h* decisions recorded")
        return
    print(f"envs: {len(hists)}   decisions: {n_decisions}   "
          f"executable horizon: {max_len}")
    print(f"raw h* (elbow rule): mean={hs.mean():.2f} median={np.median(hs):.0f} "
          f"min={hs.min()} max={hs.max()}")
    qs = np.percentile(hs, [10, 25, 50, 75, 90]).astype(int)
    print(f"percentiles 10/25/50/75/90: {qs.tolist()}")
    exec_h = np.clip(hs, 8, max_len)
    print(f"executed chunk (max(8,h*)): mean={exec_h.mean():.2f}")
    clipped = (hs < 8).mean()
    print(f"CAVEAT floor-of-8: {clipped*100:.1f}% of decisions had raw h*<8 "
          f"(floor decided, not entropy)")
    at_max = (hs >= max_len).mean()
    print(f"h* at executable max ({max_len}): {at_max*100:.1f}%")
    print(f"CAVEAT NaN/Inf: avg_entropy chunks with NaN: {n_nan_avg}, "
          f"with Inf: {n_inf_avg}, diffs non-finite: {n_nan_diff} "
          f"(out of {n_decisions})")
    # histogram
    bins = [1, 2, 4, 8, 12, 16, 24, 32, max(48, max_len + 1)]
    counts, edges = np.histogram(hs, bins=bins)
    print("h* histogram:")
    for c, lo, hi in zip(counts, edges[:-1], edges[1:]):
        bar = '#' * int(60 * c / max(counts.max(), 1))
        print(f"  [{int(lo):3d},{int(hi):3d}): {c:5d} {bar}")


if __name__ == '__main__':
    for d in sys.argv[1:]:
        analyze(d)
