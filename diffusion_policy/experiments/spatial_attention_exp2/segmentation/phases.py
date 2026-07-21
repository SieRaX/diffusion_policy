"""Deterministic task-phase segmentation (no learning, no VLM).

An episode is segmented into APPROACH / GRASP / TRANSPORT / PLACE from the gripper
command transitions and grasp intervals. `segment_episode` is pure (takes per-timestep
`gripper_open` and `grasped` arrays) so it is unit-testable; the runner supplies those
arrays from the dataset gripper command and (for the query episodes) the prelim grasp
detection (`is_grasped`). Used only for S_end stratification, the dtw fallback phase-match,
and figures.

TODO: boundary windows (grasp_window/place_window) are heuristic; refine per task if needed.
"""
import numpy as np

PHASES = ['APPROACH', 'GRASP', 'TRANSPORT', 'PLACE']
PHASE_ID = {p: i for i, p in enumerate(PHASES)}


def _runs(mask):
    mask = np.asarray(mask, dtype=bool)
    out, i, n = [], 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j + 1 < n and mask[j + 1]:
                j += 1
            out.append((i, j))
            i = j + 1
        else:
            i += 1
    return out


def segment_episode(gripper_open, grasped, grasp_window=8, place_window=12):
    """Return per-timestep phase ids (0..3). Not-grasped = APPROACH (reaching); each grasp
    interval = GRASP onset, then TRANSPORT, then PLACE near release."""
    grasped = np.asarray(grasped, dtype=bool)
    T = len(grasped)
    lab = np.zeros(T, dtype=np.int64)              # default APPROACH
    for a, b in _runs(grasped):
        gr_end = min(b + 1, a + int(grasp_window))
        pl_start = max(a, b + 1 - int(place_window))
        lab[a:b + 1] = PHASE_ID['TRANSPORT']       # fill interval, then overwrite ends
        lab[a:gr_end] = PHASE_ID['GRASP']
        lab[pl_start:b + 1] = PHASE_ID['PLACE']
    return lab


def gripper_open_from_actions(actions):
    """Gripper OPENNESS proxy from the demo action gripper channel (last dim): open if the
    command is >= 0 (robomimic convention: +open / -close)."""
    return (np.asarray(actions)[:, -1] >= 0).astype(bool)


def grasped_from_gripper(actions):
    """Cheap grasped proxy (dataset-only): gripper command closed (last dim < 0). Used for
    pool phases in the dtw fallback to avoid a full-dataset env pass."""
    return (np.asarray(actions)[:, -1] < 0).astype(bool)


def grasped_from_env(obs_source, bodies, episode, states, grasp_qpos_threshold):
    """Per-timestep grasped (any body) via the prelim grasp detection (env resets)."""
    from diffusion_policy.experiments.spatial_attention_prelim_perturb.obs_builder import obs_builder
    T = len(states)
    out = np.zeros(T, dtype=bool)
    for t in range(T):
        flags = obs_builder.detect_grasp_flags(obs_source.wrapper, states, t, bodies,
                                               float(grasp_qpos_threshold))
        out[t] = bool(np.any(flags))
    return out
