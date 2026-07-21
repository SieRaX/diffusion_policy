"""Lightweight shared helpers (no torch / matplotlib import cost)."""


def validate_timeline_episode(timeline_episode, eval_episodes):
    """figures.timeline_episode must be one of eval_episodes (plotting-only)."""
    evs = [int(e) for e in eval_episodes]
    if int(timeline_episode) not in evs:
        raise ValueError(
            f"figures.timeline_episode={timeline_episode} must be one of "
            f"eval_episodes={evs}.")
    return int(timeline_episode)
