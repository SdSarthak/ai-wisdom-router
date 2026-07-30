"""Turn per-turn mentor scores into the running weight distribution.

The distribution is what gives the advisor continuity: raw scores react to the
current message alone, while the weights carry a fraction of the previous turn
forward (WEIGHT_MOMENTUM) so the voice drifts instead of snapping.
"""

from typing import Dict, List, Sequence

from backend.config import (
    MIN_WEIGHT_THRESHOLD,
    TRAJECTORY_BONUS,
    TRAJECTORY_WINDOW,
    WEIGHT_MOMENTUM,
)
from backend.mentors.roster import MENTORS


def _normalize(weights: Dict[str, float]) -> Dict[str, float]:
    total = sum(weights.values())
    if total <= 0:
        return dict(weights)
    return {k: v / total for k, v in weights.items()}


def _round_and_sort(weights: Dict[str, float]) -> Dict[str, float]:
    return {
        k: round(v, 3)
        for k, v in sorted(weights.items(), key=lambda kv: (-kv[1], kv[0]))
    }


def _trajectory_bonus(topic_history: Sequence[str]) -> Dict[str, float]:
    """Reward specialists when the conversation keeps circling one topic.

    Looks at the last two windows of topics; any topic appearing at least
    TRAJECTORY_WINDOW times is treated as the conversation's current direction.
    """
    recent = list(topic_history[-TRAJECTORY_WINDOW * 2 :])
    if len(recent) < TRAJECTORY_WINDOW:
        return {}

    topic_counts: Dict[str, int] = {}
    for topic in recent:
        topic_counts[topic] = topic_counts.get(topic, 0) + 1

    dominant_topics = {t for t, c in topic_counts.items() if c >= TRAJECTORY_WINDOW}
    if not dominant_topics:
        return {}

    bonus: Dict[str, float] = {}
    for mentor_id, mentor in MENTORS.items():
        affinity = sum(mentor.domain_weights.get(t, 0.0) for t in dominant_topics)
        if affinity > 0:
            bonus[mentor_id] = min(affinity * TRAJECTORY_BONUS, 0.15)
    return bonus


def blend_weights(
    old_weights: Dict[str, float],
    new_scores: Dict[str, float],
    topic_history: Sequence[str],
) -> Dict[str, float]:
    """Blend the previous distribution with this turn's scores.

    Returns a normalized distribution, rounded for display, sorted by weight.
    Mentors under MIN_WEIGHT_THRESHOLD are dropped so the prompt is not diluted
    by voices contributing a rounding error — but never all of them.
    """
    if not new_scores:
        return _round_and_sort(_normalize(dict(old_weights)))

    normalized_new = _normalize(new_scores)
    traj_bonus = _trajectory_bonus(topic_history)

    boosted = {
        mentor_id: score + traj_bonus.get(mentor_id, 0.0)
        for mentor_id, score in normalized_new.items()
    }
    boosted = _normalize(boosted)

    blended: Dict[str, float] = {}
    for mentor_id in set(old_weights) | set(boosted):
        old = old_weights.get(mentor_id, 0.0)
        new = boosted.get(mentor_id, 0.0)
        blended[mentor_id] = WEIGHT_MOMENTUM * old + (1 - WEIGHT_MOMENTUM) * new

    survivors = {k: v for k, v in blended.items() if v >= MIN_WEIGHT_THRESHOLD}
    if not survivors:
        # A high threshold relative to the roster size can cut everyone; keeping
        # the strongest mentor is always better than returning no advisor at all.
        survivors = {max(blended, key=lambda k: blended[k]): 1.0}

    return _round_and_sort(_normalize(survivors))


def initialize_weights() -> Dict[str, float]:
    """An even split across the roster — the state of a brand new conversation."""
    n = len(MENTORS)
    if n == 0:
        return {}
    return {mentor_id: round(1.0 / n, 3) for mentor_id in MENTORS}


def scores_to_weights(scores: Dict[str, float], mentor_ids: List[str]) -> Dict[str, float]:
    """Normalize a subset of raw scores into a distribution (used by council mode)."""
    subset = {mid: max(0.0, scores.get(mid, 0.0)) for mid in mentor_ids}
    if sum(subset.values()) <= 0 and subset:
        even = 1.0 / len(subset)
        return _round_and_sort({mid: even for mid in subset})
    return _round_and_sort(_normalize(subset))
