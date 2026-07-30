from typing import Dict, List
from backend.config import (
    WEIGHT_MOMENTUM,
    MIN_WEIGHT_THRESHOLD,
    TRAJECTORY_WINDOW,
    TRAJECTORY_BONUS,
)
from backend.mentors.roster import MENTORS


def _normalize(weights: Dict[str, float]) -> Dict[str, float]:
    total = sum(weights.values())
    if total == 0:
        return weights
    return {k: v / total for k, v in weights.items()}


def _trajectory_bonus(topic_history: List[str]) -> Dict[str, float]:
    recent = topic_history[-TRAJECTORY_WINDOW * 2 :]
    if len(recent) < TRAJECTORY_WINDOW:
        return {}

    topic_counts: Dict[str, int] = {}
    for t in recent:
        topic_counts[t] = topic_counts.get(t, 0) + 1

    dominant_topics = {t for t, c in topic_counts.items() if c >= TRAJECTORY_WINDOW}
    if not dominant_topics:
        return {}

    bonus: Dict[str, float] = {}
    for mentor_id, mentor in MENTORS.items():
        mentor_bonus = sum(
            mentor.domain_weights.get(t, 0.0) for t in dominant_topics
        )
        if mentor_bonus > 0:
            bonus[mentor_id] = min(mentor_bonus * TRAJECTORY_BONUS, 0.15)
    return bonus


def blend_weights(
    old_weights: Dict[str, float],
    new_scores: Dict[str, float],
    topic_history: List[str],
) -> Dict[str, float]:
    normalized_new = _normalize(new_scores)
    traj_bonus = _trajectory_bonus(topic_history)

    boosted_new: Dict[str, float] = {}
    for mentor_id in normalized_new:
        boosted_new[mentor_id] = normalized_new[mentor_id] + traj_bonus.get(mentor_id, 0.0)

    boosted_new = _normalize(boosted_new)

    blended: Dict[str, float] = {}
    all_mentors = set(old_weights) | set(boosted_new)
    for mentor_id in all_mentors:
        old = old_weights.get(mentor_id, 0.0)
        new = boosted_new.get(mentor_id, 0.0)
        blended[mentor_id] = WEIGHT_MOMENTUM * old + (1 - WEIGHT_MOMENTUM) * new

    blended = {k: v for k, v in blended.items() if v >= MIN_WEIGHT_THRESHOLD}

    blended = _normalize(blended)
    return {k: round(v, 3) for k, v in sorted(blended.items(), key=lambda x: x[1], reverse=True)}


def initialize_weights() -> Dict[str, float]:
    n = len(MENTORS)
    return {mentor_id: round(1.0 / n, 3) for mentor_id in MENTORS}
