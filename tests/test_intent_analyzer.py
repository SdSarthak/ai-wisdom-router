"""Topic detection."""

import numpy as np
import pytest

from backend.mentors.roster import TOPIC_DOMAINS
from backend.scoring import intent_analyzer as ia


def test_anchors_and_domains_agree():
    """A topic added to one list but not the other silently breaks scoring."""
    ia.validate_anchors()
    assert set(ia.TOPIC_ANCHORS) == set(TOPIC_DOMAINS)


def test_every_anchor_maps_to_a_mentor_domain():
    from backend.mentors.roster import MENTORS

    claimed = set()
    for mentor in MENTORS.values():
        claimed.update(mentor.domain_weights)
    unclaimed = set(ia.TOPIC_ANCHORS) - claimed
    assert not unclaimed, f"no mentor claims these topics: {sorted(unclaimed)}"


def test_precompute_populates_every_anchor():
    assert not ia.anchors_ready()
    ia.precompute_topic_anchors()
    assert ia.anchors_ready()
    assert set(ia._anchor_vectors) == set(ia.TOPIC_ANCHORS)


def test_detect_topics_always_returns_at_least_one():
    ia.precompute_topic_anchors()
    assert len(ia.detect_topics("zzzz qqqq")) >= 1


def test_detect_topics_reuses_a_supplied_vector(monkeypatch):
    """The turn embeds once; detection must not trigger a second call."""
    ia.precompute_topic_anchors()

    def _boom(*args, **kwargs):
        raise AssertionError("detect_topics re-embedded the message")

    monkeypatch.setattr(ia, "embed_text", _boom)
    topics = ia.detect_topics("anything", message_vector=[0.5] * len(next(iter(ia._anchor_vectors.values()))))
    assert topics


def test_detected_topics_are_ordered_by_similarity(monkeypatch):
    dim = 4
    monkeypatch.setattr(
        ia,
        "_anchor_vectors",
        {
            "startup": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            "investing": np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
            "health": np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32),
        },
    )
    monkeypatch.setattr(ia, "TOPIC_SIMILARITY_THRESHOLD", 0.1)
    # Leans investing, then startup; orthogonal to health.
    topics = ia.detect_topics("x", message_vector=[0.3, 0.9, 0.0, 0.0])
    assert topics == ["investing", "startup"]


def test_below_threshold_falls_back_to_the_closest_topic(monkeypatch):
    monkeypatch.setattr(
        ia,
        "_anchor_vectors",
        {
            "startup": np.array([1.0, 0.0], dtype=np.float32),
            "investing": np.array([0.0, 1.0], dtype=np.float32),
        },
    )
    monkeypatch.setattr(ia, "TOPIC_SIMILARITY_THRESHOLD", 0.99)
    assert ia.detect_topics("x", message_vector=[0.9, 0.1]) == ["startup"]


def test_cosine_sim_handles_zero_vectors():
    zero = np.zeros(3, dtype=np.float32)
    other = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    assert ia._cosine_sim(zero, other) == 0.0
    assert ia._cosine_sim(other, other) == pytest.approx(1.0)


def test_validate_anchors_rejects_a_mismatch(monkeypatch):
    monkeypatch.setitem(ia.TOPIC_ANCHORS, "cooking", "making food in a kitchen")
    with pytest.raises(ValueError, match="cooking"):
        ia.validate_anchors()
