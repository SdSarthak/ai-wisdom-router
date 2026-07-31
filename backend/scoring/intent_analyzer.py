"""Map a message onto the topic space the mentor roster is indexed by.

Each topic has a short anchor description. Those are embedded once at startup;
detection is then a cosine comparison against the message vector, which the caller
supplies so a turn only pays for one embedding call.
"""

import logging
from typing import Dict, List, Optional, Sequence

import numpy as np

from backend.config import TOPIC_SIMILARITY_THRESHOLD
from backend.mentors.roster import TOPIC_DOMAINS
from backend.vector_store.embedder import aembed_texts, embed_text, embed_texts

logger = logging.getLogger(__name__)

TOPIC_ANCHORS: Dict[str, str] = {
    "startup": "building a startup founding a company product market fit early stage venture",
    "investing": "investing money stock market portfolio allocation returns financial assets",
    "discipline": "hard work mental toughness pushing limits consistency self control habits",
    "career": "career growth job professional development skills work promotion",
    "wealth": "building wealth financial independence passive income net worth money",
    "learning": "learning reading acquiring knowledge mental models education studying",
    "leadership": "leading a team management organizational culture decision making vision",
    "relationships": "relationships friendships family social dynamics communication love trust",
    "philosophy": "meaning of life principles stoicism values existence purpose happiness",
    "health": "fitness diet exercise mental health sleep recovery physical performance",
}

_anchor_vectors: Dict[str, np.ndarray] = {}


def anchors_ready() -> bool:
    return bool(_anchor_vectors)


def _store(embeddings: Sequence[Sequence[float]]) -> None:
    """Publish a complete anchor set, or none at all.

    Two things matter here. The set is built first and swapped in with a single
    rebind, so a request running concurrently with a rebuild reads either the old
    map or the new one but never a half-populated one — `zip` against a partial
    map would silently make some topics undetectable. And a short embedding batch
    is rejected rather than zipped away, for the same reason.
    """
    global _anchor_vectors

    expected = len(TOPIC_ANCHORS)
    vectors = list(embeddings)
    if len(vectors) != expected:
        raise ValueError(
            f"Expected {expected} topic anchor embeddings, got {len(vectors)}. "
            "Topic detection would silently ignore the missing topics."
        )

    built = {
        topic: np.asarray(vec, dtype=np.float32)
        for topic, vec in zip(TOPIC_ANCHORS.keys(), vectors)
    }
    _anchor_vectors = built
    logger.info("Precomputed %d topic anchor vectors.", len(built))


def precompute_topic_anchors() -> None:
    """Embed the topic anchors. Blocking; called from the seeding path."""
    _store(embed_texts(list(TOPIC_ANCHORS.values())))


async def aprecompute_topic_anchors() -> None:
    """Embed the topic anchors without blocking the event loop."""
    _store(await aembed_texts(list(TOPIC_ANCHORS.values())))


def clear_anchors() -> None:
    """Drop cached anchors, e.g. after switching embedding model."""
    global _anchor_vectors
    _anchor_vectors = {}


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def topic_scores(message_vector: Sequence[float]) -> Dict[str, float]:
    """Cosine similarity of a message against every topic anchor."""
    if not _anchor_vectors:
        return {}
    msg_vec = np.asarray(message_vector, dtype=np.float32)
    if msg_vec.size == 0:
        return {}
    return {
        topic: _cosine_sim(msg_vec, anchor)
        for topic, anchor in _anchor_vectors.items()
    }


def detect_topics(
    message: str,
    message_vector: Optional[Sequence[float]] = None,
) -> List[str]:
    """Return the topics a message is about, never fewer than one.

    Pass `message_vector` to reuse an embedding that has already been computed.
    """
    if not _anchor_vectors:
        precompute_topic_anchors()

    if message_vector is None:
        message_vector = embed_text(message)

    scores = topic_scores(message_vector)
    if not scores:
        return []

    detected = sorted(
        (t for t, s in scores.items() if s >= TOPIC_SIMILARITY_THRESHOLD),
        key=lambda t: scores[t],
        reverse=True,
    )

    # A message always lands somewhere; fall back to its single closest topic.
    if not detected:
        detected = [max(scores, key=lambda t: scores[t])]

    return detected


def validate_anchors() -> None:
    """Guard against a topic being added to one of the two lists but not the other."""
    anchors = set(TOPIC_ANCHORS)
    domains = set(TOPIC_DOMAINS)
    if anchors != domains:
        raise ValueError(
            "TOPIC_ANCHORS and TOPIC_DOMAINS disagree — "
            f"only in anchors: {sorted(anchors - domains)}, "
            f"only in domains: {sorted(domains - anchors)}"
        )
