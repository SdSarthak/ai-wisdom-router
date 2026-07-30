"""Score every mentor against the current message.

A mentor's score combines two signals:

  embedding — how close the message sits to that mentor's own corpus of quotes,
              measured as the mean cosine score of their best QDRANT_SEARCH_LIMIT hits
  domain    — how strongly they claim the topics detected in the message

The retrieved quotes are kept alongside the score and handed to the prompt builder,
so the answer is grounded in what the mentor actually said rather than in the
model's impression of them.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from qdrant_client.models import FieldCondition, Filter, MatchValue

from backend.config import (
    COUNCIL_MIN_SCORE_GAP,
    DOMAIN_SCORE_WEIGHT,
    EMBEDDING_SCORE_WEIGHT,
    QDRANT_COLLECTION,
    QDRANT_SEARCH_LIMIT,
)
from backend.mentors.roster import MENTORS, Mentor
from backend.vector_store.qdrant_client import get_qdrant_client

logger = logging.getLogger(__name__)


@dataclass
class Evidence:
    """A quote retrieved from a mentor's corpus."""

    text: str
    source: str
    score: float


@dataclass
class ScoringResult:
    scores: Dict[str, float] = field(default_factory=dict)
    evidence: Dict[str, List[Evidence]] = field(default_factory=dict)
    degraded: bool = False  # True when retrieval failed and only domain scores are real


def _search(client, query_vector: Sequence[float], mentor_id: str):
    """Retrieve a mentor's closest quotes, tolerating both qdrant-client APIs."""
    query_filter = Filter(
        must=[FieldCondition(key="mentor_id", match=MatchValue(value=mentor_id))]
    )
    # query_points supersedes search in qdrant-client 1.10+; search still works but
    # is deprecated. Prefer the new call when the installed version provides it.
    if hasattr(client, "query_points"):
        return client.query_points(
            collection_name=QDRANT_COLLECTION,
            query=list(query_vector),
            query_filter=query_filter,
            limit=QDRANT_SEARCH_LIMIT,
            with_payload=True,
        ).points
    return client.search(
        collection_name=QDRANT_COLLECTION,
        query_vector=list(query_vector),
        query_filter=query_filter,
        limit=QDRANT_SEARCH_LIMIT,
        with_payload=True,
    )


def _retrieve(client, query_vector: Sequence[float], mentor_id: str) -> List[Evidence]:
    hits = _search(client, query_vector, mentor_id)
    return [
        Evidence(
            text=(hit.payload or {}).get("text", ""),
            source=(hit.payload or {}).get("source", ""),
            # Cosine similarity spans [-1, 1]; anything below 0 means "unrelated",
            # not "negatively related", so clamp before averaging.
            score=max(0.0, min(1.0, float(hit.score))),
        )
        for hit in hits
        if (hit.payload or {}).get("text")
    ]


def _embedding_score(evidence: List[Evidence]) -> float:
    if not evidence:
        return 0.0
    return sum(e.score for e in evidence) / len(evidence)


def _domain_score(mentor: Mentor, detected_topics: Sequence[str]) -> float:
    if not detected_topics:
        return 0.0
    total = sum(mentor.domain_weights.get(t, 0.0) for t in detected_topics)
    return min(total / len(detected_topics), 1.0)


def score_all_mentors(
    query_vector: Optional[Sequence[float]],
    detected_topics: Sequence[str],
) -> ScoringResult:
    """Score the roster against a pre-embedded message.

    The caller passes the query vector so a turn only embeds the message once —
    topic detection and mentor scoring share it. If retrieval is unavailable the
    result falls back to domain scores alone and is flagged as degraded.
    """
    result = ScoringResult()

    client = None
    if query_vector is not None:
        try:
            client = get_qdrant_client()
            if not client.collection_exists(QDRANT_COLLECTION):
                logger.warning(
                    "Collection %s does not exist — scoring on domain weights alone. "
                    "Run: python -m backend.vector_store.seeder",
                    QDRANT_COLLECTION,
                )
                client = None
                result.degraded = True
        except Exception as exc:
            logger.warning(
                "Vector store unavailable (%s) — scoring on domain weights alone.", exc
            )
            client = None
            result.degraded = True
    else:
        result.degraded = True

    for mentor_id, mentor in MENTORS.items():
        evidence: List[Evidence] = []
        if client is not None:
            try:
                evidence = _retrieve(client, query_vector, mentor_id)
            except Exception as exc:
                logger.warning("Retrieval failed for %s: %s", mentor_id, exc)
                result.degraded = True

        result.evidence[mentor_id] = evidence
        dom = _domain_score(mentor, detected_topics)

        if not evidence and result.degraded:
            # Without retrieval the embedding term would drag every mentor toward
            # zero uniformly; use the domain signal on its own instead.
            result.scores[mentor_id] = dom
        else:
            result.scores[mentor_id] = (
                EMBEDDING_SCORE_WEIGHT * _embedding_score(evidence)
                + DOMAIN_SCORE_WEIGHT * dom
            )

    return result


def select_council_mentors(
    raw_scores: Dict[str, float],
    top_n: int,
    min_score_gap: float = COUNCIL_MIN_SCORE_GAP,
) -> List[str]:
    """Pick the council: the top N, plus a runner-up in a near-tie with the cut line.

    A mentor sitting within `min_score_gap` of the last qualifying score has not
    meaningfully lost, so they get a seat rather than being cut by an arbitrary
    boundary. At most one extra seat is granted.
    """
    if top_n < 1:
        return []

    # Sort by score desc, then id, so equal scores order deterministically.
    ranked = sorted(raw_scores.items(), key=lambda kv: (-kv[1], kv[0]))
    if len(ranked) <= top_n:
        return [mentor_id for mentor_id, _ in ranked]

    selected = [mentor_id for mentor_id, _ in ranked[:top_n]]

    cutoff_score = ranked[top_n - 1][1]
    runner_up_id, runner_up_score = ranked[top_n]
    if cutoff_score - runner_up_score < min_score_gap:
        selected.append(runner_up_id)

    return selected
