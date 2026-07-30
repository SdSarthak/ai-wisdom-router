from typing import Dict, List
from qdrant_client.models import Filter, FieldCondition, MatchValue
from backend.config import (
    QDRANT_COLLECTION,
    QDRANT_SEARCH_LIMIT,
    EMBEDDING_SCORE_WEIGHT,
    DOMAIN_SCORE_WEIGHT,
)
from backend.vector_store.qdrant_client import get_qdrant_client
from backend.vector_store.embedder import embed_text
from backend.mentors.roster import MENTORS, Mentor


def _embedding_score(mentor_id: str, query_vector: list) -> float:
    client = get_qdrant_client()
    results = client.search(
        collection_name=QDRANT_COLLECTION,
        query_vector=query_vector,
        query_filter=Filter(
            must=[FieldCondition(key="mentor_id", match=MatchValue(value=mentor_id))]
        ),
        limit=QDRANT_SEARCH_LIMIT,
    )
    if not results:
        return 0.0
    return sum(r.score for r in results) / len(results)


def _domain_score(mentor: Mentor, detected_topics: List[str]) -> float:
    if not detected_topics:
        return 0.0
    total = sum(mentor.domain_weights.get(t, 0.0) for t in detected_topics)
    return min(total / len(detected_topics), 1.0)


def score_all_mentors(message: str, detected_topics: List[str]) -> Dict[str, float]:
    query_vector = embed_text(message)
    scores = {}
    for mentor_id, mentor in MENTORS.items():
        emb_score = _embedding_score(mentor_id, query_vector)
        dom_score = _domain_score(mentor, detected_topics)
        scores[mentor_id] = (
            EMBEDDING_SCORE_WEIGHT * emb_score + DOMAIN_SCORE_WEIGHT * dom_score
        )
    return scores


def select_council_mentors(raw_scores: Dict[str, float], top_n: int = 4) -> List[str]:
    sorted_mentors = sorted(raw_scores.items(), key=lambda x: x[1], reverse=True)
    if len(sorted_mentors) <= top_n:
        return [m for m, _ in sorted_mentors]

    selected = [sorted_mentors[0][0]]
    for i in range(1, len(sorted_mentors)):
        mentor_id, score = sorted_mentors[i]
        prev_score = sorted_mentors[i - 1][1]
        if i < top_n:
            selected.append(mentor_id)
        elif prev_score - score < 0.15 and len(selected) < top_n + 1:
            selected.append(mentor_id)
        else:
            break

    return selected[:top_n]
