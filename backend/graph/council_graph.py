"""Council mode: the top-scoring mentors each answer in their own voice.

The per-mentor calls are independent, so they are issued concurrently. Ollama
serializes them internally when only one model is loaded, but the client no longer
pays the round-trip latency N times, and a mentor whose call fails is dropped from
the council instead of failing the whole turn.
"""

import asyncio
import logging
from typing import Dict, List

from backend.config import COUNCIL_TOP_N
from backend.graph.memory_store import get_session, update_session
from backend.graph.state import ConversationState
from backend.llm.ollama_client import OllamaError, agenerate
from backend.llm.prompt_builder import build_council_system_prompt
from backend.mentors.roster import MENTORS
from backend.scoring.intent_analyzer import aprecompute_topic_anchors, anchors_ready, detect_topics
from backend.scoring.mentor_scorer import score_all_mentors, select_council_mentors
from backend.scoring.weight_calculator import scores_to_weights
from backend.vector_store.embedder import aembed_text

logger = logging.getLogger(__name__)

# Council answers are short by construction; keep less history than adaptive mode
# so five parallel prompts stay small.
HISTORY_TURNS = 6
# Characters of each member's answer retained in conversation history.
DIGEST_CHARS = 300


def _digest(council_responses: Dict[str, str]) -> str:
    """Compact the council's answers into one history entry.

    Storing every full answer would let a few council turns crowd the context
    window on their own, so history keeps the gist of each voice instead.
    """
    parts = []
    for mentor_id, text in council_responses.items():
        name = MENTORS[mentor_id].display_name if mentor_id in MENTORS else mentor_id
        snippet = text.strip()
        if len(snippet) > DIGEST_CHARS:
            snippet = snippet[:DIGEST_CHARS].rstrip() + "…"
        parts.append(f"{name}: {snippet}")
    return "\n\n".join(parts)


async def run_council(session_id: str, user_message: str) -> ConversationState:
    state = get_session(session_id)

    query_vector = await aembed_text(user_message)

    if not anchors_ready():
        await aprecompute_topic_anchors()
    detected_topics = detect_topics(user_message, message_vector=query_vector)

    # Blocking vector search — see the same call in adaptive_graph.
    scoring = await asyncio.to_thread(score_all_mentors, query_vector, detected_topics)
    selected: List[str] = [
        mid
        for mid in select_council_mentors(scoring.scores, top_n=COUNCIL_TOP_N)
        if mid in MENTORS
    ]

    history = list(state.get("messages", []))[-HISTORY_TURNS:]

    async def _ask(mentor_id: str) -> str:
        prompt = build_council_system_prompt(
            MENTORS[mentor_id], evidence=scoring.evidence.get(mentor_id)
        )
        return await agenerate(
            system_prompt=prompt, user_message=user_message, history=history
        )

    results = await asyncio.gather(
        *(_ask(mid) for mid in selected), return_exceptions=True
    )

    council_responses: Dict[str, str] = {}
    for mentor_id, result in zip(selected, results):
        if isinstance(result, Exception):
            logger.warning("Council member %s failed: %s", mentor_id, result)
            continue
        council_responses[mentor_id] = result

    if not council_responses:
        # Every member failed — surface the underlying cause rather than an
        # empty council the frontend would render as a blank card.
        first_error = next((r for r in results if isinstance(r, Exception)), None)
        raise OllamaError(
            f"No council member could answer: {first_error}"
            if first_error
            else "No council member could answer."
        )

    answered = list(council_responses)

    return update_session(
        session_id,
        {
            "user_message": user_message,
            "mode": "council",
            "detected_topics": detected_topics,
            "raw_scores": scoring.scores,
            "selected_mentors": answered,
            "council_responses": council_responses,
            "weight_display": scores_to_weights(scoring.scores, answered),
            "response": "",
            "messages": [
                {"role": "human", "content": user_message},
                {"role": "assistant", "content": _digest(council_responses)},
            ],
            # One entry per turn; see the same note in adaptive_graph.
            "topic_history": detected_topics[:1],
        },
    )
