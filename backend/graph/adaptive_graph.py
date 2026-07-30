"""Adaptive mode: one synthesized voice whose composition shifts turn by turn.

Pipeline for a turn:
  embed message -> detect topics -> score mentors (+ retrieve their quotes)
  -> blend into the running weight distribution -> build a weighted system prompt
  -> generate -> persist.
"""

import logging

from backend.config import COUNCIL_TOP_N
from backend.graph.memory_store import get_session, update_session
from backend.graph.state import ConversationState
from backend.llm.ollama_client import agenerate
from backend.llm.prompt_builder import build_adaptive_system_prompt
from backend.scoring.intent_analyzer import aprecompute_topic_anchors, anchors_ready, detect_topics
from backend.scoring.mentor_scorer import score_all_mentors
from backend.scoring.weight_calculator import blend_weights
from backend.vector_store.embedder import aembed_text

logger = logging.getLogger(__name__)

# Turns of prior conversation shown to the model.
HISTORY_TURNS = 10


async def run_adaptive(session_id: str, user_message: str) -> ConversationState:
    state = get_session(session_id)

    # One embedding per turn, shared by topic detection and mentor scoring.
    query_vector = await aembed_text(user_message)

    if not anchors_ready():
        await aprecompute_topic_anchors()
    detected_topics = detect_topics(user_message, message_vector=query_vector)

    scoring = score_all_mentors(query_vector, detected_topics)

    new_weights = blend_weights(
        old_weights=state["mentor_weights"],
        new_scores=scoring.scores,
        topic_history=state["topic_history"],
    )

    system_prompt = build_adaptive_system_prompt(new_weights, evidence=scoring.evidence)
    history = list(state.get("messages", []))[-HISTORY_TURNS:]

    response = await agenerate(
        system_prompt=system_prompt,
        user_message=user_message,
        history=history,
    )

    return update_session(
        session_id,
        {
            "user_message": user_message,
            "mode": "adaptive",
            "detected_topics": detected_topics,
            "raw_scores": scoring.scores,
            "mentor_weights": new_weights,
            "weight_display": new_weights,
            "response": response,
            "council_responses": {},
            "selected_mentors": [
                mid for mid, _ in sorted(
                    new_weights.items(), key=lambda kv: -kv[1]
                )[:COUNCIL_TOP_N]
            ],
            "messages": [
                {"role": "human", "content": user_message},
                {"role": "assistant", "content": response},
            ],
            "topic_history": detected_topics,
        },
    )
