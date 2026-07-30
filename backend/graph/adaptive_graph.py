from typing import Dict
from backend.graph.state import ConversationState
from backend.graph.memory_store import get_session, update_session
from backend.scoring.intent_analyzer import detect_topics
from backend.scoring.mentor_scorer import score_all_mentors
from backend.scoring.weight_calculator import blend_weights
from backend.llm.prompt_builder import build_adaptive_system_prompt
from backend.llm.ollama_client import generate


def run_adaptive(session_id: str, user_message: str) -> ConversationState:
    state = get_session(session_id)

    # 1. Detect topics
    detected_topics = detect_topics(user_message)

    # 2. Score mentors
    raw_scores = score_all_mentors(user_message, detected_topics)

    # 3. Blend weights using trajectory
    new_weights = blend_weights(
        old_weights=state["mentor_weights"],
        new_scores=raw_scores,
        topic_history=state["topic_history"],
    )

    # 4. Build blended system prompt
    system_prompt = build_adaptive_system_prompt(new_weights)

    # 5. Generate response
    history = state["messages"][-10:] if state["messages"] else []
    response = generate(
        system_prompt=system_prompt,
        user_message=user_message,
        history=history,
    )

    # 6. Update session state
    new_messages = [
        {"role": "human", "content": user_message},
        {"role": "assistant", "content": response},
    ]
    update_session(
        session_id,
        {
            "user_message": user_message,
            "detected_topics": detected_topics,
            "raw_scores": raw_scores,
            "mentor_weights": new_weights,
            "weight_display": new_weights,
            "response": response,
            "council_responses": {},
            "messages": new_messages,
            "topic_history": detected_topics,
        },
    )

    return get_session(session_id)
