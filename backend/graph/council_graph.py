from backend.graph.state import ConversationState
from backend.graph.memory_store import get_session, update_session
from backend.scoring.intent_analyzer import detect_topics
from backend.scoring.mentor_scorer import score_all_mentors, select_council_mentors
from backend.llm.prompt_builder import build_council_system_prompt
from backend.llm.ollama_client import generate
from backend.mentors.roster import MENTORS
from backend.config import COUNCIL_TOP_N


def run_council(session_id: str, user_message: str) -> ConversationState:
    state = get_session(session_id)

    # 1. Detect topics
    detected_topics = detect_topics(user_message)

    # 2. Score and select council members
    raw_scores = score_all_mentors(user_message, detected_topics)
    selected = select_council_mentors(raw_scores, top_n=COUNCIL_TOP_N)

    # 3. Generate one response per council member
    council_responses = {}
    for mentor_id in selected:
        if mentor_id not in MENTORS:
            continue
        mentor = MENTORS[mentor_id]
        system_prompt = build_council_system_prompt(mentor)
        history = state["messages"][-6:] if state["messages"] else []
        response = generate(
            system_prompt=system_prompt,
            user_message=user_message,
            history=history,
        )
        council_responses[mentor_id] = response

    # 4. Update session state
    new_messages = [
        {"role": "human", "content": user_message},
        {"role": "assistant", "content": f"[Council] {'; '.join(council_responses.values())}"},
    ]
    update_session(
        session_id,
        {
            "user_message": user_message,
            "detected_topics": detected_topics,
            "raw_scores": raw_scores,
            "selected_mentors": selected,
            "council_responses": council_responses,
            "response": "",
            "messages": new_messages,
            "topic_history": detected_topics,
        },
    )

    return get_session(session_id)
