from typing import Dict
from backend.graph.state import ConversationState
from backend.scoring.weight_calculator import initialize_weights

# In-memory session store for quick weight/state retrieval
_sessions: Dict[str, ConversationState] = {}


def get_session(session_id: str) -> ConversationState:
    if session_id not in _sessions:
        _sessions[session_id] = {
            "user_message": "",
            "mode": "adaptive",
            "session_id": session_id,
            "messages": [],
            "topic_history": [],
            "mentor_weights": initialize_weights(),
            "conversation_summary": "",
            "detected_topics": [],
            "raw_scores": {},
            "selected_mentors": [],
            "response": "",
            "council_responses": {},
            "weight_display": initialize_weights(),
        }
    return _sessions[session_id]


def update_session(session_id: str, updates: dict):
    session = get_session(session_id)
    for key, value in updates.items():
        if key in ("messages", "topic_history"):
            session[key] = session.get(key, []) + value
        else:
            session[key] = value


def clear_session(session_id: str):
    _sessions.pop(session_id, None)


def get_session_weights(session_id: str) -> Dict[str, float]:
    return get_session(session_id).get("weight_display", initialize_weights())
