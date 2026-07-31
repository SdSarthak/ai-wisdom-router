"""In-process conversation store.

Deliberately not a database: sessions are cheap, disposable and only meaningful
while the browser tab is open. Bounded on both axes — history is trimmed per
session and the least recently used sessions are evicted — so a long-running
server cannot grow without limit.
"""

import threading
from collections import OrderedDict
from typing import Dict

from backend.config import MAX_HISTORY_MESSAGES, MAX_SESSIONS
from backend.graph.state import ConversationState
from backend.scoring.weight_calculator import initialize_weights

# Accumulating fields are appended to rather than replaced on update.
_APPEND_FIELDS = ("messages", "topic_history")

_sessions: "OrderedDict[str, ConversationState]" = OrderedDict()
_lock = threading.Lock()


def _new_state(session_id: str) -> ConversationState:
    weights = initialize_weights()
    return {
        "user_message": "",
        "mode": "adaptive",
        "session_id": session_id,
        "messages": [],
        "topic_history": [],
        "mentor_weights": dict(weights),
        "conversation_summary": "",
        "detected_topics": [],
        "raw_scores": {},
        "selected_mentors": [],
        "response": "",
        "council_responses": {},
        "weight_display": dict(weights),
    }


def get_session(session_id: str) -> ConversationState:
    """Fetch a session, creating it on first use. Marks it as recently used."""
    with _lock:
        session = _sessions.get(session_id)
        if session is None:
            session = _new_state(session_id)
            _sessions[session_id] = session
            _evict_locked()
        else:
            _sessions.move_to_end(session_id)
        return session


def _evict_locked() -> None:
    while len(_sessions) > MAX_SESSIONS:
        _sessions.popitem(last=False)


def update_session(session_id: str, updates: Dict) -> ConversationState:
    """Apply updates to a session; append-only fields are extended and trimmed."""
    with _lock:
        session = _sessions.get(session_id)
        if session is None:
            session = _new_state(session_id)
            _sessions[session_id] = session
            _evict_locked()

        for key, value in updates.items():
            if key in _APPEND_FIELDS:
                combined = list(session.get(key, [])) + list(value)
                # Keep the tail; the newest turns are the ones that matter.
                session[key] = combined[-MAX_HISTORY_MESSAGES:]
            else:
                session[key] = value

        _sessions.move_to_end(session_id)
        return session


def clear_session(session_id: str) -> bool:
    """Forget a session. Returns True if one existed."""
    with _lock:
        return _sessions.pop(session_id, None) is not None


def peek_session_weights(session_id: str) -> Dict[str, float]:
    """The stored distribution for a session, without creating one.

    Read paths must not allocate: `/api/session/{id}/weights` is unauthenticated,
    so creating on read would let anyone mint MAX_SESSIONS entries and evict every
    real conversation from the LRU. An unknown session simply reads as the even
    split a new conversation would start from.
    """
    with _lock:
        session = _sessions.get(session_id)
        if session is None:
            return initialize_weights()
        _sessions.move_to_end(session_id)
        return dict(session.get("weight_display") or initialize_weights())


def session_count() -> int:
    with _lock:
        return len(_sessions)


def clear_all() -> None:
    """Drop every session. Used between tests."""
    with _lock:
        _sessions.clear()
