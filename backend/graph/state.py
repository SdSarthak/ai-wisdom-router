"""Shape of a conversation's server-side state.

`messages` and `topic_history` accumulate across turns — `memory_store.update_session`
appends to them and trims to MAX_HISTORY_MESSAGES. Every other field describes the
most recent turn only and is overwritten each time.
"""

from typing import Dict, List, TypedDict


class ConversationState(TypedDict):
    # Input
    user_message: str
    mode: str          # "adaptive" | "council"
    session_id: str

    # Accumulated across turns
    messages: List[Dict[str, str]]
    topic_history: List[str]

    # Adaptive mode — the evolving distribution that carries between turns
    mentor_weights: Dict[str, float]
    conversation_summary: str

    # Scoring artifacts for the current turn
    detected_topics: List[str]
    raw_scores: Dict[str, float]
    selected_mentors: List[str]

    # Output
    response: str                       # adaptive: the single blended answer
    council_responses: Dict[str, str]   # council: {mentor_id: answer}
    weight_display: Dict[str, float]    # what the sidebar renders
