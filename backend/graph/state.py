import operator
from typing import TypedDict, List, Dict, Annotated


class ConversationState(TypedDict):
    # Input fields
    user_message: str
    mode: str          # "adaptive" | "council"
    session_id: str

    # Accumulated history
    messages: Annotated[List[Dict], operator.add]
    topic_history: Annotated[List[str], operator.add]

    # Adaptive mode — evolving weights
    mentor_weights: Dict[str, float]
    conversation_summary: str

    # Scoring artifacts (reset each turn)
    detected_topics: List[str]
    raw_scores: Dict[str, float]
    selected_mentors: List[str]

    # Output
    response: str                       # adaptive: single blended response
    council_responses: Dict[str, str]   # council: {mentor_id: response_text}
    weight_display: Dict[str, float]    # what the frontend shows
