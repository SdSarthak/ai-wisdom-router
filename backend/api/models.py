from pydantic import BaseModel
from typing import Dict, List, Optional


class ChatRequest(BaseModel):
    session_id: str
    message: str
    mode: str = "adaptive"   # "adaptive" | "council"


class CouncilMemberResponse(BaseModel):
    mentor_id: str
    mentor_name: str
    response: str
    color: str


class ChatResponse(BaseModel):
    session_id: str
    mode: str
    response: Optional[str] = None
    mentor_weights: Dict[str, float]
    mentor_names: Dict[str, str]
    council_responses: Optional[List[CouncilMemberResponse]] = None
    detected_topics: List[str] = []


class SessionWeightsResponse(BaseModel):
    session_id: str
    mentor_weights: Dict[str, float]
    mentor_names: Dict[str, str]
