"""Request and response schemas for the HTTP API."""

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator

Mode = Literal["adaptive", "council"]


class ChatRequest(BaseModel):
    session_id: str = Field(..., min_length=1, max_length=128)
    message: str = Field(..., min_length=1, max_length=8000)
    mode: Mode = "adaptive"

    @field_validator("message")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("message must not be empty")
        return stripped


class MentorInfo(BaseModel):
    id: str
    display_name: str
    domains: List[str]
    color: str


class CouncilMemberResponse(BaseModel):
    mentor_id: str
    mentor_name: str
    response: str
    color: str


class ChatResponse(BaseModel):
    session_id: str
    mode: Mode
    response: Optional[str] = None
    mentor_weights: Dict[str, float]
    mentor_names: Dict[str, str]
    council_responses: Optional[List[CouncilMemberResponse]] = None
    detected_topics: List[str] = []


class SessionWeightsResponse(BaseModel):
    session_id: str
    mentor_weights: Dict[str, float]
    mentor_names: Dict[str, str]


class SessionClearedResponse(BaseModel):
    session_id: str
    existed: bool


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    ollama: Dict[str, Any]
    vector_store: Dict[str, Any]
    topic_anchors_ready: bool
    active_sessions: int
