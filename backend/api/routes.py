"""HTTP surface for the wisdom router."""

import asyncio
import logging

from fastapi import APIRouter, HTTPException, Path

from backend.api.models import (
    ChatRequest,
    ChatResponse,
    CouncilMemberResponse,
    HealthResponse,
    MentorInfo,
    SessionClearedResponse,
    SessionWeightsResponse,
)
from backend.graph.adaptive_graph import run_adaptive
from backend.graph.council_graph import run_council
from backend.graph.memory_store import clear_session, peek_session_weights, session_count
from backend.llm.ollama_client import OllamaError
from backend.llm.ollama_client import ping as ollama_ping
from backend.mentors.roster import MENTORS
from backend.scoring.intent_analyzer import anchors_ready
from backend.vector_store.qdrant_client import ping as qdrant_ping

logger = logging.getLogger(__name__)

router = APIRouter()

DEFAULT_COLOR = "#888888"

_MENTOR_NAMES = {mid: m.display_name for mid, m in MENTORS.items()}
_MENTOR_COLORS = {mid: m.color for mid, m in MENTORS.items()}


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    try:
        if request.mode == "council":
            state = await run_council(request.session_id, request.message)
            council = [
                CouncilMemberResponse(
                    mentor_id=mentor_id,
                    mentor_name=_MENTOR_NAMES.get(mentor_id, mentor_id),
                    response=text,
                    color=_MENTOR_COLORS.get(mentor_id, DEFAULT_COLOR),
                )
                for mentor_id, text in state["council_responses"].items()
            ]
            return ChatResponse(
                session_id=request.session_id,
                mode="council",
                mentor_weights=state["weight_display"],
                mentor_names=_MENTOR_NAMES,
                council_responses=council,
                detected_topics=state["detected_topics"],
            )

        state = await run_adaptive(request.session_id, request.message)
        return ChatResponse(
            session_id=request.session_id,
            mode="adaptive",
            response=state["response"],
            mentor_weights=state["weight_display"],
            mentor_names=_MENTOR_NAMES,
            detected_topics=state["detected_topics"],
        )
    except OllamaError as exc:
        # The model backend is down or misconfigured — that is not the caller's
        # fault, and the message tells them exactly what to fix.
        logger.error("Generation failed: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected failure handling chat request")
        raise HTTPException(status_code=500, detail=f"Internal error: {exc}") from exc


# Both session routes take an id straight off the URL. Bounding it here keeps an
# oversized path segment from becoming a dictionary key in the session store.
SessionIdPath = Path(..., min_length=1, max_length=128)


@router.get("/session/{session_id}/weights", response_model=SessionWeightsResponse)
async def get_weights(session_id: str = SessionIdPath) -> SessionWeightsResponse:
    return SessionWeightsResponse(
        session_id=session_id,
        # Peek, never create: this route is unauthenticated, and allocating on
        # read would let anyone flush live conversations out of the LRU.
        mentor_weights=peek_session_weights(session_id),
        mentor_names=_MENTOR_NAMES,
    )


@router.delete("/session/{session_id}", response_model=SessionClearedResponse)
async def reset_session(session_id: str = SessionIdPath) -> SessionClearedResponse:
    return SessionClearedResponse(
        session_id=session_id, existed=clear_session(session_id)
    )


@router.get("/mentors", response_model=dict)
async def list_mentors() -> dict:
    return {
        mentor_id: MentorInfo(
            id=mentor.id,
            display_name=mentor.display_name,
            domains=mentor.domains,
            color=mentor.color,
        ).model_dump()
        for mentor_id, mentor in MENTORS.items()
    }


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    # Both probes are blocking: Ollama over HTTP with a multi-second timeout, and
    # Qdrant against local disk. Run on worker threads so an unreachable-but-not-
    # refusing Ollama stalls this request only, not every other request in flight —
    # the frontend polls this endpoint precisely when the backend is unwell.
    ollama, store = await asyncio.gather(
        asyncio.to_thread(ollama_ping),
        asyncio.to_thread(qdrant_ping),
    )
    healthy = (
        ollama.get("reachable")
        and ollama.get("llm_model_available")
        and ollama.get("embedding_model_available")
        and store.get("reachable")
        and store.get("points", 0) > 0
    )
    return HealthResponse(
        status="ok" if healthy else "degraded",
        ollama=ollama,
        vector_store=store,
        topic_anchors_ready=anchors_ready(),
        active_sessions=session_count(),
    )
