from fastapi import APIRouter
from backend.api.models import ChatRequest, ChatResponse, CouncilMemberResponse, SessionWeightsResponse
from backend.graph.adaptive_graph import run_adaptive
from backend.graph.council_graph import run_council
from backend.graph.memory_store import get_session_weights, clear_session
from backend.mentors.roster import MENTORS

router = APIRouter()

_MENTOR_NAMES = {mid: m.display_name for mid, m in MENTORS.items()}
_MENTOR_COLORS = {mid: m.color for mid, m in MENTORS.items()}


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if request.mode == "council":
        state = run_council(request.session_id, request.message)
        council = [
            CouncilMemberResponse(
                mentor_id=mid,
                mentor_name=_MENTOR_NAMES.get(mid, mid),
                response=resp,
                color=_MENTOR_COLORS.get(mid, "#888888"),
            )
            for mid, resp in state["council_responses"].items()
        ]
        return ChatResponse(
            session_id=request.session_id,
            mode="council",
            mentor_weights=state["mentor_weights"],
            mentor_names=_MENTOR_NAMES,
            council_responses=council,
            detected_topics=state["detected_topics"],
        )
    else:
        state = run_adaptive(request.session_id, request.message)
        return ChatResponse(
            session_id=request.session_id,
            mode="adaptive",
            response=state["response"],
            mentor_weights=state["weight_display"],
            mentor_names=_MENTOR_NAMES,
            detected_topics=state["detected_topics"],
        )


@router.get("/session/{session_id}/weights", response_model=SessionWeightsResponse)
async def get_weights(session_id: str):
    weights = get_session_weights(session_id)
    return SessionWeightsResponse(
        session_id=session_id,
        mentor_weights=weights,
        mentor_names=_MENTOR_NAMES,
    )


@router.delete("/session/{session_id}")
async def reset_session(session_id: str):
    clear_session(session_id)
    return {"status": "cleared", "session_id": session_id}


@router.get("/mentors")
async def list_mentors():
    return {
        mid: {
            "display_name": m.display_name,
            "domains": m.domains,
            "color": m.color,
        }
        for mid, m in MENTORS.items()
    }
