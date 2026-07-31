"""The two turn pipelines, driven directly rather than over HTTP.

These cover what the API tests cannot see: what a turn writes back into session
state, how the council behaves when only some members answer, and whether two
conversations running at once stay separate.
"""

import asyncio

import pytest

from backend.graph import council_graph
from backend.graph.adaptive_graph import run_adaptive
from backend.graph.council_graph import run_council
from backend.graph.memory_store import get_session
from backend.llm.ollama_client import OllamaError
from backend.mentors.roster import MENTORS


# ── Trajectory history ───────────────────────────────────────────────

MULTI_TOPIC = ["discipline", "wealth", "career"]


def _force_multi_topic(monkeypatch, module):
    """Make every turn detect three topics, as real bge-m3 vectors tend to."""
    monkeypatch.setattr(
        f"backend.graph.{module}.detect_topics",
        lambda message, message_vector=None: list(MULTI_TOPIC),
    )


async def test_a_multi_topic_turn_still_contributes_one_history_entry(stub_llm, monkeypatch):
    """The trajectory window counts turns, so a turn must add exactly one entry."""
    _force_multi_topic(monkeypatch, "adaptive_graph")
    for _ in range(4):
        state = await run_adaptive("s1", "anything at all")
    assert state["topic_history"] == ["discipline"] * 4


async def test_council_turns_also_contribute_one_entry(stub_llm, monkeypatch):
    _force_multi_topic(monkeypatch, "council_graph")
    for _ in range(3):
        state = await run_council("s1", "anything at all")
    assert state["topic_history"] == ["discipline"] * 3


async def test_a_sustained_multi_topic_conversation_earns_the_trajectory_bonus(
    stub_llm, monkeypatch
):
    """The regression this guards: with every detected topic appended, three
    topics per turn filled the window with two turns' entries, so no topic ever
    reached TRAJECTORY_WINDOW occurrences and the bonus could never fire."""
    from backend.config import TRAJECTORY_WINDOW
    from backend.scoring.weight_calculator import _trajectory_bonus

    _force_multi_topic(monkeypatch, "adaptive_graph")
    for _ in range(TRAJECTORY_WINDOW):
        state = await run_adaptive("s1", "same subject again")
    assert _trajectory_bonus(state["topic_history"]) != {}


# ── Adaptive turn ────────────────────────────────────────────────────

async def test_adaptive_turn_writes_a_complete_state(stub_llm):
    state = await run_adaptive("s1", "how do I pick what to work on?")
    assert state["mode"] == "adaptive"
    assert state["response"]
    assert state["council_responses"] == {}
    assert state["detected_topics"]
    assert set(state["selected_mentors"]) <= set(MENTORS)
    assert pytest.approx(sum(state["weight_display"].values()), abs=0.02) == 1.0


async def test_adaptive_turn_stores_both_sides_of_the_exchange(stub_llm):
    await run_adaptive("s1", "first question")
    state = await run_adaptive("s1", "second question")
    roles = [m["role"] for m in state["messages"]]
    assert roles == ["human", "assistant", "human", "assistant"]
    assert state["messages"][0]["content"] == "first question"


async def test_a_failed_turn_does_not_corrupt_history(stub_llm, monkeypatch):
    """A turn that dies mid-generation must leave the previous state alone."""
    await run_adaptive("s1", "the good turn")

    async def _down(*args, **kwargs):
        raise OllamaError("model went away")

    monkeypatch.setattr("backend.graph.adaptive_graph.agenerate", _down)
    with pytest.raises(OllamaError):
        await run_adaptive("s1", "the doomed turn")

    messages = get_session("s1")["messages"]
    assert [m["content"] for m in messages] == ["the good turn", "answer to: the good turn"]


# ── Council turn ─────────────────────────────────────────────────────

async def test_council_drops_only_the_members_that_failed(monkeypatch):
    question = "should I push harder or rest?"

    async def _ok(system_prompt, user_message, history=None):
        return "an answer"

    monkeypatch.setattr("backend.graph.council_graph.agenerate", _ok)
    baseline = await run_council("probe", question)
    seats = list(baseline["council_responses"])
    assert len(seats) >= 2, "need a council to lose a member from"
    victim = seats[0]

    async def _selective(system_prompt, user_message, history=None):
        if MENTORS[victim].display_name in system_prompt:
            raise OllamaError("that one timed out")
        return "an answer"

    monkeypatch.setattr("backend.graph.council_graph.agenerate", _selective)
    state = await run_council("fresh", question)

    assert victim not in state["council_responses"]
    assert len(state["council_responses"]) == len(seats) - 1
    # The sidebar must describe exactly the mentors that spoke.
    assert set(state["weight_display"]) == set(state["council_responses"])
    assert set(state["selected_mentors"]) == set(state["council_responses"])


async def test_council_reports_the_underlying_cause_when_all_members_fail(monkeypatch):
    async def _down(*args, **kwargs):
        raise OllamaError("ollama is not running")

    monkeypatch.setattr("backend.graph.council_graph.agenerate", _down)
    with pytest.raises(OllamaError, match="ollama is not running"):
        await run_council("s1", "anything")


async def test_council_history_entry_is_a_bounded_digest(monkeypatch):
    """Storing every full answer would let a few council turns fill the context."""
    long_answer = "word " * 4000

    async def _verbose(system_prompt, user_message, history=None):
        return long_answer

    monkeypatch.setattr("backend.graph.council_graph.agenerate", _verbose)
    state = await run_council("s1", "give me the long version")

    stored = state["messages"][-1]["content"]
    assert len(stored) < len(long_answer)
    per_member = council_graph.DIGEST_CHARS + 100
    assert len(stored) <= per_member * len(state["council_responses"])


async def test_digest_names_each_voice():
    digest = council_graph._digest({"paul_graham": "do things that don't scale"})
    assert digest.startswith("Paul Graham:")


async def test_digest_tolerates_an_unknown_mentor_id():
    assert "ghost" in council_graph._digest({"ghost": "who am I"})


# ── Concurrency ──────────────────────────────────────────────────────

async def test_concurrent_turns_on_different_sessions_stay_separate(stub_llm):
    await asyncio.gather(
        run_adaptive("alice", "alice asks about startups"),
        run_adaptive("bob", "bob asks about health"),
        run_council("carol", "carol asks about money"),
    )
    alice = get_session("alice")["messages"]
    bob = get_session("bob")["messages"]
    assert alice[0]["content"] == "alice asks about startups"
    assert bob[0]["content"] == "bob asks about health"
    assert len(get_session("carol")["messages"]) == 2


async def test_concurrent_cold_starts_agree_on_the_anchor_set(stub_llm):
    """Several first-ever turns racing to build the anchors must not leave a
    partially populated map behind."""
    from backend.scoring import intent_analyzer as ia

    assert not ia.anchors_ready()
    await asyncio.gather(*(run_adaptive(f"s{i}", "a question about wealth") for i in range(6)))
    assert set(ia._anchor_vectors) == set(ia.TOPIC_ANCHORS)


async def test_many_turns_do_not_grow_history_without_bound(stub_llm, monkeypatch):
    from backend.config import MAX_HISTORY_MESSAGES

    for i in range(MAX_HISTORY_MESSAGES + 15):
        await run_adaptive("s1", f"question {i}")
    assert len(get_session("s1")["messages"]) == MAX_HISTORY_MESSAGES
    assert len(get_session("s1")["topic_history"]) <= MAX_HISTORY_MESSAGES
