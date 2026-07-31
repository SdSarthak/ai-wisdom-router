"""End-to-end HTTP behaviour with generation stubbed out."""

import pytest

from backend.mentors.roster import MENTORS


def test_mentors_endpoint_lists_the_roster(client):
    body = client.get("/api/mentors").json()
    assert set(body) == set(MENTORS)
    assert body["paul_graham"]["display_name"] == "Paul Graham"
    assert body["paul_graham"]["color"].startswith("#")


def test_adaptive_chat_returns_an_answer_and_weights(client):
    response = client.post(
        "/api/chat",
        json={"session_id": "s1", "message": "how do I find a startup idea?"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["mode"] == "adaptive"
    assert body["response"]
    assert body["council_responses"] is None
    assert body["detected_topics"]
    assert pytest.approx(sum(body["mentor_weights"].values()), abs=0.02) == 1.0
    assert set(body["mentor_names"]) == set(MENTORS)


def test_council_chat_returns_one_card_per_member(client):
    response = client.post(
        "/api/chat",
        json={"session_id": "s1", "message": "should I quit my job?", "mode": "council"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["mode"] == "council"
    assert body["response"] is None
    members = body["council_responses"]
    assert 1 <= len(members) <= len(MENTORS)
    for member in members:
        assert member["mentor_id"] in MENTORS
        assert member["response"]
        assert member["color"].startswith("#")
    # The sidebar distribution describes the mentors that actually answered.
    assert set(body["mentor_weights"]) == {m["mentor_id"] for m in members}


def test_council_members_are_distinct(client):
    body = client.post(
        "/api/chat",
        json={"session_id": "s1", "message": "how do I build wealth?", "mode": "council"},
    ).json()
    ids = [m["mentor_id"] for m in body["council_responses"]]
    assert len(ids) == len(set(ids))


def test_conversation_history_is_passed_to_the_model(client, stub_llm):
    client.post("/api/chat", json={"session_id": "s1", "message": "first question"})
    client.post("/api/chat", json={"session_id": "s1", "message": "second question"})
    assert stub_llm[-1]["history"], "second turn saw no history"
    assert stub_llm[-1]["history"][0]["content"] == "first question"


def test_sessions_are_isolated(client, stub_llm):
    client.post("/api/chat", json={"session_id": "alice", "message": "alice question"})
    client.post("/api/chat", json={"session_id": "bob", "message": "bob question"})
    assert stub_llm[-1]["history"] == []


def test_weights_persist_across_turns(client):
    client.post("/api/chat", json={"session_id": "s1", "message": "discipline and hard work"})
    stored = client.get("/api/session/s1/weights").json()
    assert pytest.approx(sum(stored["mentor_weights"].values()), abs=0.02) == 1.0


def test_weights_for_an_unknown_session_are_the_even_split(client):
    body = client.get("/api/session/never-seen/weights").json()
    assert len(set(body["mentor_weights"].values())) == 1


def test_reset_clears_the_conversation(client, stub_llm):
    client.post("/api/chat", json={"session_id": "s1", "message": "first"})
    deleted = client.delete("/api/session/s1").json()
    assert deleted["existed"] is True
    assert client.delete("/api/session/s1").json()["existed"] is False

    client.post("/api/chat", json={"session_id": "s1", "message": "after reset"})
    assert stub_llm[-1]["history"] == []


def test_retrieved_quotes_reach_the_system_prompt(client, stub_llm, seeded_store):
    client.post(
        "/api/chat",
        json={"session_id": "s1", "message": "startup ideas that seem like bad ideas"},
    )
    prompt = stub_llm[-1]["system_prompt"]
    assert "actually said" in prompt


# ── Validation and failure handling ──────────────────────────────────

def test_empty_message_is_rejected(client):
    assert client.post("/api/chat", json={"session_id": "s1", "message": "   "}).status_code == 422


def test_missing_message_is_rejected(client):
    assert client.post("/api/chat", json={"session_id": "s1"}).status_code == 422


def test_unknown_mode_is_rejected(client):
    response = client.post(
        "/api/chat", json={"session_id": "s1", "message": "hi", "mode": "oracle"}
    )
    assert response.status_code == 422


def test_blank_session_id_is_rejected(client):
    response = client.post("/api/chat", json={"session_id": "   ", "message": "hi"})
    assert response.status_code == 422


def test_oversized_session_id_is_rejected(client):
    response = client.post(
        "/api/chat", json={"session_id": "x" * 5000, "message": "hi"}
    )
    assert response.status_code == 422


def test_oversized_message_is_rejected(client):
    response = client.post(
        "/api/chat", json={"session_id": "s1", "message": "x" * 20000}
    )
    assert response.status_code == 422


def test_oversized_session_id_in_the_path_is_rejected(client):
    assert client.get(f"/api/session/{'x' * 500}/weights").status_code == 422
    assert client.delete(f"/api/session/{'x' * 500}").status_code == 422


def test_unicode_message_survives_the_round_trip(client, stub_llm):
    message = "如何建立财富? 🚀 ¿Y la disciplina?"
    response = client.post(
        "/api/chat", json={"session_id": "s1", "message": message}
    )
    assert response.status_code == 200
    assert stub_llm[-1]["user_message"] == message


def test_reading_weights_does_not_create_sessions(client):
    """An unauthenticated read that allocated would be an eviction lever."""
    from backend.graph.memory_store import session_count

    before = session_count()
    for i in range(25):
        assert client.get(f"/api/session/probe-{i}/weights").status_code == 200
    assert session_count() == before


def test_backend_outage_returns_503(client, monkeypatch):
    from backend.llm.ollama_client import OllamaError

    async def _down(*args, **kwargs):
        raise OllamaError("Could not reach Ollama at http://localhost:11434")

    monkeypatch.setattr("backend.graph.adaptive_graph.agenerate", _down)
    response = client.post("/api/chat", json={"session_id": "s1", "message": "hi"})
    assert response.status_code == 503
    assert "Ollama" in response.json()["detail"]


def test_council_tolerates_one_member_failing(client, monkeypatch):
    from backend.llm.ollama_client import OllamaError

    calls = {"n": 0}

    async def _flaky(system_prompt, user_message, history=None):
        calls["n"] += 1
        if calls["n"] == 1:
            raise OllamaError("that one timed out")
        return "a real answer"

    monkeypatch.setattr("backend.graph.council_graph.agenerate", _flaky)
    response = client.post(
        "/api/chat",
        json={"session_id": "s1", "message": "advice please", "mode": "council"},
    )
    assert response.status_code == 200
    members = response.json()["council_responses"]
    assert members and all(m["response"] == "a real answer" for m in members)


def test_council_reports_a_total_outage(client, monkeypatch):
    from backend.llm.ollama_client import OllamaError

    async def _down(*args, **kwargs):
        raise OllamaError("everything is down")

    monkeypatch.setattr("backend.graph.council_graph.agenerate", _down)
    response = client.post(
        "/api/chat",
        json={"session_id": "s1", "message": "advice", "mode": "council"},
    )
    assert response.status_code == 503


# ── Health ───────────────────────────────────────────────────────────

def test_health_reports_degraded_without_a_backend(client):
    body = client.get("/api/health").json()
    assert body["status"] == "degraded"
    assert body["ollama"]["reachable"] is False
    assert body["vector_store"]["reachable"] is True


def test_health_counts_active_sessions(client):
    client.post("/api/chat", json={"session_id": "s1", "message": "hello"})
    assert client.get("/api/health").json()["active_sessions"] >= 1


def test_frontend_is_served_at_the_root(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "AI Wisdom Router" in response.text
