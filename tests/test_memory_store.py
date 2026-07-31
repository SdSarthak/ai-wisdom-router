"""Session state: accumulation, trimming and eviction."""

import pytest

from backend.graph import memory_store as store
from backend.mentors.roster import MENTORS


def test_new_session_starts_on_an_even_split():
    state = store.get_session("s1")
    assert state["session_id"] == "s1"
    assert set(state["mentor_weights"]) == set(MENTORS)
    assert pytest.approx(sum(state["mentor_weights"].values()), abs=0.01) == 1.0


def test_append_fields_accumulate_across_turns():
    store.update_session("s1", {"messages": [{"role": "human", "content": "one"}]})
    store.update_session("s1", {"messages": [{"role": "human", "content": "two"}]})
    contents = [m["content"] for m in store.get_session("s1")["messages"]]
    assert contents == ["one", "two"]


def test_scalar_fields_are_replaced_not_appended():
    store.update_session("s1", {"response": "first"})
    store.update_session("s1", {"response": "second"})
    assert store.get_session("s1")["response"] == "second"


def test_history_is_trimmed_to_the_newest_turns(monkeypatch):
    monkeypatch.setattr(store, "MAX_HISTORY_MESSAGES", 4)
    for i in range(10):
        store.update_session("s1", {"messages": [{"role": "human", "content": str(i)}]})
    contents = [m["content"] for m in store.get_session("s1")["messages"]]
    assert contents == ["6", "7", "8", "9"]


def test_topic_history_is_trimmed_too(monkeypatch):
    monkeypatch.setattr(store, "MAX_HISTORY_MESSAGES", 3)
    for topic in ["a", "b", "c", "d", "e"]:
        store.update_session("s1", {"topic_history": [topic]})
    assert store.get_session("s1")["topic_history"] == ["c", "d", "e"]


def test_least_recently_used_sessions_are_evicted(monkeypatch):
    monkeypatch.setattr(store, "MAX_SESSIONS", 3)
    for i in range(3):
        store.get_session(f"s{i}")
    # Touching s0 makes s1 the least recently used.
    store.get_session("s0")
    store.get_session("s3")
    assert store.session_count() == 3
    assert store.clear_session("s1") is False
    assert store.clear_session("s0") is True


def test_updating_a_session_marks_it_recently_used(monkeypatch):
    monkeypatch.setattr(store, "MAX_SESSIONS", 2)
    store.get_session("a")
    store.get_session("b")
    store.update_session("a", {"response": "keep me"})
    store.get_session("c")  # evicts b, not a
    assert store.clear_session("b") is False
    assert store.clear_session("a") is True


def test_clear_session_reports_whether_it_existed():
    store.get_session("s1")
    assert store.clear_session("s1") is True
    assert store.clear_session("s1") is False


def test_cleared_session_returns_to_defaults():
    store.update_session("s1", {"weight_display": {"paul_graham": 1.0}})
    store.clear_session("s1")
    assert set(store.peek_session_weights("s1")) == set(MENTORS)


# ── Reads must not allocate ──────────────────────────────────────────

def test_peeking_weights_does_not_create_a_session():
    assert store.session_count() == 0
    assert set(store.peek_session_weights("never-seen")) == set(MENTORS)
    assert store.session_count() == 0


def test_peeking_returns_the_stored_distribution():
    store.update_session("s1", {"weight_display": {"paul_graham": 1.0}})
    assert store.peek_session_weights("s1") == {"paul_graham": 1.0}


def test_peeking_cannot_evict_a_live_session(monkeypatch):
    """The weights route is unauthenticated; if reading allocated, anyone could
    push every real conversation out of the LRU with MAX_SESSIONS requests."""
    monkeypatch.setattr(store, "MAX_SESSIONS", 3)
    store.update_session("real", {"response": "important"})
    for i in range(50):
        store.peek_session_weights(f"attacker-{i}")
    assert store.session_count() == 1
    assert store.get_session("real")["response"] == "important"


def test_peeking_marks_a_known_session_recently_used(monkeypatch):
    monkeypatch.setattr(store, "MAX_SESSIONS", 2)
    store.get_session("a")
    store.get_session("b")
    store.peek_session_weights("a")
    store.get_session("c")  # evicts b, not a
    assert store.clear_session("b") is False
    assert store.clear_session("a") is True


def test_update_creates_a_missing_session():
    store.update_session("brand-new", {"response": "hi"})
    assert store.get_session("brand-new")["response"] == "hi"
