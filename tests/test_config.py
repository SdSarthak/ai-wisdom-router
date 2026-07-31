"""Configuration parsing and validation."""

import pytest

from backend import config
from backend.mentors.roster import MENTORS, TOPIC_DOMAINS


def test_default_configuration_is_valid():
    config.validate()


def test_env_overrides_are_applied():
    # conftest sets these before the backend is imported.
    assert config.QDRANT_MODE == "memory"
    assert config.QDRANT_COLLECTION == "test_mentor_knowledge"


def test_int_parser_rejects_nonsense(monkeypatch):
    monkeypatch.setenv("SOME_INT", "not-a-number")
    with pytest.raises(ValueError, match="SOME_INT"):
        config._int("SOME_INT", 1)


def test_float_parser_rejects_nonsense(monkeypatch):
    monkeypatch.setenv("SOME_FLOAT", "abc")
    with pytest.raises(ValueError, match="SOME_FLOAT"):
        config._float("SOME_FLOAT", 1.0)


def test_blank_env_var_falls_back_to_the_default(monkeypatch):
    monkeypatch.setenv("SOME_STR", "   ")
    assert config._str("SOME_STR", "fallback") == "fallback"


@pytest.mark.parametrize("raw,expected", [("true", True), ("1", True), ("on", True), ("no", False), ("false", False)])
def test_bool_parser(monkeypatch, raw, expected):
    monkeypatch.setenv("SOME_BOOL", raw)
    assert config._bool("SOME_BOOL", not expected) is expected


def test_validate_rejects_an_unknown_qdrant_mode(monkeypatch):
    monkeypatch.setattr(config, "QDRANT_MODE", "postgres")
    with pytest.raises(ValueError, match="QDRANT_MODE"):
        config.validate()


def test_validate_rejects_frozen_momentum(monkeypatch):
    monkeypatch.setattr(config, "WEIGHT_MOMENTUM", 1.0)
    with pytest.raises(ValueError, match="WEIGHT_MOMENTUM"):
        config.validate()


def test_validate_rejects_an_empty_council(monkeypatch):
    monkeypatch.setattr(config, "COUNCIL_TOP_N", 0)
    with pytest.raises(ValueError, match="COUNCIL_TOP_N"):
        config.validate()


def test_validate_rejects_zero_scoring_weights(monkeypatch):
    monkeypatch.setattr(config, "EMBEDDING_SCORE_WEIGHT", 0.0)
    monkeypatch.setattr(config, "DOMAIN_SCORE_WEIGHT", 0.0)
    with pytest.raises(ValueError):
        config.validate()


@pytest.mark.parametrize(
    "name,value",
    [
        # Zero or negative bounds silently invert a `list[-n:]` tail slice into
        # "keep everything", so they must never reach the session store.
        ("MAX_HISTORY_MESSAGES", 0),
        ("MAX_HISTORY_MESSAGES", -5),
        ("MAX_HISTORY_MESSAGES", 1),
        ("MAX_SESSIONS", 0),
        ("TRAJECTORY_WINDOW", 0),
        ("TRAJECTORY_WINDOW", -1),
        # Values that would fail deep inside a driver with an opaque message.
        ("QDRANT_SEARCH_LIMIT", 0),
        ("LLM_TIMEOUT_SECONDS", 0.0),
        ("EMBED_TIMEOUT_SECONDS", -1.0),
        ("PORT", 0),
        ("PORT", 70000),
        ("MIN_WEIGHT_THRESHOLD", 1.0),
        ("MIN_WEIGHT_THRESHOLD", -0.1),
        ("TOPIC_SIMILARITY_THRESHOLD", 1.5),
        ("LLM_TEMPERATURE", -0.5),
        ("TRAJECTORY_BONUS", -0.1),
        ("EMBEDDING_SCORE_WEIGHT", -1.0),
        ("CORS_ORIGINS", []),
    ],
)
def test_validate_rejects_out_of_range_settings(monkeypatch, name, value):
    monkeypatch.setattr(config, name, value)
    with pytest.raises(ValueError):
        config.validate()


def test_validate_ignores_qdrant_port_when_not_in_server_mode(monkeypatch):
    """An unused port should not block a local or in-memory deployment."""
    monkeypatch.setattr(config, "QDRANT_MODE", "memory")
    monkeypatch.setattr(config, "QDRANT_PORT", 0)
    config.validate()


def test_history_bound_actually_bounds_at_its_minimum(monkeypatch):
    """The lower bound validate() enforces must still trim, not keep everything."""
    from backend.graph import memory_store as store

    monkeypatch.setattr(config, "MAX_HISTORY_MESSAGES", 2)
    config.validate()
    monkeypatch.setattr(store, "MAX_HISTORY_MESSAGES", 2)
    for i in range(8):
        store.update_session("bound", {"messages": [{"role": "human", "content": str(i)}]})
    assert len(store.get_session("bound")["messages"]) == 2


# ── Roster integrity ─────────────────────────────────────────────────

def test_mentor_ids_match_their_keys():
    for mentor_id, mentor in MENTORS.items():
        assert mentor.id == mentor_id


def test_mentor_colors_are_hex():
    for mentor in MENTORS.values():
        assert mentor.color.startswith("#") and len(mentor.color) == 7


def test_mentor_domains_are_known_topics():
    for mentor in MENTORS.values():
        assert set(mentor.domains) <= set(TOPIC_DOMAINS)
        assert set(mentor.domain_weights) <= set(TOPIC_DOMAINS)


def test_declared_domains_carry_weight():
    """A domain listed but unweighted would never influence routing."""
    for mentor in MENTORS.values():
        for domain in mentor.domains:
            assert mentor.domain_weights.get(domain, 0.0) > 0.0


def test_domain_weights_are_in_range():
    for mentor in MENTORS.values():
        assert all(0.0 <= w <= 1.0 for w in mentor.domain_weights.values())


def test_every_topic_has_at_least_one_specialist():
    for topic in TOPIC_DOMAINS:
        best = max(m.domain_weights.get(topic, 0.0) for m in MENTORS.values())
        assert best > 0.0, f"no mentor covers {topic}"
