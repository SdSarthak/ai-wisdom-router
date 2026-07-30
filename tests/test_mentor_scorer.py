"""Mentor scoring and council selection."""

import pytest

from backend.mentors.roster import MENTORS
from backend.scoring import mentor_scorer as ms


# ── Domain scoring ───────────────────────────────────────────────────

def test_domain_score_matches_declared_expertise():
    goggins = MENTORS["david_goggins"]
    assert ms._domain_score(goggins, ["discipline"]) == pytest.approx(1.0)
    assert ms._domain_score(goggins, ["investing"]) == 0.0


def test_domain_score_averages_across_topics():
    goggins = MENTORS["david_goggins"]
    # discipline 1.0, investing 0.0 -> 0.5
    assert ms._domain_score(goggins, ["discipline", "investing"]) == pytest.approx(0.5)


def test_domain_score_with_no_topics_is_zero():
    assert ms._domain_score(MENTORS["paul_graham"], []) == 0.0


# ── Council selection ────────────────────────────────────────────────

def test_council_returns_everyone_when_roster_is_small():
    assert set(ms.select_council_mentors({"a": 0.5, "b": 0.4}, top_n=4)) == {"a", "b"}


def test_council_takes_the_top_n():
    scores = {"a": 0.9, "b": 0.8, "c": 0.7, "d": 0.6, "e": 0.1}
    assert ms.select_council_mentors(scores, top_n=3, min_score_gap=0.05) == ["a", "b", "c"]


def test_council_grants_an_extra_seat_on_a_near_tie():
    """A mentor who lost by a rounding error should still be heard."""
    scores = {"a": 0.9, "b": 0.8, "c": 0.7, "d": 0.69}
    assert ms.select_council_mentors(scores, top_n=3, min_score_gap=0.15) == [
        "a",
        "b",
        "c",
        "d",
    ]


def test_council_refuses_the_extra_seat_on_a_clear_gap():
    scores = {"a": 0.9, "b": 0.8, "c": 0.7, "d": 0.1}
    assert ms.select_council_mentors(scores, top_n=3, min_score_gap=0.15) == [
        "a",
        "b",
        "c",
    ]


def test_council_grants_at_most_one_extra_seat():
    scores = {"a": 0.90, "b": 0.89, "c": 0.88, "d": 0.87, "e": 0.86, "f": 0.85}
    assert len(ms.select_council_mentors(scores, top_n=3, min_score_gap=0.5)) == 4


def test_council_ties_break_deterministically():
    scores = {"c": 0.5, "a": 0.5, "b": 0.5, "d": 0.5}
    first = ms.select_council_mentors(scores, top_n=2, min_score_gap=0.0)
    second = ms.select_council_mentors(dict(reversed(list(scores.items()))), top_n=2, min_score_gap=0.0)
    assert first == second == ["a", "b"]


def test_council_with_zero_seats_is_empty():
    assert ms.select_council_mentors({"a": 1.0}, top_n=0) == []


# ── Scoring against the vector store ─────────────────────────────────

def test_scoring_without_a_vector_store_falls_back_to_domains(vector):
    """No Qdrant collection must not mean every mentor scores zero."""
    result = ms.score_all_mentors(vector("how do I stay disciplined"), ["discipline"])
    assert result.degraded is True
    assert set(result.scores) == set(MENTORS)
    assert result.scores["david_goggins"] > result.scores["warren_buffett"]


def test_scoring_with_no_query_vector_is_degraded():
    result = ms.score_all_mentors(None, ["investing"])
    assert result.degraded is True
    assert result.scores["warren_buffett"] > result.scores["david_goggins"]


def test_scoring_retrieves_evidence_from_a_seeded_store(seeded_store, vector):
    result = ms.score_all_mentors(vector("startup ideas and founders"), ["startup"])
    assert result.degraded is False
    assert set(result.scores) == set(MENTORS)
    # Every mentor gets their own quotes back, never another mentor's.
    for mentor_id, evidence in result.evidence.items():
        assert evidence, f"no evidence retrieved for {mentor_id}"
        assert all(e.text for e in evidence)


def test_evidence_scores_are_clamped_to_unit_range(seeded_store, vector):
    result = ms.score_all_mentors(vector("wealth and leverage"), ["wealth"])
    for evidence in result.evidence.values():
        assert all(0.0 <= e.score <= 1.0 for e in evidence)
    assert all(0.0 <= s <= 1.0 for s in result.scores.values())
