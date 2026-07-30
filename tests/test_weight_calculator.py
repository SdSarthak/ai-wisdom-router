"""Weight blending: the mechanism that gives the advisor continuity."""

import pytest

from backend.mentors.roster import MENTORS
from backend.scoring import weight_calculator as wc


def test_initialize_weights_is_an_even_split():
    weights = wc.initialize_weights()
    assert set(weights) == set(MENTORS)
    assert pytest.approx(sum(weights.values()), abs=0.01) == 1.0
    assert len(set(weights.values())) == 1


def test_blend_weights_normalizes_to_one():
    old = wc.initialize_weights()
    scores = {mid: 0.5 for mid in MENTORS}
    blended = wc.blend_weights(old, scores, [])
    assert pytest.approx(sum(blended.values()), abs=0.01) == 1.0


def test_blend_weights_is_sorted_descending():
    scores = {
        "paul_graham": 0.9,
        "naval_ravikant": 0.5,
        "warren_buffett": 0.3,
        "david_goggins": 0.2,
        "charlie_munger": 0.4,
    }
    blended = wc.blend_weights(wc.initialize_weights(), scores, [])
    values = list(blended.values())
    assert values == sorted(values, reverse=True)
    assert next(iter(blended)) == "paul_graham"


def test_momentum_damps_a_sudden_swing():
    """One strong signal must not instantly hand the whole prompt to one mentor."""
    old = {"paul_graham": 0.0, "david_goggins": 1.0}
    scores = {"paul_graham": 1.0, "david_goggins": 0.0}
    blended = wc.blend_weights(old, scores, [])
    assert blended["paul_graham"] < 1.0
    assert blended.get("david_goggins", 0.0) > 0.0
    # But the new signal should still win the turn.
    assert blended["paul_graham"] > blended["david_goggins"]


def test_repeated_turns_converge_toward_the_new_signal():
    weights = wc.initialize_weights()
    scores = {mid: (1.0 if mid == "david_goggins" else 0.05) for mid in MENTORS}
    for _ in range(6):
        weights = wc.blend_weights(weights, scores, [])
    assert weights["david_goggins"] > 0.7


def test_trajectory_bonus_rewards_topic_specialists():
    scores = {mid: 0.4 for mid in MENTORS}
    flat = wc.blend_weights(wc.initialize_weights(), scores, [])
    # Three turns on one topic marks it as the conversation's direction.
    focused = wc.blend_weights(
        wc.initialize_weights(), scores, ["discipline", "discipline", "discipline"]
    )
    assert focused["david_goggins"] > flat["david_goggins"]


def test_trajectory_bonus_needs_a_full_window():
    assert wc._trajectory_bonus(["discipline"]) == {}
    assert wc._trajectory_bonus(["discipline", "startup"]) == {}
    assert wc._trajectory_bonus(["discipline"] * 3) != {}


def test_weak_mentors_are_dropped_below_the_threshold():
    scores = {
        "paul_graham": 1.0,
        "naval_ravikant": 1.0,
        "warren_buffett": 0.001,
        "david_goggins": 0.001,
        "charlie_munger": 0.001,
    }
    old = {"paul_graham": 0.5, "naval_ravikant": 0.5}
    blended = wc.blend_weights(old, scores, [])
    assert "warren_buffett" not in blended
    assert pytest.approx(sum(blended.values()), abs=0.01) == 1.0


def test_a_high_threshold_never_empties_the_roster(monkeypatch):
    """Guard the degenerate config where every mentor falls under the cut."""
    monkeypatch.setattr(wc, "MIN_WEIGHT_THRESHOLD", 0.99)
    blended = wc.blend_weights(wc.initialize_weights(), {mid: 0.5 for mid in MENTORS}, [])
    assert len(blended) == 1
    assert pytest.approx(sum(blended.values()), abs=0.01) == 1.0


def test_empty_scores_preserve_the_previous_distribution():
    old = wc.initialize_weights()
    assert wc.blend_weights(old, {}, []) == old


def test_scores_to_weights_normalizes_a_subset():
    result = wc.scores_to_weights({"a": 3.0, "b": 1.0, "c": 99.0}, ["a", "b"])
    assert set(result) == {"a", "b"}
    assert result["a"] == pytest.approx(0.75)
    assert result["b"] == pytest.approx(0.25)


def test_scores_to_weights_handles_an_all_zero_council():
    result = wc.scores_to_weights({"a": 0.0, "b": 0.0}, ["a", "b"])
    assert result == {"a": 0.5, "b": 0.5}


def test_scores_to_weights_ignores_negative_scores():
    result = wc.scores_to_weights({"a": 1.0, "b": -5.0}, ["a", "b"])
    assert result["b"] == 0.0
    assert result["a"] == pytest.approx(1.0)
