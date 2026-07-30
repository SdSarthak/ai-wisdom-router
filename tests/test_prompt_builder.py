"""System prompt assembly, including retrieval grounding."""

from backend.llm import prompt_builder as pb
from backend.mentors.roster import MENTORS
from backend.scoring.mentor_scorer import Evidence


def _evidence(text, score=0.9, source="Essay"):
    return Evidence(text=text, source=source, score=score)


def test_adaptive_prompt_lists_the_heaviest_mentors():
    weights = {
        "paul_graham": 0.5,
        "naval_ravikant": 0.3,
        "warren_buffett": 0.15,
        "david_goggins": 0.05,
    }
    prompt = pb.build_adaptive_system_prompt(weights, top_k=3)
    assert "Paul Graham (50%)" in prompt
    assert "Naval Ravikant (30%)" in prompt
    assert "Warren Buffett (15%)" in prompt
    assert "David Goggins" not in prompt


def test_adaptive_prompt_forbids_attribution():
    prompt = pb.build_adaptive_system_prompt({"paul_graham": 1.0})
    assert "do NOT attribute" in prompt.lower() or "not attribute" in prompt.lower()


def test_adaptive_prompt_includes_retrieved_quotes():
    """The point of the vector store: real quotes reach the prompt."""
    evidence = {"paul_graham": [_evidence("Do things that don't scale.")]}
    prompt = pb.build_adaptive_system_prompt({"paul_graham": 1.0}, evidence=evidence)
    assert "Do things that don't scale." in prompt


def test_weak_evidence_is_filtered_out():
    evidence = {"paul_graham": [_evidence("Barely related quote.", score=0.05)]}
    prompt = pb.build_adaptive_system_prompt({"paul_graham": 1.0}, evidence=evidence)
    assert "Barely related quote." not in prompt


def test_adaptive_prompt_limits_quotes_per_mentor():
    quotes = [_evidence(f"Quote number {i}.") for i in range(10)]
    prompt = pb.build_adaptive_system_prompt(
        {"paul_graham": 1.0}, evidence={"paul_graham": quotes}
    )
    included = sum(1 for i in range(10) if f"Quote number {i}." in prompt)
    assert included == pb.ADAPTIVE_QUOTES_PER_MENTOR


def test_adaptive_prompt_survives_an_unknown_mentor_id():
    prompt = pb.build_adaptive_system_prompt({"nobody_at_all": 1.0})
    assert prompt
    assert "life advisor" in prompt


def test_adaptive_prompt_works_without_evidence():
    prompt = pb.build_adaptive_system_prompt({"charlie_munger": 1.0}, evidence=None)
    assert "Charlie Munger" in prompt


def test_council_prompt_stays_in_character():
    prompt = pb.build_council_system_prompt(MENTORS["david_goggins"])
    assert "David Goggins" in prompt
    assert "never say you are an ai" in prompt.lower()
    assert MENTORS["david_goggins"].persona_prompt in prompt


def test_council_prompt_includes_evidence_with_sources():
    prompt = pb.build_council_system_prompt(
        MENTORS["warren_buffett"],
        evidence=[_evidence("Be fearful when others are greedy.", source="1986 Letter")],
    )
    assert "Be fearful when others are greedy." in prompt
    assert "1986 Letter" in prompt


def test_council_prompt_without_evidence_has_no_empty_quote_header():
    prompt = pb.build_council_system_prompt(MENTORS["naval_ravikant"], evidence=[])
    assert "Things you have said" not in prompt
