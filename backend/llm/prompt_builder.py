"""Assemble system prompts from the current mentor weighting and retrieved quotes.

Retrieval feeds the prompt, not just the scoring: the quotes that made a mentor
score highly are quoted back to the model as grounding, so the answer reflects
what they actually said rather than the model's impression of them.
"""

from typing import Dict, List, Mapping, Optional, Sequence

from backend.mentors.roster import MENTORS, Mentor
from backend.scoring.mentor_scorer import Evidence

# How many quotes to ground each voice with. Enough to set a tone, few enough
# that the persona instructions still dominate the prompt.
ADAPTIVE_QUOTES_PER_MENTOR = 2
COUNCIL_QUOTES = 4
# Quotes below this similarity are noise dressed up as evidence.
MIN_EVIDENCE_SCORE = 0.30


def _relevant(evidence: Optional[Sequence[Evidence]], limit: int) -> List[Evidence]:
    if not evidence:
        return []
    strong = [e for e in evidence if e.score >= MIN_EVIDENCE_SCORE]
    return strong[:limit]


def _format_quotes(evidence: Sequence[Evidence]) -> List[str]:
    lines = []
    for item in evidence:
        source = f" — {item.source}" if item.source else ""
        lines.append(f'    "{item.text}"{source}')
    return lines


def build_adaptive_system_prompt(
    mentor_weights: Dict[str, float],
    evidence: Optional[Mapping[str, Sequence[Evidence]]] = None,
    top_k: int = 3,
) -> str:
    """One blended voice, weighted by the current distribution."""
    ranked = sorted(mentor_weights.items(), key=lambda kv: (-kv[1], kv[0]))
    contributors = [(mid, w) for mid, w in ranked if mid in MENTORS][:top_k]

    if not contributors:
        return (
            "You are a thoughtful life advisor. Be direct, concrete and honest. "
            "Avoid generic advice and platitudes."
        )

    lines = [
        "You are an AI life advisor that synthesizes the wisdom of several great "
        "thinkers. Channel their combined perspective in roughly these proportions:",
        "",
    ]

    for mentor_id, weight in contributors:
        mentor = MENTORS[mentor_id]
        lines.append(f"• {mentor.display_name} ({int(round(weight * 100))}%): {mentor.persona_prompt}")
        quotes = _relevant(evidence.get(mentor_id) if evidence else None, ADAPTIVE_QUOTES_PER_MENTOR)
        if quotes:
            lines.append("  Relevant things they have actually said:")
            lines.extend(_format_quotes(quotes))
        lines.append("")

    lines.append(
        "Synthesize these perspectives into ONE cohesive response in a single "
        "confident voice. Do NOT attribute lines to individual thinkers, do NOT "
        "name them, and do NOT quote the material above verbatim — let it shape "
        "your reasoning and priorities instead. The proportions describe how much "
        "each viewpoint should colour the answer, not how many words to spend. "
        "Be direct, specific and concrete. No platitudes, no numbered life-hack lists."
    )
    return "\n".join(lines)


def build_council_system_prompt(
    mentor: Mentor,
    evidence: Optional[Sequence[Evidence]] = None,
) -> str:
    """A single mentor answering in their own voice."""
    lines = [
        f"You are {mentor.display_name}. Respond exactly as they would — in their "
        f"distinctive voice, tone and style.",
        "",
        mentor.persona_prompt,
    ]

    quotes = _relevant(evidence, COUNCIL_QUOTES)
    if quotes:
        lines.extend(
            [
                "",
                "Things you have said that bear on this question — stay consistent "
                "with them, but do not simply recite them:",
            ]
        )
        lines.extend(_format_quotes(quotes))

    lines.extend(
        [
            "",
            "Be concise: 3-6 sentences. Open with your characteristic framing of the "
            "problem, then give one concrete piece of advice. Never break character "
            "and never say you are an AI.",
        ]
    )
    return "\n".join(lines)
