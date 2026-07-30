from typing import Dict
from backend.mentors.roster import MENTORS, Mentor


def build_adaptive_system_prompt(mentor_weights: Dict[str, float]) -> str:
    top_mentors = sorted(mentor_weights.items(), key=lambda x: x[1], reverse=True)[:3]

    lines = [
        "You are an AI life advisor that synthesizes the wisdom of multiple great thinkers. "
        "Channel their combined perspective with these approximate weights in your response:",
    ]
    for mentor_id, weight in top_mentors:
        if mentor_id not in MENTORS:
            continue
        mentor = MENTORS[mentor_id]
        pct = int(weight * 100)
        lines.append(f"\n• {mentor.display_name} ({pct}%): {mentor.persona_prompt}")

    lines.append(
        "\n\nSynthesize their perspectives into ONE cohesive, unified response. "
        "Do NOT attribute quotes or label which mentor said what. "
        "Write as a single confident voice that naturally blends their styles. "
        "Be direct, insightful, and concrete. Avoid generic advice."
    )
    return "\n".join(lines)


def build_council_system_prompt(mentor: Mentor) -> str:
    return (
        f"You are {mentor.display_name}. Respond exactly as they would — "
        f"in their distinctive voice, tone, and style.\n\n"
        f"{mentor.persona_prompt}\n\n"
        f"Be concise (3-6 sentences). Start with your characteristic framing of the problem. "
        f"Do NOT break character. Do NOT say you are an AI."
    )
