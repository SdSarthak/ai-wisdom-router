"""Runtime configuration.

Every value can be overridden with an environment variable of the same name,
either exported in the shell or placed in a `.env` file at the project root.
See `.env.example` for the full list with defaults.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(PROJECT_ROOT / ".env")


def _str(name: str, default: str) -> str:
    value = os.getenv(name)
    return default if value is None or value.strip() == "" else value.strip()


def _int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        raise ValueError(f"{name} must be an integer, got {raw!r}")


def _float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        raise ValueError(f"{name} must be a number, got {raw!r}")


def _bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


# ── Ollama ───────────────────────────────────────────────────────────
OLLAMA_BASE_URL = _str("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
LLM_MODEL = _str("LLM_MODEL", "qwen3:14b")
EMBEDDING_MODEL = _str("EMBEDDING_MODEL", "bge-m3")
EMBEDDING_DIM = _int("EMBEDDING_DIM", 1024)
LLM_TEMPERATURE = _float("LLM_TEMPERATURE", 0.7)
LLM_TIMEOUT_SECONDS = _float("LLM_TIMEOUT_SECONDS", 180.0)
EMBED_TIMEOUT_SECONDS = _float("EMBED_TIMEOUT_SECONDS", 60.0)

# ── Qdrant ───────────────────────────────────────────────────────────
# "server"  -> talk to a Qdrant instance over HTTP (docker run qdrant/qdrant)
# "local"   -> embedded on-disk store at QDRANT_PATH, no server needed
# "memory"  -> embedded, in-process, wiped on restart (used by the test suite)
QDRANT_MODE = _str("QDRANT_MODE", "local").lower()
QDRANT_HOST = _str("QDRANT_HOST", "localhost")
QDRANT_PORT = _int("QDRANT_PORT", 6333)
QDRANT_PATH = _str("QDRANT_PATH", str(PROJECT_ROOT / "qdrant_data"))
QDRANT_COLLECTION = _str("QDRANT_COLLECTION", "mentor_knowledge")
QDRANT_SEARCH_LIMIT = _int("QDRANT_SEARCH_LIMIT", 5)

# ── Weight dynamics ──────────────────────────────────────────────────
# How much of the previous turn's weighting carries over (0 = no memory,
# 1 = frozen). Higher values make the advisor's voice shift more slowly.
WEIGHT_MOMENTUM = _float("WEIGHT_MOMENTUM", 0.35)
# Mentors below this share of the distribution are dropped entirely.
MIN_WEIGHT_THRESHOLD = _float("MIN_WEIGHT_THRESHOLD", 0.05)
# Cosine similarity a message must reach to count as being "about" a topic.
TOPIC_SIMILARITY_THRESHOLD = _float("TOPIC_SIMILARITY_THRESHOLD", 0.45)

# Relative pull of corpus similarity vs. declared domain expertise.
EMBEDDING_SCORE_WEIGHT = _float("EMBEDDING_SCORE_WEIGHT", 0.6)
DOMAIN_SCORE_WEIGHT = _float("DOMAIN_SCORE_WEIGHT", 0.4)

# Staying on one topic for this many turns rewards its specialists.
TRAJECTORY_WINDOW = _int("TRAJECTORY_WINDOW", 3)
TRAJECTORY_BONUS = _float("TRAJECTORY_BONUS", 0.10)

# ── Council mode ─────────────────────────────────────────────────────
COUNCIL_TOP_N = _int("COUNCIL_TOP_N", 4)
# A trailing mentor within this score gap of the cut line still gets a seat.
COUNCIL_MIN_SCORE_GAP = _float("COUNCIL_MIN_SCORE_GAP", 0.15)

# ── Session store ────────────────────────────────────────────────────
# Turns of history kept per session, and how many sessions live in memory
# before the least recently used ones are evicted.
MAX_HISTORY_MESSAGES = _int("MAX_HISTORY_MESSAGES", 20)
MAX_SESSIONS = _int("MAX_SESSIONS", 500)

# ── Server ───────────────────────────────────────────────────────────
HOST = _str("HOST", "127.0.0.1")
PORT = _int("PORT", 8000)
# Comma-separated list, or "*" to allow any origin.
CORS_ORIGINS = [o.strip() for o in _str("CORS_ORIGINS", "*").split(",") if o.strip()]
# Seed the vector store on startup. Turn off for a fast boot once seeded.
SEED_ON_STARTUP = _bool("SEED_ON_STARTUP", True)


def validate() -> None:
    """Fail fast on configuration that would break at request time.

    Several of these guard settings whose invalid values fail *silently* rather
    than loudly — a zero window turns a `list[-n:]` tail slice into the whole
    list, which is the opposite of the intended bound.
    """
    if QDRANT_MODE not in ("server", "local", "memory"):
        raise ValueError(
            f"QDRANT_MODE must be one of 'server', 'local', 'memory' — got {QDRANT_MODE!r}"
        )
    if not 0.0 <= WEIGHT_MOMENTUM < 1.0:
        raise ValueError("WEIGHT_MOMENTUM must be in [0.0, 1.0)")
    if EMBEDDING_DIM <= 0:
        raise ValueError("EMBEDDING_DIM must be positive")
    if COUNCIL_TOP_N < 1:
        raise ValueError("COUNCIL_TOP_N must be at least 1")
    if EMBEDDING_SCORE_WEIGHT + DOMAIN_SCORE_WEIGHT <= 0:
        raise ValueError(
            "EMBEDDING_SCORE_WEIGHT + DOMAIN_SCORE_WEIGHT must be greater than 0"
        )
    if EMBEDDING_SCORE_WEIGHT < 0 or DOMAIN_SCORE_WEIGHT < 0:
        raise ValueError(
            "EMBEDDING_SCORE_WEIGHT and DOMAIN_SCORE_WEIGHT must not be negative"
        )

    # MAX_HISTORY_MESSAGES <= 0 makes `combined[-MAX_HISTORY_MESSAGES:]` return
    # the entire list, so the session store would grow without bound — exactly
    # the failure the setting exists to prevent.
    if MAX_HISTORY_MESSAGES < 2:
        raise ValueError(
            "MAX_HISTORY_MESSAGES must be at least 2 (one user turn and one reply)"
        )
    if MAX_SESSIONS < 1:
        raise ValueError("MAX_SESSIONS must be at least 1")

    # TRAJECTORY_WINDOW <= 0 has the same tail-slice problem, and every topic
    # then trivially clears a count of >= 0, handing the bonus to everyone.
    if TRAJECTORY_WINDOW < 1:
        raise ValueError("TRAJECTORY_WINDOW must be at least 1")
    if TRAJECTORY_BONUS < 0:
        raise ValueError("TRAJECTORY_BONUS must not be negative")

    if QDRANT_SEARCH_LIMIT < 1:
        raise ValueError("QDRANT_SEARCH_LIMIT must be at least 1")
    if not 0.0 <= MIN_WEIGHT_THRESHOLD < 1.0:
        raise ValueError("MIN_WEIGHT_THRESHOLD must be in [0.0, 1.0)")
    if not -1.0 <= TOPIC_SIMILARITY_THRESHOLD <= 1.0:
        raise ValueError(
            "TOPIC_SIMILARITY_THRESHOLD is a cosine similarity and must be in [-1.0, 1.0]"
        )
    if not 0.0 <= LLM_TEMPERATURE <= 2.0:
        raise ValueError("LLM_TEMPERATURE must be in [0.0, 2.0]")
    if LLM_TIMEOUT_SECONDS <= 0 or EMBED_TIMEOUT_SECONDS <= 0:
        raise ValueError(
            "LLM_TIMEOUT_SECONDS and EMBED_TIMEOUT_SECONDS must be greater than 0"
        )
    if not 1 <= PORT <= 65535:
        raise ValueError(f"PORT must be in 1..65535 — got {PORT}")
    if QDRANT_MODE == "server" and not 1 <= QDRANT_PORT <= 65535:
        raise ValueError(f"QDRANT_PORT must be in 1..65535 — got {QDRANT_PORT}")
    if not CORS_ORIGINS:
        raise ValueError("CORS_ORIGINS must name at least one origin, or be '*'")
