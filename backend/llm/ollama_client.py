"""Thin client for the Ollama HTTP API.

Ollama's surface is small enough that talking to it directly is simpler and far
more stable than pinning a LangChain integration package. Two endpoints are used:
`/api/chat` for generation and `/api/embed` for embeddings, with a fallback to the
older single-prompt `/api/embeddings` for Ollama versions predating batch support.
"""

import re
from typing import Any, Dict, List, Optional

import httpx

from backend.config import (
    EMBED_TIMEOUT_SECONDS,
    EMBEDDING_MODEL,
    LLM_MODEL,
    LLM_TEMPERATURE,
    LLM_TIMEOUT_SECONDS,
    OLLAMA_BASE_URL,
)

# Reasoning models such as qwen3 wrap their scratchpad in <think> tags. It is
# not part of the answer and must never reach the user.
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_ORPHAN_THINK = re.compile(r"^.*?</think>", re.DOTALL | re.IGNORECASE)

_ROLE_MAP = {"human": "user", "user": "user", "assistant": "assistant"}


class OllamaError(RuntimeError):
    """Ollama is unreachable, timed out, or returned an unusable response.

    `status_code` is set when the failure was an HTTP status rather than a
    transport error, so callers can tell "this endpoint does not exist on this
    Ollama version" apart from "Ollama is not answering at all".
    """

    def __init__(self, message: str, status_code: Optional[int] = None) -> None:
        super().__init__(message)
        self.status_code = status_code


# Statuses that mean "this Ollama build does not serve that route" — the only
# case in which falling back to an older endpoint is the right move.
_ENDPOINT_MISSING = frozenset({404, 405, 501})


def strip_thinking(text: str) -> str:
    """Remove reasoning-model scratchpad blocks from a completion."""
    cleaned = _THINK_BLOCK.sub("", text)
    # A truncated or unopened block can leave a closing tag with no opener.
    if "</think>" in cleaned:
        cleaned = _ORPHAN_THINK.sub("", cleaned)
    # An opener with no closer means generation was cut off mid-reasoning.
    # Everything from that point on is scratchpad; returning it would render
    # the model's private deliberation to the user as if it were the answer.
    opener = cleaned.lower().find("<think>")
    if opener != -1:
        cleaned = cleaned[:opener]
    return cleaned.strip()


def build_messages(
    system_prompt: str,
    user_message: str,
    history: Optional[List[Dict[str, str]]] = None,
    history_limit: int = 6,
) -> List[Dict[str, str]]:
    """Assemble an Ollama chat payload from a system prompt, history and a turn."""
    messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]
    for turn in (history or [])[-history_limit:]:
        role = _ROLE_MAP.get(turn.get("role", ""))
        content = turn.get("content", "")
        if role is None or not content:
            continue
        messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": user_message})
    return messages


def _wrap_http_error(exc: Exception, url: str, path: str) -> "OllamaError":
    if isinstance(exc, httpx.HTTPStatusError):
        return OllamaError(
            f"Ollama returned {exc.response.status_code} for {path}: "
            f"{exc.response.text[:200]}",
            status_code=exc.response.status_code,
        )
    if isinstance(exc, httpx.HTTPError):
        return OllamaError(f"Could not reach Ollama at {url}: {exc}")
    return OllamaError(f"Ollama returned malformed JSON from {path}: {exc}")


def _post(path: str, payload: Dict[str, Any], timeout: float) -> Dict[str, Any]:
    url = f"{OLLAMA_BASE_URL}{path}"
    try:
        response = httpx.post(url, json=payload, timeout=timeout)
        response.raise_for_status()
        return response.json()
    except (httpx.HTTPError, ValueError) as exc:
        raise _wrap_http_error(exc, url, path) from exc


async def _apost(path: str, payload: Dict[str, Any], timeout: float) -> Dict[str, Any]:
    url = f"{OLLAMA_BASE_URL}{path}"
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            return response.json()
    except (httpx.HTTPError, ValueError) as exc:
        raise _wrap_http_error(exc, url, path) from exc


# ── Chat ─────────────────────────────────────────────────────────────

def _chat_payload(messages: List[Dict[str, str]]) -> Dict[str, Any]:
    return {
        "model": LLM_MODEL,
        "messages": messages,
        "stream": False,
        "options": {"temperature": LLM_TEMPERATURE},
    }


def _extract_content(data: Dict[str, Any]) -> str:
    content = (data.get("message") or {}).get("content", "") or ""
    text = strip_thinking(content) if content else ""
    if not text:
        if "<think>" in content.lower():
            # The model spent its whole budget reasoning and never reached an
            # answer. Saying "is the model pulled?" here would send the reader
            # chasing the wrong problem.
            raise OllamaError(
                f"{LLM_MODEL!r} produced only reasoning and no answer — the "
                "completion was cut short. Raise LLM_TIMEOUT_SECONDS, or "
                "configure LLM_MODEL to a model without a reasoning scratchpad."
            )
        raise OllamaError(
            f"Ollama returned an empty completion for model {LLM_MODEL!r}. "
            f"Is the model pulled? Try: ollama pull {LLM_MODEL}"
        )
    return text


def generate(
    system_prompt: str,
    user_message: str,
    history: Optional[List[Dict[str, str]]] = None,
) -> str:
    """Blocking completion. Used by scripts and the smoke-test CLI."""
    messages = build_messages(system_prompt, user_message, history)
    return _extract_content(_post("/api/chat", _chat_payload(messages), LLM_TIMEOUT_SECONDS))


async def agenerate(
    system_prompt: str,
    user_message: str,
    history: Optional[List[Dict[str, str]]] = None,
) -> str:
    """Async completion. Council mode fans these out concurrently."""
    messages = build_messages(system_prompt, user_message, history)
    data = await _apost("/api/chat", _chat_payload(messages), LLM_TIMEOUT_SECONDS)
    return _extract_content(data)


# ── Embeddings ───────────────────────────────────────────────────────

def _parse_embeddings(data: Dict[str, Any], expected: int) -> List[List[float]]:
    vectors = data.get("embeddings")
    if vectors is None and "embedding" in data:
        vectors = [data["embedding"]]
    if not vectors or len(vectors) != expected:
        raise OllamaError(
            f"Ollama returned {len(vectors or [])} embeddings for {expected} input(s). "
            f"Is the model pulled? Try: ollama pull {EMBEDDING_MODEL}"
        )
    return vectors


def _legacy_payload(text: str) -> Dict[str, Any]:
    return {"model": EMBEDDING_MODEL, "prompt": text}


def _embed_one(text: str) -> List[float]:
    data = _post("/api/embeddings", _legacy_payload(text), EMBED_TIMEOUT_SECONDS)
    return _parse_embeddings(data, 1)[0]


async def _aembed_one(text: str) -> List[float]:
    data = await _apost("/api/embeddings", _legacy_payload(text), EMBED_TIMEOUT_SECONDS)
    return _parse_embeddings(data, 1)[0]


def _should_fall_back(exc: OllamaError) -> bool:
    """Only retry on the legacy endpoint when the batch route is genuinely absent.

    Falling back on *any* failure turns one unreachable host into one request
    per text, each waiting out EMBED_TIMEOUT_SECONDS — seeding the corpus would
    hang for len(quotes) x timeout before reporting the same connection error.
    """
    return exc.status_code in _ENDPOINT_MISSING


def embed(texts: List[str]) -> List[List[float]]:
    """Blocking batch embedding."""
    if not texts:
        return []
    payload = {"model": EMBEDDING_MODEL, "input": texts}
    try:
        data = _post("/api/embed", payload, EMBED_TIMEOUT_SECONDS)
    except OllamaError as exc:
        # Ollama < 0.3.4 only exposes the single-prompt /api/embeddings endpoint.
        if not _should_fall_back(exc):
            raise
        return [_embed_one(t) for t in texts]
    return _parse_embeddings(data, len(texts))


async def aembed(texts: List[str]) -> List[List[float]]:
    """Async batch embedding.

    Carries the same legacy fallback as `embed`: without it every chat turn
    fails on an Ollama older than 0.3.4 even though seeding, which takes the
    blocking path, succeeds.
    """
    if not texts:
        return []
    payload = {"model": EMBEDDING_MODEL, "input": texts}
    try:
        data = await _apost("/api/embed", payload, EMBED_TIMEOUT_SECONDS)
    except OllamaError as exc:
        if not _should_fall_back(exc):
            raise
        return [await _aembed_one(t) for t in texts]
    return _parse_embeddings(data, len(texts))


# ── Health ───────────────────────────────────────────────────────────

def ping(timeout: float = 5.0) -> Dict[str, Any]:
    """Report whether Ollama is up and whether the configured models are pulled."""
    try:
        response = httpx.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=timeout)
        response.raise_for_status()
        installed = {m.get("name", "") for m in response.json().get("models", [])}
    except (httpx.HTTPError, ValueError) as exc:
        return {"reachable": False, "detail": str(exc)}

    def _has(model: str) -> bool:
        # Ollama reports "name:tag"; a bare name implies the :latest tag.
        return model in installed or (f"{model}:latest" in installed)

    return {
        "reachable": True,
        "llm_model": LLM_MODEL,
        "llm_model_available": _has(LLM_MODEL),
        "embedding_model": EMBEDDING_MODEL,
        "embedding_model_available": _has(EMBEDDING_MODEL),
    }
