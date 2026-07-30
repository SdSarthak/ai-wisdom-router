"""The Ollama HTTP client, exercised against stubbed transport."""

import httpx
import pytest

from backend.llm import ollama_client as oc


# ── Reasoning-model scratchpad ───────────────────────────────────────

def test_thinking_block_is_removed():
    assert oc.strip_thinking("<think>weighing options</think>Do the hard thing.") == (
        "Do the hard thing."
    )


def test_multiline_thinking_block_is_removed():
    raw = "<think>\nline one\nline two\n</think>\n\nThe answer."
    assert oc.strip_thinking(raw) == "The answer."


def test_orphan_closing_tag_is_removed():
    """A truncated stream can emit </think> with no opener."""
    assert oc.strip_thinking("stray reasoning</think>The answer.") == "The answer."


def test_text_without_thinking_is_untouched():
    assert oc.strip_thinking("  Plain answer.  ") == "Plain answer."


# ── Message assembly ─────────────────────────────────────────────────

def test_system_prompt_comes_first_and_message_last():
    messages = oc.build_messages("SYS", "hello")
    assert messages[0] == {"role": "system", "content": "SYS"}
    assert messages[-1] == {"role": "user", "content": "hello"}


def test_internal_human_role_maps_to_ollamas_user_role():
    messages = oc.build_messages("SYS", "now", [{"role": "human", "content": "before"}])
    assert messages[1] == {"role": "user", "content": "before"}


def test_history_is_capped():
    history = [{"role": "human", "content": str(i)} for i in range(50)]
    messages = oc.build_messages("SYS", "now", history, history_limit=4)
    # system + 4 history + current
    assert len(messages) == 6
    assert messages[1]["content"] == "46"


def test_unknown_roles_and_empty_turns_are_dropped():
    history = [
        {"role": "system", "content": "injected"},
        {"role": "human", "content": ""},
        {"role": "assistant", "content": "kept"},
    ]
    messages = oc.build_messages("SYS", "now", history)
    assert [m["content"] for m in messages] == ["SYS", "kept", "now"]


# ── Transport behaviour ──────────────────────────────────────────────

def _mock_post(monkeypatch, handler):
    def _fake(url, json=None, timeout=None):
        return handler(url, json)

    monkeypatch.setattr(oc.httpx, "post", _fake)


def test_generate_returns_cleaned_content(monkeypatch):
    def handler(url, payload):
        assert url.endswith("/api/chat")
        assert payload["stream"] is False
        return httpx.Response(
            200,
            json={"message": {"role": "assistant", "content": "<think>x</think>Ship it."}},
            request=httpx.Request("POST", url),
        )

    _mock_post(monkeypatch, handler)
    assert oc.generate("SYS", "should I ship?") == "Ship it."


def test_connection_failure_raises_ollama_error(monkeypatch):
    def handler(url, payload):
        raise httpx.ConnectError("refused", request=httpx.Request("POST", url))

    _mock_post(monkeypatch, handler)
    with pytest.raises(oc.OllamaError, match="Could not reach Ollama"):
        oc.generate("SYS", "hi")


def test_http_error_status_raises_ollama_error(monkeypatch):
    def handler(url, payload):
        return httpx.Response(404, text="model not found", request=httpx.Request("POST", url))

    _mock_post(monkeypatch, handler)
    with pytest.raises(oc.OllamaError, match="404"):
        oc.generate("SYS", "hi")


def test_empty_completion_names_the_model_to_pull(monkeypatch):
    def handler(url, payload):
        return httpx.Response(
            200, json={"message": {"content": ""}}, request=httpx.Request("POST", url)
        )

    _mock_post(monkeypatch, handler)
    with pytest.raises(oc.OllamaError, match="ollama pull"):
        oc.generate("SYS", "hi")


def test_embed_uses_the_batch_endpoint(monkeypatch):
    seen = {}

    def handler(url, payload):
        seen["url"] = url
        seen["input"] = payload["input"]
        return httpx.Response(
            200,
            json={"embeddings": [[0.1, 0.2], [0.3, 0.4]]},
            request=httpx.Request("POST", url),
        )

    _mock_post(monkeypatch, handler)
    vectors = oc.embed(["a", "b"])
    assert seen["url"].endswith("/api/embed")
    assert seen["input"] == ["a", "b"]
    assert vectors == [[0.1, 0.2], [0.3, 0.4]]


def test_embed_falls_back_to_the_single_prompt_endpoint(monkeypatch):
    """Ollama below 0.3.4 has no /api/embed."""
    calls = []

    def handler(url, payload):
        calls.append(url)
        if url.endswith("/api/embed"):
            return httpx.Response(404, text="not found", request=httpx.Request("POST", url))
        return httpx.Response(
            200, json={"embedding": [1.0, 2.0]}, request=httpx.Request("POST", url)
        )

    _mock_post(monkeypatch, handler)
    assert oc.embed(["a", "b"]) == [[1.0, 2.0], [1.0, 2.0]]
    assert any(u.endswith("/api/embeddings") for u in calls)


def test_embedding_count_mismatch_is_an_error(monkeypatch):
    def handler(url, payload):
        return httpx.Response(
            200, json={"embeddings": [[0.1]]}, request=httpx.Request("POST", url)
        )

    _mock_post(monkeypatch, handler)
    with pytest.raises(oc.OllamaError):
        oc._parse_embeddings({"embeddings": [[0.1]]}, 2)


def test_embed_of_nothing_makes_no_request(monkeypatch):
    def handler(url, payload):
        raise AssertionError("should not have called Ollama")

    _mock_post(monkeypatch, handler)
    assert oc.embed([]) == []


def test_ping_reports_unreachable(monkeypatch):
    def _fake_get(url, timeout=None):
        raise httpx.ConnectError("refused", request=httpx.Request("GET", url))

    monkeypatch.setattr(oc.httpx, "get", _fake_get)
    assert oc.ping()["reachable"] is False


def test_ping_detects_installed_models(monkeypatch):
    def _fake_get(url, timeout=None):
        return httpx.Response(
            200,
            json={"models": [{"name": f"{oc.LLM_MODEL}"}, {"name": "other:latest"}]},
            request=httpx.Request("GET", url),
        )

    monkeypatch.setattr(oc.httpx, "get", _fake_get)
    result = oc.ping()
    assert result["reachable"] is True
    assert result["llm_model_available"] is True
    assert result["embedding_model_available"] is False
