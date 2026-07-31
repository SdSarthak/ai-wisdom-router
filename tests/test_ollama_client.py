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


def test_unterminated_thinking_block_is_not_leaked():
    """A completion cut off mid-reasoning is all scratchpad, not an answer."""
    assert oc.strip_thinking("<think>the user probably means") == ""


def test_answer_before_an_unterminated_block_is_kept():
    raw = "Do the hard thing.<think>should I add a caveat"
    assert oc.strip_thinking(raw) == "Do the hard thing."


def test_pairs_are_removed_before_the_unclosed_check():
    raw = "<think>first</think>Real answer.<think>second thought"
    assert oc.strip_thinking(raw) == "Real answer."


def test_reasoning_only_completion_does_not_blame_a_missing_model(monkeypatch):
    """Telling the reader to pull a model they already have wastes their time."""
    def handler(url, payload):
        return httpx.Response(
            200,
            json={"message": {"content": "<think>still deliberating"}},
            request=httpx.Request("POST", url),
        )

    _mock_post(monkeypatch, handler)
    with pytest.raises(oc.OllamaError, match="only reasoning"):
        oc.generate("SYS", "hi")


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


def test_embed_does_not_retry_every_text_when_ollama_is_down(monkeypatch):
    """A transport failure is not a missing endpoint.

    Retrying per text would cost len(texts) x EMBED_TIMEOUT_SECONDS before
    reporting the same connection error that the first call already proved.
    """
    calls = []

    def handler(url, payload):
        calls.append(url)
        raise httpx.ConnectError("refused", request=httpx.Request("POST", url))

    _mock_post(monkeypatch, handler)
    with pytest.raises(oc.OllamaError, match="Could not reach Ollama"):
        oc.embed(["a", "b", "c", "d"])
    assert calls == [calls[0]]
    assert calls[0].endswith("/api/embed")


def test_embed_does_not_fall_back_on_a_server_error(monkeypatch):
    calls = []

    def handler(url, payload):
        calls.append(url)
        return httpx.Response(500, text="boom", request=httpx.Request("POST", url))

    _mock_post(monkeypatch, handler)
    with pytest.raises(oc.OllamaError, match="500"):
        oc.embed(["a", "b"])
    assert len(calls) == 1


def test_http_status_errors_carry_their_status_code(monkeypatch):
    def handler(url, payload):
        return httpx.Response(404, text="nope", request=httpx.Request("POST", url))

    _mock_post(monkeypatch, handler)
    with pytest.raises(oc.OllamaError) as excinfo:
        oc.generate("SYS", "hi")
    assert excinfo.value.status_code == 404


def test_transport_errors_have_no_status_code(monkeypatch):
    def handler(url, payload):
        raise httpx.ConnectError("refused", request=httpx.Request("POST", url))

    _mock_post(monkeypatch, handler)
    with pytest.raises(oc.OllamaError) as excinfo:
        oc.generate("SYS", "hi")
    assert excinfo.value.status_code is None


# ── Async transport ──────────────────────────────────────────────────

def _mock_apost(monkeypatch, handler):
    class _FakeAsyncClient:
        def __init__(self, timeout=None):
            self.timeout = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc_info):
            return False

        async def post(self, url, json=None):
            return handler(url, json)

    monkeypatch.setattr(oc.httpx, "AsyncClient", _FakeAsyncClient)


async def test_aembed_uses_the_batch_endpoint(monkeypatch):
    seen = {}

    def handler(url, payload):
        seen["url"] = url
        return httpx.Response(
            200,
            json={"embeddings": [[0.1], [0.2]]},
            request=httpx.Request("POST", url),
        )

    _mock_apost(monkeypatch, handler)
    assert await oc.aembed(["a", "b"]) == [[0.1], [0.2]]
    assert seen["url"].endswith("/api/embed")


async def test_aembed_falls_back_to_the_single_prompt_endpoint(monkeypatch):
    """Every chat turn embeds through aembed; without the fallback the whole
    app is unusable on Ollama < 0.3.4 even though seeding works."""
    calls = []

    def handler(url, payload):
        calls.append(url)
        if url.endswith("/api/embed"):
            return httpx.Response(404, text="not found", request=httpx.Request("POST", url))
        return httpx.Response(
            200, json={"embedding": [1.0, 2.0]}, request=httpx.Request("POST", url)
        )

    _mock_apost(monkeypatch, handler)
    assert await oc.aembed(["a", "b"]) == [[1.0, 2.0], [1.0, 2.0]]
    assert sum(1 for u in calls if u.endswith("/api/embeddings")) == 2


async def test_aembed_does_not_retry_every_text_when_ollama_is_down(monkeypatch):
    calls = []

    def handler(url, payload):
        calls.append(url)
        raise httpx.ConnectError("refused", request=httpx.Request("POST", url))

    _mock_apost(monkeypatch, handler)
    with pytest.raises(oc.OllamaError, match="Could not reach Ollama"):
        await oc.aembed(["a", "b", "c"])
    assert len(calls) == 1


async def test_aembed_of_nothing_makes_no_request(monkeypatch):
    def handler(url, payload):
        raise AssertionError("should not have called Ollama")

    _mock_apost(monkeypatch, handler)
    assert await oc.aembed([]) == []


async def test_agenerate_strips_reasoning(monkeypatch):
    def handler(url, payload):
        assert url.endswith("/api/chat")
        return httpx.Response(
            200,
            json={"message": {"content": "<think>hmm</think>Do it now."}},
            request=httpx.Request("POST", url),
        )

    _mock_apost(monkeypatch, handler)
    assert await oc.agenerate("SYS", "when?") == "Do it now."


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
