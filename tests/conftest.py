"""Shared test fixtures.

The suite never talks to Ollama or a Qdrant server. Embeddings are replaced with a
deterministic hash-based stand-in and generation with a scripted stub, so the
routing logic is tested rather than the model's mood. Qdrant runs in-process.

Environment is set before any backend module is imported, because config reads it
at import time.
"""

import os

# Must precede the first backend import.
os.environ.setdefault("QDRANT_MODE", "memory")
os.environ.setdefault("QDRANT_COLLECTION", "test_mentor_knowledge")
os.environ.setdefault("EMBEDDING_DIM", "16")
os.environ.setdefault("SEED_ON_STARTUP", "false")
os.environ.setdefault("OLLAMA_BASE_URL", "http://localhost:1")

import hashlib
import math
from typing import Dict, List

import pytest

from backend.config import EMBEDDING_DIM

TEST_DIM = EMBEDDING_DIM


def fake_vector(text: str, dim: int = TEST_DIM) -> List[float]:
    """A deterministic pseudo-embedding.

    Tokens are hashed into buckets, so texts sharing vocabulary land near each
    other under cosine similarity — enough structure for retrieval and topic
    detection to behave meaningfully in tests.
    """
    vec = [0.0] * dim
    tokens = [t for t in text.lower().split() if t]
    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        bucket = digest[0] % dim
        sign = 1.0 if digest[1] % 2 == 0 else -1.0
        vec[bucket] += sign
    norm = math.sqrt(sum(v * v for v in vec))
    if norm == 0.0:
        # An empty or stop-word-only string still needs a usable unit vector.
        return [1.0] + [0.0] * (dim - 1)
    return [v / norm for v in vec]


@pytest.fixture
def vector():
    """The stand-in embedding function, for tests that need a query vector."""
    return fake_vector


@pytest.fixture(autouse=True)
def fake_embeddings(monkeypatch):
    """Replace the Ollama embedding calls everywhere they are used.

    `backend.vector_store.embedder` imports these by name, and every other module
    reaches embeddings through that module, so patching here covers the app.
    """

    def _embed(texts: List[str]) -> List[List[float]]:
        return [fake_vector(t) for t in texts]

    async def _aembed(texts: List[str]) -> List[List[float]]:
        return [fake_vector(t) for t in texts]

    monkeypatch.setattr("backend.vector_store.embedder.embed", _embed)
    monkeypatch.setattr("backend.vector_store.embedder.aembed", _aembed)
    yield


@pytest.fixture(autouse=True)
def clean_state():
    """Reset module-level caches so tests cannot leak into one another."""
    from backend.graph import memory_store
    from backend.scoring import intent_analyzer
    from backend.vector_store import qdrant_client

    memory_store.clear_all()
    intent_analyzer.clear_anchors()
    qdrant_client.reset_client()
    yield
    memory_store.clear_all()
    intent_analyzer.clear_anchors()
    qdrant_client.reset_client()


@pytest.fixture
def seeded_store():
    """An in-memory Qdrant loaded with the real mentor corpus."""
    from backend.vector_store.seeder import seed_mentor_knowledge

    seed_mentor_knowledge()
    yield


@pytest.fixture
def stub_llm(monkeypatch):
    """Record every generation call and return a predictable answer."""
    calls: List[Dict[str, object]] = []

    async def _agenerate(system_prompt: str, user_message: str, history=None) -> str:
        calls.append(
            {
                "system_prompt": system_prompt,
                "user_message": user_message,
                "history": list(history or []),
            }
        )
        return f"answer to: {user_message}"

    monkeypatch.setattr("backend.graph.adaptive_graph.agenerate", _agenerate)
    monkeypatch.setattr("backend.graph.council_graph.agenerate", _agenerate)
    return calls


@pytest.fixture
def client(stub_llm):
    """A TestClient with the application lifespan actually running."""
    from fastapi.testclient import TestClient

    from backend.main import app

    with TestClient(app) as test_client:
        yield test_client
