"""Embedding helpers on top of the Ollama client.

Vectors are validated against EMBEDDING_DIM here rather than at insert time, so a
model/dimension mismatch surfaces with a message that says how to fix it instead of
as an opaque Qdrant rejection.
"""

from typing import List

from backend.config import EMBEDDING_DIM, EMBEDDING_MODEL
from backend.llm.ollama_client import OllamaError, aembed, embed


def _check_dims(vectors: List[List[float]]) -> List[List[float]]:
    for vec in vectors:
        if len(vec) != EMBEDDING_DIM:
            raise OllamaError(
                f"{EMBEDDING_MODEL!r} produced {len(vec)}-dimensional vectors but "
                f"EMBEDDING_DIM is {EMBEDDING_DIM}. Set EMBEDDING_DIM={len(vec)} "
                "and delete the existing Qdrant collection so it is rebuilt."
            )
    return vectors


def embed_text(text: str) -> List[float]:
    return _check_dims(embed([text]))[0]


def embed_texts(texts: List[str]) -> List[List[float]]:
    return _check_dims(embed(texts))


async def aembed_text(text: str) -> List[float]:
    return _check_dims(await aembed([text]))[0]


async def aembed_texts(texts: List[str]) -> List[List[float]]:
    return _check_dims(await aembed(texts))
