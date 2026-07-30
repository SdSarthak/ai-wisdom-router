"""Qdrant connection management.

Three modes are supported (see QDRANT_MODE in config):

  server  — HTTP connection to a running Qdrant instance
  local   — embedded on-disk store; no server to run, survives restarts
  memory  — embedded, in-process, wiped on restart

`local` is the default so a fresh clone works with nothing but Ollama installed.
"""

import os
from typing import Optional

from qdrant_client import QdrantClient as _QdrantClient

from backend.config import QDRANT_HOST, QDRANT_MODE, QDRANT_PATH, QDRANT_PORT

_client: Optional[_QdrantClient] = None


class QdrantUnavailable(RuntimeError):
    """The configured Qdrant backend could not be opened."""


def get_qdrant_client() -> _QdrantClient:
    global _client
    if _client is not None:
        return _client

    try:
        if QDRANT_MODE == "memory":
            _client = _QdrantClient(location=":memory:")
        elif QDRANT_MODE == "server":
            _client = _QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        else:
            # Embedded on-disk. Qdrant takes a lock on this directory, so only
            # one process may hold it at a time.
            os.makedirs(QDRANT_PATH, exist_ok=True)
            _client = _QdrantClient(path=QDRANT_PATH)
    except Exception as exc:  # qdrant-client raises a variety of driver errors
        target = (
            f"{QDRANT_HOST}:{QDRANT_PORT}" if QDRANT_MODE == "server" else QDRANT_PATH
        )
        raise QdrantUnavailable(
            f"Could not open Qdrant in {QDRANT_MODE!r} mode ({target}): {exc}"
        ) from exc

    return _client


def reset_client() -> None:
    """Drop the cached client. Used between tests and after a config change."""
    global _client
    if _client is not None:
        try:
            _client.close()
        except Exception:
            pass
    _client = None


def ping() -> dict:
    """Report whether the vector store is reachable and how many points it holds."""
    from backend.config import QDRANT_COLLECTION

    try:
        client = get_qdrant_client()
        exists = client.collection_exists(QDRANT_COLLECTION)
        points = (
            client.count(QDRANT_COLLECTION, exact=True).count if exists else 0
        )
        return {
            "reachable": True,
            "mode": QDRANT_MODE,
            "collection": QDRANT_COLLECTION,
            "collection_exists": exists,
            "points": points,
        }
    except Exception as exc:
        return {"reachable": False, "mode": QDRANT_MODE, "detail": str(exc)}
