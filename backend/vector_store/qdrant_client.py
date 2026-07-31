"""Qdrant connection management.

Three modes are supported (see QDRANT_MODE in config):

  server  — HTTP connection to a running Qdrant instance
  local   — embedded on-disk store; no server to run, survives restarts
  memory  — embedded, in-process, wiped on restart

`local` is the default so a fresh clone works with nothing but Ollama installed.
"""

import logging
import os
import threading
from typing import Optional

from qdrant_client import QdrantClient as _QdrantClient

from backend.config import QDRANT_HOST, QDRANT_MODE, QDRANT_PATH, QDRANT_PORT

logger = logging.getLogger(__name__)

_client: Optional[_QdrantClient] = None
# Retrieval runs on worker threads, so several turns can reach a cold cache at
# once. In `local` mode the loser of that race would fail on the directory lock
# that the winner already holds — or, worse, silently orphan a second client
# still holding open sqlite handles.
_client_lock = threading.Lock()


class QdrantUnavailable(RuntimeError):
    """The configured Qdrant backend could not be opened."""


def _open_client() -> _QdrantClient:
    try:
        if QDRANT_MODE == "memory":
            return _QdrantClient(location=":memory:")
        if QDRANT_MODE == "server":
            return _QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        # Embedded on-disk. Qdrant takes a lock on this directory, so only
        # one process may hold it at a time.
        os.makedirs(QDRANT_PATH, exist_ok=True)
        return _QdrantClient(path=QDRANT_PATH)
    except Exception as exc:  # qdrant-client raises a variety of driver errors
        target = (
            f"{QDRANT_HOST}:{QDRANT_PORT}" if QDRANT_MODE == "server" else QDRANT_PATH
        )
        raise QdrantUnavailable(
            f"Could not open Qdrant in {QDRANT_MODE!r} mode ({target}): {exc}"
        ) from exc


def get_qdrant_client() -> _QdrantClient:
    global _client
    # Fast path: an already-open client needs no lock.
    client = _client
    if client is not None:
        return client

    with _client_lock:
        if _client is None:
            _client = _open_client()
        return _client


def reset_client() -> None:
    """Drop the cached client. Used between tests and after a config change."""
    global _client
    with _client_lock:
        client, _client = _client, None

    if client is not None:
        try:
            client.close()
        except Exception as exc:
            # Already closed, or the on-disk lock was released underneath us —
            # either way the goal is just to drop the reference.
            logger.debug("Ignoring error while closing Qdrant client: %s", exc)


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
