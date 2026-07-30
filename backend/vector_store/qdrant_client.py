from qdrant_client import QdrantClient as _QdrantClient
from backend.config import QDRANT_MODE, QDRANT_HOST, QDRANT_PORT

_client = None


def get_qdrant_client() -> _QdrantClient:
    global _client
    if _client is None:
        if QDRANT_MODE == "memory":
            _client = _QdrantClient(":memory:")
        else:
            _client = _QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    return _client
