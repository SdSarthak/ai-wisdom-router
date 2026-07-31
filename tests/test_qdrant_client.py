"""Connection management for the vector store."""

import threading

import pytest

from backend.vector_store import qdrant_client as qc


def test_the_client_is_cached():
    assert qc.get_qdrant_client() is qc.get_qdrant_client()


def test_concurrent_cold_starts_open_exactly_one_client(monkeypatch):
    """Retrieval runs on worker threads, so a cold cache is reached in parallel.

    In `local` mode a second client would collide with the directory lock the
    first one holds, so creation has to happen once and once only.
    """
    qc.reset_client()
    opened = []
    barrier = threading.Barrier(8)
    real_open = qc._open_client

    def _counting_open():
        opened.append(1)
        return real_open()

    monkeypatch.setattr(qc, "_open_client", _counting_open)

    results = []

    def _worker():
        barrier.wait()
        results.append(qc.get_qdrant_client())

    threads = [threading.Thread(target=_worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(opened) == 1
    assert len({id(c) for c in results}) == 1


def test_reset_releases_the_client_before_dropping_it():
    closed = {"n": 0}

    class _Fake:
        def close(self):
            closed["n"] += 1

    qc.reset_client()
    qc._client = _Fake()
    qc.reset_client()
    assert closed["n"] == 1
    assert qc._client is None


def test_reset_still_drops_a_client_that_fails_to_close():
    """A half-dead handle must not pin the cache forever."""

    class _Stubborn:
        def close(self):
            raise RuntimeError("already gone")

    qc.reset_client()
    qc._client = _Stubborn()
    qc.reset_client()
    assert qc._client is None


def test_an_unopenable_backend_raises_a_named_error(monkeypatch):
    qc.reset_client()
    monkeypatch.setattr(qc, "QDRANT_MODE", "server")
    monkeypatch.setattr(qc, "QDRANT_HOST", "no-such-host.invalid")

    def _boom(*args, **kwargs):
        raise OSError("name or service not known")

    monkeypatch.setattr(qc, "_QdrantClient", _boom)
    with pytest.raises(qc.QdrantUnavailable, match="server"):
        qc.get_qdrant_client()


def test_ping_reports_the_failure_instead_of_raising(monkeypatch):
    qc.reset_client()

    def _boom():
        raise qc.QdrantUnavailable("nothing at that path")

    monkeypatch.setattr(qc, "get_qdrant_client", _boom)
    result = qc.ping()
    assert result["reachable"] is False
    assert "nothing at that path" in result["detail"]


def test_ping_reports_an_empty_collection_as_zero_points():
    qc.reset_client()
    result = qc.ping()
    assert result["reachable"] is True
    assert result["collection_exists"] is False
    assert result["points"] == 0
