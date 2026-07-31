"""Corpus seeding into the vector store."""

import pytest

from backend.config import QDRANT_COLLECTION
from backend.mentors.roster import MENTORS
from backend.mentors.seed_data import SEED_DATA
from backend.vector_store import seeder
from backend.vector_store.qdrant_client import get_qdrant_client


def _total_quotes():
    return sum(len(q) for q in SEED_DATA.values())


def test_seed_data_covers_every_mentor():
    assert set(SEED_DATA) == set(MENTORS)


def test_seed_data_entries_are_well_formed():
    known_topics = set()
    for mentor in MENTORS.values():
        known_topics.update(mentor.domain_weights)

    for mentor_id, quotes in SEED_DATA.items():
        assert quotes, f"{mentor_id} has no quotes"
        for text, source, topics in quotes:
            assert text.strip(), f"empty quote for {mentor_id}"
            assert source.strip(), f"missing source for a {mentor_id} quote"
            assert topics, f"untagged quote for {mentor_id}: {text[:40]}"
            unknown = set(topics) - known_topics
            assert not unknown, f"{mentor_id} quote tagged with unknown topics {unknown}"


def test_seeding_writes_every_quote():
    written = seeder.seed_mentor_knowledge()
    assert written == _total_quotes()
    client = get_qdrant_client()
    assert client.count(QDRANT_COLLECTION, exact=True).count == _total_quotes()


def test_seeding_is_idempotent():
    seeder.seed_mentor_knowledge()
    written_again = seeder.seed_mentor_knowledge()
    assert written_again == 0
    client = get_qdrant_client()
    assert client.count(QDRANT_COLLECTION, exact=True).count == _total_quotes()


def test_force_rebuilds_without_duplicating():
    seeder.seed_mentor_knowledge()
    seeder.seed_mentor_knowledge(force=True)
    client = get_qdrant_client()
    assert client.count(QDRANT_COLLECTION, exact=True).count == _total_quotes()


def test_points_carry_their_mentor_and_source():
    seeder.seed_mentor_knowledge()
    client = get_qdrant_client()
    points, _ = client.scroll(QDRANT_COLLECTION, limit=5, with_payload=True)
    for point in points:
        assert point.payload["mentor_id"] in MENTORS
        assert point.payload["text"]
        assert point.payload["type"] == "quote"


def test_dimension_change_triggers_a_rebuild(monkeypatch):
    """Switching embedding model must not leave a mismatched collection behind."""
    seeder.seed_mentor_knowledge()
    client = get_qdrant_client()

    monkeypatch.setattr(seeder, "EMBEDDING_DIM", 8)
    seeder.ensure_collection(client)
    info = client.get_collection(QDRANT_COLLECTION)
    assert info.config.params.vectors.size == 8
    assert client.count(QDRANT_COLLECTION, exact=True).count == 0


def test_adding_a_mentor_does_not_relabel_existing_points(monkeypatch):
    """Point ids must not depend on a mentor's position in SEED_DATA.

    With running positional ids, inserting a mentor shifted every later mentor's
    id range, so the newcomer's points overwrote an existing mentor's — leaving
    quotes attributed to the wrong person.
    """
    from backend.mentors.roster import Mentor

    seeder.seed_mentor_knowledge()
    client = get_qdrant_client()
    before = {mid: seeder._mentor_point_count(client, mid) for mid in SEED_DATA}

    monkeypatch.setitem(
        seeder.MENTORS,
        "new_voice",
        Mentor(
            id="new_voice",
            display_name="New Voice",
            domains=["career"],
            domain_weights={"career": 1.0},
            persona_prompt="p",
            color="#123456",
        ),
    )
    reordered = {"new_voice": [("a brand new quote", "somewhere", ["career"])]}
    reordered.update(SEED_DATA)  # the newcomer sits ahead of everyone else
    monkeypatch.setattr(seeder, "SEED_DATA", reordered)

    seeder.seed_mentor_knowledge()

    after = {mid: seeder._mentor_point_count(client, mid) for mid in SEED_DATA}
    assert after == before
    assert seeder._mentor_point_count(client, "new_voice") == 1
    assert client.count(QDRANT_COLLECTION, exact=True).count == _total_quotes() + 1


def test_shrinking_a_corpus_drops_the_stale_points(monkeypatch):
    """Deleted quotes must stop being retrievable, not linger as orphans."""
    seeder.seed_mentor_knowledge()
    client = get_qdrant_client()

    trimmed = dict(SEED_DATA)
    trimmed["paul_graham"] = list(SEED_DATA["paul_graham"])[:2]
    monkeypatch.setattr(seeder, "SEED_DATA", trimmed)

    seeder.seed_mentor_knowledge()
    assert seeder._mentor_point_count(client, "paul_graham") == 2


def test_point_ids_are_stable_across_runs():
    assert seeder._point_id("paul_graham", 0) == seeder._point_id("paul_graham", 0)
    assert seeder._point_id("paul_graham", 0) != seeder._point_id("paul_graham", 1)
    assert seeder._point_id("paul_graham", 0) != seeder._point_id("naval_ravikant", 0)


def test_unknown_mentor_in_seed_data_is_skipped(monkeypatch):
    expected = _total_quotes()  # baseline before the bogus entry is injected
    monkeypatch.setitem(
        seeder.SEED_DATA, "someone_not_on_the_roster", [("a quote", "src", ["career"])]
    )
    written = seeder.seed_mentor_knowledge()
    assert written == expected
