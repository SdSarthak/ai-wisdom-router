"""Load the mentor quote corpus into Qdrant.

Run directly to (re)build the collection:

    python -m backend.vector_store.seeder          # seed if empty
    python -m backend.vector_store.seeder --force  # drop and rebuild
"""

import logging
import uuid
from typing import List

from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    MatchValue,
    PointStruct,
    VectorParams,
)

from backend.config import EMBEDDING_DIM, QDRANT_COLLECTION
from backend.mentors.roster import MENTORS
from backend.mentors.seed_data import SEED_DATA
from backend.vector_store.embedder import embed_texts
from backend.vector_store.qdrant_client import get_qdrant_client

logger = logging.getLogger(__name__)

# Quotes are short; one request per mentor keeps Ollama busy without
# building a payload large enough to trip its request size limits.
EMBED_BATCH_SIZE = 32

# Fixed namespace for deriving point ids. Ids must depend on the mentor and the
# mentor's own quote index only — never on the mentor's position in SEED_DATA,
# which changes the moment a mentor is added, removed or reordered.
_POINT_NAMESPACE = uuid.UUID("8f2a4f7c-0b3d-5e6a-9c11-4d7e3a2b6c58")


def _point_id(mentor_id: str, index: int) -> str:
    return str(uuid.uuid5(_POINT_NAMESPACE, f"{mentor_id}:{index}"))


def _mentor_filter(mentor_id: str) -> Filter:
    return Filter(
        must=[FieldCondition(key="mentor_id", match=MatchValue(value=mentor_id))]
    )


def _batched(items: List, size: int):
    for i in range(0, len(items), size):
        yield items[i : i + size]


def ensure_collection(client, recreate: bool = False) -> None:
    """Create the collection, rebuilding it if the vector size no longer matches."""
    exists = client.collection_exists(QDRANT_COLLECTION)

    if exists and not recreate:
        info = client.get_collection(QDRANT_COLLECTION)
        current_dim = info.config.params.vectors.size
        if current_dim != EMBEDDING_DIM:
            logger.warning(
                "Collection %s holds %d-dim vectors but EMBEDDING_DIM is %d — rebuilding.",
                QDRANT_COLLECTION,
                current_dim,
                EMBEDDING_DIM,
            )
            recreate = True
        else:
            return

    if exists:
        client.delete_collection(QDRANT_COLLECTION)
    client.create_collection(
        collection_name=QDRANT_COLLECTION,
        vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
    )
    logger.info("Created collection %s (dim=%d).", QDRANT_COLLECTION, EMBEDDING_DIM)


def _mentor_point_count(client, mentor_id: str) -> int:
    return client.count(
        collection_name=QDRANT_COLLECTION,
        count_filter=_mentor_filter(mentor_id),
        exact=True,
    ).count


def seed_mentor_knowledge(force: bool = False) -> int:
    """Embed and upsert every quote in SEED_DATA. Returns the number of points written.

    Idempotent: mentors whose quotes are already present are skipped, so adding a
    new mentor to the roster only costs the embeddings for that mentor.
    """
    client = get_qdrant_client()
    ensure_collection(client, recreate=force)

    written = 0

    for mentor_id, quotes in SEED_DATA.items():
        if mentor_id not in MENTORS:
            logger.warning(
                "seed_data has quotes for unknown mentor %r — skipping.", mentor_id
            )
            continue

        # Ids are derived from (mentor_id, index within that mentor's quotes), so
        # they are stable under insertion and reordering of SEED_DATA. Running
        # positional ids used to shift every later mentor's range, which made
        # adding one mentor overwrite another mentor's points with the newcomer's
        # payload — the corpus silently changed attribution.
        ids = [_point_id(mentor_id, i) for i in range(len(quotes))]

        if not force and _mentor_point_count(client, mentor_id) == len(quotes):
            logger.info("%s already seeded (%d quotes) — skipping.", mentor_id, len(quotes))
            continue

        # Clear first: if this mentor's corpus shrank, upserting alone would
        # leave the dropped quotes behind as retrievable orphans.
        client.delete(
            collection_name=QDRANT_COLLECTION,
            points_selector=FilterSelector(filter=_mentor_filter(mentor_id)),
        )

        for id_batch, quote_batch in zip(
            _batched(ids, EMBED_BATCH_SIZE), _batched(quotes, EMBED_BATCH_SIZE)
        ):
            vectors = embed_texts([text for text, _, _ in quote_batch])
            points = [
                PointStruct(
                    id=pid,
                    vector=vector,
                    payload={
                        "mentor_id": mentor_id,
                        "text": text,
                        "source": source,
                        "topics": topics,
                        "type": "quote",
                    },
                )
                for pid, vector, (text, source, topics) in zip(
                    id_batch, vectors, quote_batch
                )
            ]
            client.upsert(collection_name=QDRANT_COLLECTION, points=points)
            written += len(points)

        logger.info("Seeded %d quotes for %s.", len(quotes), mentor_id)

    total = client.count(QDRANT_COLLECTION, exact=True).count
    logger.info("Seeding complete. %d new points, %d total.", written, total)
    return written


if __name__ == "__main__":
    import argparse

    from backend import config

    logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")
    parser = argparse.ArgumentParser(description="Seed mentor knowledge into Qdrant.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="drop the collection and re-embed every quote",
    )
    args = parser.parse_args()
    # Writing a corpus under a bad EMBEDDING_DIM would build a collection the
    # server then refuses to query, so check the environment before spending
    # the embedding calls.
    config.validate()
    seed_mentor_knowledge(force=args.force)
