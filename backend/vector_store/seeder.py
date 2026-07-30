"""Load the mentor quote corpus into Qdrant.

Run directly to (re)build the collection:

    python -m backend.vector_store.seeder          # seed if empty
    python -m backend.vector_store.seeder --force  # drop and rebuild
"""

import logging
from typing import List

from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
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
        count_filter=Filter(
            must=[FieldCondition(key="mentor_id", match=MatchValue(value=mentor_id))]
        ),
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
    point_id = 0

    for mentor_id, quotes in SEED_DATA.items():
        if mentor_id not in MENTORS:
            logger.warning(
                "seed_data has quotes for unknown mentor %r — skipping.", mentor_id
            )
            point_id += len(quotes)
            continue

        # Deterministic ids keep re-seeding an upsert rather than a duplicate.
        ids = list(range(point_id + 1, point_id + 1 + len(quotes)))
        point_id += len(quotes)

        if not force and _mentor_point_count(client, mentor_id) == len(quotes):
            logger.info("%s already seeded (%d quotes) — skipping.", mentor_id, len(quotes))
            continue

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

    logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")
    parser = argparse.ArgumentParser(description="Seed mentor knowledge into Qdrant.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="drop the collection and re-embed every quote",
    )
    args = parser.parse_args()
    seed_mentor_knowledge(force=args.force)
