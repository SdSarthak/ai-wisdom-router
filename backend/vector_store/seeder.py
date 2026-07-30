from qdrant_client.models import Distance, VectorParams, PointStruct
from backend.config import QDRANT_COLLECTION, EMBEDDING_DIM
from backend.vector_store.qdrant_client import get_qdrant_client
from backend.vector_store.embedder import embed_text
from backend.mentors.seed_data import SEED_DATA


def seed_mentor_knowledge():
    client = get_qdrant_client()

    try:
        client.get_collection(QDRANT_COLLECTION)
        print(f"[seeder] Collection '{QDRANT_COLLECTION}' already exists — skipping seed.")
        return
    except Exception:
        pass

    client.create_collection(
        collection_name=QDRANT_COLLECTION,
        vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
    )
    print(f"[seeder] Created collection '{QDRANT_COLLECTION}'.")

    point_id = 1
    for mentor_id, quotes in SEED_DATA.items():
        points = []
        for text, source, topics in quotes:
            vector = embed_text(text)
            points.append(
                PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={
                        "mentor_id": mentor_id,
                        "text": text,
                        "source": source,
                        "topics": topics,
                        "type": "quote",
                    },
                )
            )
            point_id += 1

        client.upsert(collection_name=QDRANT_COLLECTION, points=points)
        print(f"[seeder] Seeded {len(points)} quotes for {mentor_id}.")

    print(f"[seeder] Done. Total points: {point_id - 1}")


if __name__ == "__main__":
    seed_mentor_knowledge()
