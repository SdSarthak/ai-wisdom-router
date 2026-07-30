import numpy as np
from typing import List, Dict
from backend.config import TOPIC_SIMILARITY_THRESHOLD
from backend.vector_store.embedder import embed_text, embed_texts

TOPIC_ANCHORS: Dict[str, str] = {
    "startup": "building a startup founding a company product market fit early stage venture",
    "investing": "investing money stock market portfolio allocation returns financial assets",
    "discipline": "hard work mental toughness pushing limits consistency self control habits",
    "career": "career growth job professional development skills work promotion",
    "wealth": "building wealth financial independence passive income net worth money",
    "learning": "learning reading acquiring knowledge mental models education studying",
    "leadership": "leading a team management organizational culture decision making vision",
    "relationships": "relationships friendships family social dynamics communication love trust",
    "philosophy": "meaning of life principles stoicism values existence purpose happiness",
    "health": "fitness diet exercise mental health sleep recovery physical performance",
}

_anchor_vectors: Dict[str, np.ndarray] = {}


def precompute_topic_anchors():
    global _anchor_vectors
    texts = list(TOPIC_ANCHORS.values())
    topics = list(TOPIC_ANCHORS.keys())
    embeddings = embed_texts(texts)
    for topic, vec in zip(topics, embeddings):
        _anchor_vectors[topic] = np.array(vec, dtype=np.float32)
    print(f"[intent] Precomputed {len(_anchor_vectors)} topic anchor vectors.")


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def detect_topics(message: str) -> List[str]:
    if not _anchor_vectors:
        precompute_topic_anchors()

    msg_vec = np.array(embed_text(message), dtype=np.float32)
    scores = {
        topic: _cosine_sim(msg_vec, anchor_vec)
        for topic, anchor_vec in _anchor_vectors.items()
    }
    detected = [t for t, s in scores.items() if s >= TOPIC_SIMILARITY_THRESHOLD]

    # Always return at least the top topic
    if not detected:
        detected = [max(scores, key=scores.get)]

    return detected
