from langchain_community.embeddings import OllamaEmbeddings
from backend.config import EMBEDDING_MODEL, OLLAMA_BASE_URL

_embedder = None


def get_embedder() -> OllamaEmbeddings:
    global _embedder
    if _embedder is None:
        _embedder = OllamaEmbeddings(
            model=EMBEDDING_MODEL,
            base_url=OLLAMA_BASE_URL,
        )
    return _embedder


def embed_text(text: str) -> list:
    return get_embedder().embed_query(text)


def embed_texts(texts: list) -> list:
    return get_embedder().embed_documents(texts)
