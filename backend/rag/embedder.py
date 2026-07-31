from typing import List

from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL

_model = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def embed_text(text: str) -> List[float]:
    return embed_batch([text])[0]


def embed_batch(texts: List[str]) -> List[List[float]]:
    model = _get_model()
    vectors = model.encode(texts, normalize_embeddings=True)
    return [v.tolist() for v in vectors]
