from typing import List

from fastembed import TextEmbedding

from config import EMBEDDING_MODEL

_model = None


def _get_model() -> TextEmbedding:
    global _model
    if _model is None:
        # fastembed (ONNX-based) instead of sentence-transformers (PyTorch-based) —
        # same model, same 384-dim normalized output, but a fraction of the memory
        # footprint. Needed to fit Render's free-tier 512MB RAM limit.
        _model = TextEmbedding(model_name=EMBEDDING_MODEL)
    return _model


def embed_text(text: str) -> List[float]:
    return embed_batch([text])[0]


def embed_batch(texts: List[str]) -> List[List[float]]:
    model = _get_model()
    vectors = model.embed(texts)
    return [v.tolist() for v in vectors]
