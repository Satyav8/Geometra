from typing import List

from config import EMBEDDING_BACKEND, EMBEDDING_MODEL, OPENAI_API_KEY, OPENAI_EMBEDDING_MODEL

_local_model = None


def _get_local_model():
    global _local_model
    if _local_model is None:
        # fastembed (ONNX-based) instead of sentence-transformers (PyTorch-based) —
        # same model, same 384-dim normalized output, but a fraction of the memory
        # footprint. Needed to fit Render's free-tier 512MB RAM limit.
        from fastembed import TextEmbedding

        _local_model = TextEmbedding(model_name=EMBEDDING_MODEL)
    return _local_model


def embed_text(text: str) -> List[float]:
    return embed_batch([text])[0]


def embed_batch(texts: List[str]) -> List[List[float]]:
    if EMBEDDING_BACKEND == "openai":
        return _embed_batch_openai(texts)
    return _embed_batch_local(texts)


def _embed_batch_local(texts: List[str]) -> List[List[float]]:
    model = _get_local_model()
    vectors = model.embed(texts)
    return [v.tolist() for v in vectors]


def _embed_batch_openai(texts: List[str]) -> List[List[float]]:
    from openai import OpenAI

    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.embeddings.create(model=OPENAI_EMBEDDING_MODEL, input=texts)
    return [d.embedding for d in response.data]
