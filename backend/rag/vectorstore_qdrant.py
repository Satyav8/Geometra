from typing import List

import requests

from config import EMBEDDING_DIM, QDRANT_API_KEY, QDRANT_COLLECTION, QDRANT_URL

HEADERS = {"api-key": QDRANT_API_KEY, "Content-Type": "application/json"}


def _collection_url(suffix: str = "") -> str:
    return f"{QDRANT_URL}/collections/{QDRANT_COLLECTION}{suffix}"


def recreate_and_store(
    chunk_ids: List[str], embeddings: List[List[float]], documents: List[str], metadatas: List[dict]
) -> None:
    requests.delete(_collection_url(), headers=HEADERS, timeout=15)
    resp = requests.put(
        _collection_url(),
        headers=HEADERS,
        json={"vectors": {"size": EMBEDDING_DIM, "distance": "Cosine"}},
        timeout=15,
    )
    resp.raise_for_status()

    points = [
        {
            "id": i,
            "vector": embedding,
            "payload": {**meta, "chunk_id": meta.get("chunk_id", chunk_id), "text": text},
        }
        for i, (chunk_id, embedding, text, meta) in enumerate(
            zip(chunk_ids, embeddings, documents, metadatas)
        )
    ]
    resp = requests.put(_collection_url("/points"), headers=HEADERS, json={"points": points}, timeout=30)
    resp.raise_for_status()


def query(query_embedding: List[float], top_k: int) -> List[dict]:
    resp = requests.post(
        _collection_url("/points/search"),
        headers=HEADERS,
        json={"vector": query_embedding, "limit": top_k, "with_payload": True},
        timeout=15,
    )
    resp.raise_for_status()
    results = resp.json()["result"]

    # Qdrant's "score" for Cosine distance is the cosine similarity itself (higher =
    # better), same orientation and scale as Chroma's 1 - distance, so no conversion.
    return [
        {
            "chunk_id": r["payload"].get("chunk_id", str(r["id"])),
            "section": r["payload"].get("section_name", ""),
            "text": r["payload"].get("text", ""),
            "similarity_score": round(r["score"], 4),
        }
        for r in results
    ]


def count() -> int:
    resp = requests.get(_collection_url(), headers=HEADERS, timeout=10)
    resp.raise_for_status()
    return resp.json()["result"]["points_count"]


def check_health() -> bool:
    try:
        resp = requests.get(_collection_url(), headers=HEADERS, timeout=10)
        return resp.ok
    except Exception:
        return False
