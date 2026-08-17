from typing import List, Set

import requests

from config import EMBEDDING_DIM, QDRANT_API_KEY, QDRANT_COLLECTION, QDRANT_URL

HEADERS = {"api-key": QDRANT_API_KEY, "Content-Type": "application/json"}
_UPSERT_BATCH_SIZE = 25


def _collection_url(suffix: str = "") -> str:
    return f"{QDRANT_URL}/collections/{QDRANT_COLLECTION}{suffix}"


def _collection_exists() -> bool:
    resp = requests.get(_collection_url(), headers=HEADERS, timeout=10)
    return resp.ok


def _ensure_collection() -> None:
    if _collection_exists():
        return
    resp = requests.put(
        _collection_url(),
        headers=HEADERS,
        json={"vectors": {"size": EMBEDDING_DIM, "distance": "Cosine"}},
        timeout=15,
    )
    resp.raise_for_status()


def _points_payload(chunk_ids: List[str], embeddings: List[List[float]], documents: List[str], metadatas: List[dict]) -> List[dict]:
    # Point id IS the content-derived chunk_id (a UUID string - see ingestor.py) so the
    # same content always lands on the same point, and get_all_ids() can be diffed
    # directly against it without a separate id-mapping lookup.
    return [
        {
            "id": chunk_id,
            "vector": embedding,
            "payload": {**meta, "chunk_id": chunk_id, "text": text},
        }
        for chunk_id, embedding, text, meta in zip(chunk_ids, embeddings, documents, metadatas)
    ]


def recreate_and_store(
    chunk_ids: List[str], embeddings: List[List[float]], documents: List[str], metadatas: List[dict]
) -> None:
    """Full wipe-and-rebuild. Kept for an explicit manual reset; normal ingestion uses
    upsert_chunks/delete_chunks instead so unchanged content isn't re-embedded."""
    requests.delete(_collection_url(), headers=HEADERS, timeout=15)
    resp = requests.put(
        _collection_url(),
        headers=HEADERS,
        json={"vectors": {"size": EMBEDDING_DIM, "distance": "Cosine"}},
        timeout=15,
    )
    resp.raise_for_status()

    points = _points_payload(chunk_ids, embeddings, documents, metadatas)
    for i in range(0, len(points), _UPSERT_BATCH_SIZE):
        batch = points[i : i + _UPSERT_BATCH_SIZE]
        resp = requests.put(_collection_url("/points"), headers=HEADERS, json={"points": batch}, timeout=30)
        resp.raise_for_status()


def upsert_chunks(
    chunk_ids: List[str], embeddings: List[List[float]], documents: List[str], metadatas: List[dict]
) -> None:
    if not chunk_ids:
        return
    _ensure_collection()
    points = _points_payload(chunk_ids, embeddings, documents, metadatas)
    # Sent in smaller batches, not one giant request - a single PUT with a large batch of
    # 1536-dim vectors has been observed to time out on the write, even within a 30s
    # timeout; smaller requests are both more reliable and fail (and retry) more cheaply.
    for i in range(0, len(points), _UPSERT_BATCH_SIZE):
        batch = points[i : i + _UPSERT_BATCH_SIZE]
        resp = requests.put(_collection_url("/points"), headers=HEADERS, json={"points": batch}, timeout=30)
        resp.raise_for_status()


def delete_chunks(chunk_ids: List[str]) -> None:
    if not chunk_ids:
        return
    resp = requests.post(
        _collection_url("/points/delete"), headers=HEADERS, json={"points": chunk_ids}, timeout=30
    )
    resp.raise_for_status()


def get_all_ids() -> Set[str]:
    if not _collection_exists():
        return set()
    ids: Set[str] = set()
    offset = None
    while True:
        body = {"limit": 250, "with_payload": False, "with_vector": False}
        if offset is not None:
            body["offset"] = offset
        resp = requests.post(_collection_url("/points/scroll"), headers=HEADERS, json=body, timeout=15)
        resp.raise_for_status()
        result = resp.json()["result"]
        ids.update(str(p["id"]) for p in result["points"])
        offset = result.get("next_page_offset")
        if offset is None:
            break
    return ids


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
