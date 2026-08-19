"""Isolated Qdrant collection for scraped geometra.in website content (vision, positioning,
onboarding), kept SEPARATE from the production "geometra_faq" collection so this test-only
knowledge never reaches the real app until explicitly merged in. Mirrors
rag/vectorstore_qdrant.py's API but parameterized to a different collection name -
deliberately not touching rag/vectorstore_qdrant.py or config.QDRANT_COLLECTION, since
those are shared with routers/chat.py.
"""
import hashlib
import uuid

import requests

from config import EMBEDDING_DIM, QDRANT_API_KEY, QDRANT_URL

WEBSITE_COLLECTION = "geometra_website"
HEADERS = {"api-key": QDRANT_API_KEY, "Content-Type": "application/json"}


def _url(suffix: str = "") -> str:
    return f"{QDRANT_URL}/collections/{WEBSITE_COLLECTION}{suffix}"


def _collection_exists() -> bool:
    resp = requests.get(_url(), headers=HEADERS, timeout=10)
    return resp.ok


def _ensure_collection() -> None:
    if _collection_exists():
        return
    resp = requests.put(
        _url(),
        headers=HEADERS,
        json={"vectors": {"size": EMBEDDING_DIM, "distance": "Cosine"}},
        timeout=15,
    )
    resp.raise_for_status()


def stable_chunk_id(section_name: str, text: str) -> str:
    digest = hashlib.md5(f"{section_name}::{text}".encode("utf-8")).digest()
    return str(uuid.UUID(bytes=digest))


def upsert_chunks(chunk_ids, embeddings, documents, metadatas) -> None:
    if not chunk_ids:
        return
    _ensure_collection()
    points = [
        {"id": cid, "vector": emb, "payload": {**meta, "chunk_id": cid, "text": text}}
        for cid, emb, text, meta in zip(chunk_ids, embeddings, documents, metadatas)
    ]
    for i in range(0, len(points), 25):
        batch = points[i : i + 25]
        resp = requests.put(_url("/points"), headers=HEADERS, json={"points": batch}, timeout=30)
        resp.raise_for_status()


def query(query_embedding, top_k: int):
    if not _collection_exists():
        return []
    resp = requests.post(
        _url("/points/search"),
        headers=HEADERS,
        json={"vector": query_embedding, "limit": top_k, "with_payload": True},
        timeout=15,
    )
    resp.raise_for_status()
    results = resp.json()["result"]
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
    if not _collection_exists():
        return 0
    resp = requests.get(_url(), headers=HEADERS, timeout=10)
    resp.raise_for_status()
    return resp.json()["result"]["points_count"]
