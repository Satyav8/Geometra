from typing import List

import chromadb
from chromadb.config import Settings

from config import CHROMA_COLLECTION, CHROMA_PERSIST_DIR

_client = None
_collection = None


def _get_collection():
    global _client, _collection
    if _collection is None:
        _client = chromadb.PersistentClient(
            path=CHROMA_PERSIST_DIR, settings=Settings(anonymized_telemetry=False)
        )
        _collection = _client.get_or_create_collection(
            CHROMA_COLLECTION, metadata={"hnsw:space": "cosine"}
        )
    return _collection


def recreate_and_store(
    chunk_ids: List[str], embeddings: List[List[float]], documents: List[str], metadatas: List[dict]
) -> None:
    global _client, _collection
    client = chromadb.PersistentClient(
        path=CHROMA_PERSIST_DIR, settings=Settings(anonymized_telemetry=False)
    )
    try:
        client.delete_collection(CHROMA_COLLECTION)
    except Exception:
        pass
    collection = client.create_collection(CHROMA_COLLECTION, metadata={"hnsw:space": "cosine"})
    collection.add(ids=chunk_ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
    _client, _collection = client, collection


def query(query_embedding: List[float], top_k: int) -> List[dict]:
    collection = _get_collection()
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)

    ids = results["ids"][0] if results["ids"] else []
    documents = results["documents"][0] if results["documents"] else []
    metadatas = results["metadatas"][0] if results["metadatas"] else []
    distances = results["distances"][0] if results["distances"] else []

    return [
        {
            "chunk_id": meta.get("chunk_id", chunk_id),
            "section": meta.get("section_name", ""),
            "text": text,
            "similarity_score": round(1.0 - distance, 4),
        }
        for chunk_id, text, meta, distance in zip(ids, documents, metadatas, distances)
    ]


def count() -> int:
    return _get_collection().count()


def check_health() -> bool:
    try:
        _get_collection()
        return True
    except Exception:
        return False
