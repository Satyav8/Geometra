from typing import List, Tuple

import chromadb
from chromadb.config import Settings

from config import (
    CHROMA_PERSIST_DIR,
    CHROMA_COLLECTION,
    TOP_K_CHUNKS,
    MIN_SIMILARITY_SCORE,
    LOW_CONFIDENCE_THRESHOLD,
)
from models import SourceChunk
from rag.embedder import embed_text

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


def retrieve(query: str) -> Tuple[List[SourceChunk], str]:
    collection = _get_collection()
    query_embedding = embed_text(query)

    results = collection.query(
        query_embeddings=[query_embedding], n_results=TOP_K_CHUNKS
    )

    ids = results["ids"][0] if results["ids"] else []
    documents = results["documents"][0] if results["documents"] else []
    metadatas = results["metadatas"][0] if results["metadatas"] else []
    distances = results["distances"][0] if results["distances"] else []

    chunks: List[SourceChunk] = []
    for chunk_id, text, meta, distance in zip(ids, documents, metadatas, distances):
        similarity_score = 1.0 - distance
        chunks.append(
            SourceChunk(
                chunk_id=meta.get("chunk_id", chunk_id),
                section=meta.get("section_name", ""),
                text=text,
                similarity_score=round(similarity_score, 4),
            )
        )

    top1_score = chunks[0].similarity_score if chunks else 0.0

    if top1_score >= LOW_CONFIDENCE_THRESHOLD:
        confidence_level = "high"
    elif top1_score >= MIN_SIMILARITY_SCORE:
        confidence_level = "low"
    else:
        confidence_level = "unknown"

    return chunks, confidence_level
