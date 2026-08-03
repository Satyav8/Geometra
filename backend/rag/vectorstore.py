"""Thin dispatcher: routes every vector DB call to either the local ChromaDB
implementation (default, local dev/tests) or the Qdrant Cloud implementation
(production, set VECTOR_DB_BACKEND=qdrant). Ingestor/retriever import from here,
never directly from vectorstore_chroma / vectorstore_qdrant."""

from config import VECTOR_DB_BACKEND

if VECTOR_DB_BACKEND == "qdrant":
    from rag.vectorstore_qdrant import check_health, count, query, recreate_and_store
else:
    from rag.vectorstore_chroma import check_health, count, query, recreate_and_store

__all__ = ["check_health", "count", "query", "recreate_and_store"]
