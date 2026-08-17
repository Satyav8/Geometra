"""Thin dispatcher: routes every vector DB call to either the local ChromaDB
implementation (default, local dev/tests) or the Qdrant Cloud implementation
(production, set VECTOR_DB_BACKEND=qdrant). Ingestor/retriever import from here,
never directly from vectorstore_chroma / vectorstore_qdrant."""

from config import VECTOR_DB_BACKEND

if VECTOR_DB_BACKEND == "qdrant":
    from rag.vectorstore_qdrant import (
        check_health,
        count,
        delete_chunks,
        get_all_ids,
        query,
        recreate_and_store,
        upsert_chunks,
    )
else:
    from rag.vectorstore_chroma import (
        check_health,
        count,
        delete_chunks,
        get_all_ids,
        query,
        recreate_and_store,
        upsert_chunks,
    )

__all__ = [
    "check_health",
    "count",
    "delete_chunks",
    "get_all_ids",
    "query",
    "recreate_and_store",
    "upsert_chunks",
]
