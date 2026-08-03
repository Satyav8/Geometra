from typing import List, Tuple

from config import LOW_CONFIDENCE_THRESHOLD, MIN_SIMILARITY_SCORE, TOP_K_CHUNKS
from models import SourceChunk
from rag import vectorstore
from rag.embedder import embed_text


def retrieve(query: str) -> Tuple[List[SourceChunk], str]:
    query_embedding = embed_text(query)
    results = vectorstore.query(query_embedding, TOP_K_CHUNKS)

    chunks = [
        SourceChunk(
            chunk_id=r["chunk_id"],
            section=r["section"],
            text=r["text"],
            similarity_score=r["similarity_score"],
        )
        for r in results
    ]

    top1_score = chunks[0].similarity_score if chunks else 0.0

    if top1_score >= LOW_CONFIDENCE_THRESHOLD:
        confidence_level = "high"
    elif top1_score >= MIN_SIMILARITY_SCORE:
        confidence_level = "low"
    else:
        confidence_level = "unknown"

    return chunks, confidence_level
