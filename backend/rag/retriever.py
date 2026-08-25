from typing import List, Tuple

import website_kb
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


def retrieve_combined(query_text: str) -> Tuple[List[SourceChunk], str]:
    """Same as retrieve(), but also queries the isolated geometra_website Qdrant
    collection (see website_kb.py) and merges results in, re-sorted by similarity and
    re-scored for confidence. Ported from backend/try_it_yourself.py's retrieve_combined()
    after testing confirmed the website content is a useful addition to the FAQ base."""
    faq_chunks, _ = retrieve(query_text)
    query_embedding = embed_text(query_text)
    try:
        website_results = website_kb.query(query_embedding, top_k=5)
    except Exception as e:
        # The geometra_website Qdrant collection was originally built with a different
        # embedding backend's vector size than what's currently configured (e.g. local
        # 384-dim vs OpenAI's 1536-dim), which raises here as an HTTP 400 from Qdrant. A
        # website-KB lookup failure must never take down the whole chat response - fall
        # back to FAQ-only results, same as send_ticket_email/write_escalated_question's
        # "never raise" convention elsewhere in this codebase.
        print(f"[website_kb] query failed, continuing with FAQ-only results: {e}")
        website_results = []
    website_chunks = [
        SourceChunk(
            chunk_id=r["chunk_id"], section=r["section"], text=r["text"],
            similarity_score=r["similarity_score"],
        )
        for r in website_results
    ]
    combined = sorted(faq_chunks + website_chunks, key=lambda c: c.similarity_score, reverse=True)[:TOP_K_CHUNKS]

    top1 = combined[0].similarity_score if combined else 0.0
    if top1 >= LOW_CONFIDENCE_THRESHOLD:
        confidence = "high"
    elif top1 >= MIN_SIMILARITY_SCORE:
        confidence = "low"
    else:
        confidence = "unknown"
    return combined, confidence
