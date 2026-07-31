import time
import uuid
from typing import List

from fastapi import APIRouter, HTTPException

from config import (
    CHECK_IN_MESSAGE,
    ESCALATION_TURN_THRESHOLD,
    FALLBACK_MESSAGE,
    GRATITUDE_MESSAGE,
    OUT_OF_SCOPE_MESSAGE,
    SUPPORT_EMAIL,
)
from database import (
    ensure_session,
    increment_session_turns,
    write_evaluation_logs,
    write_message,
    write_unknown_question,
)
from evaluation.metrics import evaluate_core_metrics, sqlite_log_integrity
from llm.client import call_llm
from llm.guardrails import (
    check_fallback_leakage,
    check_numerical_hallucination,
    check_response_length,
    check_uncertainty_language,
)
from llm.prompts import SYSTEM_PROMPT, build_user_message
from integrations.supabase_client import write_escalated_question
from models import ChatRequest, ChatResponse, SourceChunk
from rag.relevance import compute_criticality, is_gratitude, is_query_relevant
from rag.retriever import retrieve
from rag.spelling import correct_query

router = APIRouter()


def _validate_request(request: ChatRequest) -> None:
    try:
        uuid.UUID(request.session_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="session_id must be a valid UUID")
    if not (1 <= len(request.query) <= 500):
        raise HTTPException(status_code=400, detail="query must be between 1 and 500 characters")
    if request.turn_number < 1:
        raise HTTPException(status_code=400, detail="turn_number must be >= 1")


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    start_time = time.time()
    _validate_request(request)

    ensure_session(request.session_id)

    # Typo-corrected text drives retrieval/relevance/the LLM prompt (a misspelled word
    # can otherwise tank embedding similarity or miss a keyword match); the customer's
    # original raw text is still what gets stored/escalated, so records stay verbatim.
    query = correct_query(request.query)

    sources: List[SourceChunk] = []
    input_tokens = 0
    output_tokens = 0
    is_unknown_question = False

    if is_gratitude(query):
        # Courtesy close — no need to hit retrieval or the LLM for this.
        chunks = []
        confidence_level = "high"
        is_relevant = True
        response = GRATITUDE_MESSAGE
    else:
        chunks, confidence_level = retrieve(query)
        is_relevant = is_query_relevant(query)

        if confidence_level == "unknown":
            if is_relevant:
                # On-topic, but the knowledge base doesn't cover it — flag for the team.
                response = FALLBACK_MESSAGE
                is_unknown_question = True
            else:
                # Not about Geometra at all — reject outright, nothing for the team to review.
                response = OUT_OF_SCOPE_MESSAGE
        else:
            sources = chunks
            user_message = build_user_message(query, chunks, confidence_level)
            response, input_tokens, output_tokens = call_llm(SYSTEM_PROMPT, user_message)

            response, _ = check_fallback_leakage(response)
            response, _ = check_uncertainty_language(response)
            response, _ = check_response_length(response)
            response, _ = check_numerical_hallucination(response, chunks)

    # The LLM can independently land on the exact fallback sentence (Rule 2) even at
    # "low" confidence, not just via the hardcoded "unknown" branch above. Either way,
    # the response promises the team was notified — so treat it as unknown consistently.
    if response.strip() == FALLBACK_MESSAGE:
        is_unknown_question = True

    response_latency_ms = int((time.time() - start_time) * 1000)

    evaluation = evaluate_core_metrics(
        query=query,
        response=response,
        chunks=chunks,
        confidence_level=confidence_level,
        is_relevant=is_relevant,
        session_id=request.session_id,
        turn_number=request.turn_number,
        response_latency_ms=response_latency_ms,
        input_tokens=input_tokens,
    )

    # Metrics above evaluate the substantive answer only; this check-in nudge is UX,
    # appended after so it doesn't skew hallucination/citation/etc. scoring.
    is_check_in_turn = request.turn_number == ESCALATION_TURN_THRESHOLD
    if is_check_in_turn:
        response = response + CHECK_IN_MESSAGE

    # Frontend renders this as a clickable "email support" button — offered whenever
    # the fallback text is shown, or on the 6-turn check-in. Not shown for the hard
    # out-of-scope rejection (that's off-topic chatter, not a knowledge gap).
    support_email = SUPPORT_EMAIL if (is_unknown_question or is_check_in_turn) else None

    message_id = write_message(
        session_id=request.session_id,
        turn_number=request.turn_number,
        query=request.query,
        response=response,
        retrieved_chunk_ids=[c.chunk_id for c in chunks],
        similarity_scores=[c.similarity_score for c in chunks],
        confidence_level=confidence_level,
        is_unknown_question=is_unknown_question,
        response_latency_ms=response_latency_ms,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )

    if is_unknown_question:
        top1 = chunks[0].similarity_score if chunks else 0.0
        write_unknown_question(request.session_id, request.query, top1)
        write_escalated_question(
            question=request.query,
            criticality=compute_criticality(top1),
            similarity_score=top1,
            session_id=request.session_id,
            turn_number=request.turn_number,
        )

    integrity_result = sqlite_log_integrity(
        message_id,
        {
            "session_id": request.session_id,
            "turn_number": request.turn_number,
            "query": request.query,
            "response": response,
            "confidence_level": confidence_level,
            "is_unknown_question": is_unknown_question,
            "response_latency_ms": response_latency_ms,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        },
    )
    evaluation.append(integrity_result)

    write_evaluation_logs(message_id, evaluation)
    increment_session_turns(request.session_id)

    return ChatResponse(
        session_id=request.session_id,
        turn_number=request.turn_number,
        response=response,
        sources=sources,
        confidence_level=confidence_level,
        is_unknown_question=is_unknown_question,
        evaluation=evaluation,
        response_latency_ms=response_latency_ms,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        support_email=support_email,
    )
