import time
import uuid
from typing import List

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request

from config import (
    CHECK_IN_MESSAGE,
    ESCALATION_TURN_THRESHOLD,
    SUPPORT_EMAIL,
    TICKET_RAISED_MESSAGE,
)
from database import (
    ensure_session,
    get_session_messages,
    get_session_state,
    increment_session_turns,
    set_session_state,
    write_evaluation_logs,
    write_message,
    write_unknown_question,
)
from evaluation.metrics import evaluate_core_metrics, sqlite_log_integrity
from integrations.resend_client import send_ticket_email
from integrations.supabase_client import write_escalated_question
from llm.two_pass import process_turn
from models import ChatRequest, ChatResponse, SourceChunk
from rag.relevance import compute_criticality
from rag.spelling import correct_query
from rate_limiter import limiter

router = APIRouter()


def _validate_request(chat_request: ChatRequest) -> None:
    try:
        uuid.UUID(chat_request.session_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="session_id must be a valid UUID")
    if not (1 <= len(chat_request.query) <= 500):
        raise HTTPException(status_code=400, detail="query must be between 1 and 500 characters")
    if chat_request.turn_number < 1:
        raise HTTPException(status_code=400, detail="turn_number must be >= 1")


@router.post("/chat", response_model=ChatResponse)
# Two independent windows, not one: the per-minute cap stops a scripted burst or a
# runaway frontend retry loop without interrupting a genuine back-and-forth conversation;
# the per-hour cap catches sustained low-and-slow abuse that stays under the per-minute
# rate. Keyed by client IP (see rate_limiter.py) - this is cost/abuse control on the one
# endpoint that spends real OpenAI/Qdrant money per call, not a fairness mechanism.
@limiter.limit("20/minute")
@limiter.limit("200/hour")
def chat(request: Request, chat_request: ChatRequest, background_tasks: BackgroundTasks) -> ChatResponse:
    start_time = time.time()
    _validate_request(chat_request)

    ensure_session(chat_request.session_id)

    # Typo-corrected text drives retrieval/relevance/the LLM prompt inside process_turn();
    # the customer's original raw text (chat_request.query) is still what gets stored/escalated.
    query = correct_query(chat_request.query)

    session_state = get_session_state(chat_request.session_id)
    awaiting = session_state["awaiting"]

    history = []
    for row in get_session_messages(chat_request.session_id):
        history.append(("customer", row["query"]))
        history.append(("sam", row["response"]))

    result = process_turn(
        query, chat_request.query, history, awaiting,
        existing_pending_query=session_state["pending_ticket_query"],
        existing_pending_similarity=session_state["pending_ticket_similarity"],
    )

    response = result.response
    chunks = result.chunks
    confidence_level = result.confidence_level
    sources: List[SourceChunk] = chunks if result.show_sources else []
    is_unknown_question = False
    ticket_number = None

    # A confirmed ticket only exists on the turn the customer says "yes" - the question
    # being escalated is whatever was held from the turn that actually triggered the
    # offer (see llm/two_pass.py's pending_query), not the "yes" itself.
    if result.raise_ticket_now:
        ticket_query = session_state["pending_ticket_query"] or chat_request.query
        ticket_similarity = session_state["pending_ticket_similarity"] or 0.0
        unknown_question_id = write_unknown_question(chat_request.session_id, ticket_query, ticket_similarity)
        ticket_number = f"GEO-{unknown_question_id:03d}"
        response = TICKET_RAISED_MESSAGE.format(ticket_number=ticket_number)
        write_escalated_question(
            question=ticket_query,
            criticality=compute_criticality(ticket_similarity),
            similarity_score=ticket_similarity,
            session_id=chat_request.session_id,
            turn_number=chat_request.turn_number,
        )
        is_unknown_question = True

    if result.update_pending:
        set_session_state(
            chat_request.session_id, result.new_awaiting,
            pending_ticket_query=result.pending_query,
            pending_ticket_similarity=result.pending_similarity,
        )
    else:
        set_session_state(chat_request.session_id, result.new_awaiting)

    response_latency_ms = int((time.time() - start_time) * 1000)
    # Retrieval only ran on the Pass 1/2 branch - confidence_level == "unknown" there means
    # the fast-path scope gate rejected it as off-topic. Every deterministic short-circuit
    # branch reports "high" (see llm/two_pass.py's _short_circuit), which is treated as
    # relevant here too - none of them are the off-topic rejection case.
    is_relevant = confidence_level != "unknown"
    evaluation = evaluate_core_metrics(
        query=query,
        response=response,
        chunks=chunks,
        confidence_level=confidence_level,
        is_relevant=is_relevant,
        session_id=chat_request.session_id,
        turn_number=chat_request.turn_number,
        response_latency_ms=response_latency_ms,
        input_tokens=result.input_tokens,
    )

    # The turn-6 nudge only makes sense after a genuine substantive answer - not right
    # after a greeting, a ticket offer, a refusal, or a clarifying question of our own.
    is_check_in_turn = chat_request.turn_number == ESCALATION_TURN_THRESHOLD and not result.skip_check_in
    if is_check_in_turn:
        response = response + CHECK_IN_MESSAGE

    support_email = SUPPORT_EMAIL if (is_unknown_question or is_check_in_turn) else None

    message_id = write_message(
        session_id=chat_request.session_id,
        turn_number=chat_request.turn_number,
        query=chat_request.query,
        response=response,
        retrieved_chunk_ids=[c.chunk_id for c in chunks],
        similarity_scores=[c.similarity_score for c in chunks],
        confidence_level=confidence_level,
        is_unknown_question=is_unknown_question,
        response_latency_ms=response_latency_ms,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
    )

    # Everything below this point is internal bookkeeping the customer never waits on:
    # the ticket email, the log-integrity metric, the evaluation-log write and the turn
    # counter. On Supabase each of these is its own network round trip, and measured
    # against production they were adding seconds to the reply the customer is sitting
    # there waiting for. Handed to a background task so the response goes out first.
    #
    # write_message() above deliberately stays synchronous: the next turn's Pass 1 reads
    # this session's transcript back via get_session_messages(), so deferring that one
    # write could cost the following turn its conversation history.
    def _log_and_notify() -> None:
        if is_unknown_question:
            # Sent after write_message so the transcript includes this turn's own record.
            send_ticket_email(ticket_number, chat_request.session_id, get_session_messages(chat_request.session_id))

        integrity_result = sqlite_log_integrity(
            message_id,
            {
                "session_id": chat_request.session_id,
                "turn_number": chat_request.turn_number,
                "query": chat_request.query,
                "response": response,
                "confidence_level": confidence_level,
                "is_unknown_question": is_unknown_question,
                "response_latency_ms": response_latency_ms,
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
            },
        )
        write_evaluation_logs(message_id, evaluation + [integrity_result])
        increment_session_turns(chat_request.session_id)

    background_tasks.add_task(_log_and_notify)

    return ChatResponse(
        session_id=chat_request.session_id,
        turn_number=chat_request.turn_number,
        response=response,
        sources=sources,
        confidence_level=confidence_level,
        is_unknown_question=is_unknown_question,
        evaluation=evaluation,
        response_latency_ms=response_latency_ms,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        support_email=support_email,
    )
