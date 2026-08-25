import re
from typing import Dict, List, Optional

from config import (
    CLARIFY_DECLINE_PROMPT_MESSAGE,
    FALLBACK_MESSAGE,
    FILLER_RESPONSE_MESSAGE,
    GRATITUDE_MESSAGE,
    GREETING_MESSAGE,
    MANNEQUIN_EXCLUSION_MESSAGE,
    OUT_OF_SCOPE_MESSAGE,
    SAFETY_REFUSAL_MESSAGE,
    TICKET_DECLINED_MESSAGE,
    TICKET_ESCALATION_MESSAGE,
    TICKET_OFFER_MESSAGE,
    ALLOWED_PRICE_NUMBERS,
    LOW_CONFIDENCE_THRESHOLD,
    MIN_COMPLETENESS_WORDS,
)
from database import count_session_messages, get_message_by_id, get_session_message_flags
from llm.guardrails import BANNED_PHRASES, _extract_numbers
from llm.two_pass import looks_like_clarify_question
from models import EvaluationResult, SourceChunk

CITATION_RE = re.compile(r"\[Source:\s*([^\]]+)\]")

# Two-pass flow messages that carry no factual claims of their own (fixed text, or a
# clarifying/ticket-offer question) - added alongside the original FALLBACK_MESSAGE/
# OUT_OF_SCOPE_MESSAGE check below so metrics like answer_faithfulness don't mis-score a
# ticket offer or clarifying question as an ungrounded hallucination. Purely additive: the
# original two-value check still passes everything it always did.
_FIXED_NON_SUBSTANTIVE_MESSAGES = {
    FALLBACK_MESSAGE, OUT_OF_SCOPE_MESSAGE, TICKET_OFFER_MESSAGE, TICKET_ESCALATION_MESSAGE,
    SAFETY_REFUSAL_MESSAGE, MANNEQUIN_EXCLUSION_MESSAGE, TICKET_DECLINED_MESSAGE,
    CLARIFY_DECLINE_PROMPT_MESSAGE, FILLER_RESPONSE_MESSAGE, GRATITUDE_MESSAGE, GREETING_MESSAGE,
}


def _is_non_substantive(response: str) -> bool:
    stripped = response.strip()
    if stripped in _FIXED_NON_SUBSTANTIVE_MESSAGES:
        return True
    if stripped.startswith("Unfortunately, Geometra isn't able to measure that"):
        return True  # dynamic exclusion-category/mannequin-style message
    return looks_like_clarify_question(stripped)


def _result(metric_name: str, passed: bool, score: Optional[float], detail: str) -> EvaluationResult:
    return EvaluationResult(metric_name=metric_name, passed=passed, score=score, detail=detail)


def retrieval_precision(chunks: List[SourceChunk]) -> EvaluationResult:
    top1 = chunks[0].similarity_score if chunks else 0.0
    passed = top1 >= LOW_CONFIDENCE_THRESHOLD
    return _result(
        "retrieval_precision", passed, top1,
        f"Top chunk score {top1} {'>=' if passed else '<'} {LOW_CONFIDENCE_THRESHOLD}",
    )


def answer_faithfulness(response: str, chunks: List[SourceChunk]) -> EvaluationResult:
    if _is_non_substantive(response):
        return _result("answer_faithfulness", True, 1.0, "Fallback/scope response — no factual claims to verify")

    chunk_words = set(re.findall(r"[a-z0-9]+", " ".join(c.text for c in chunks).lower()))
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", response) if s.strip()]

    checked = 0
    unfounded = 0
    for sentence in sentences:
        if sentence.startswith("[Source"):
            continue
        checked += 1
        content_words = {w for w in re.findall(r"[a-z0-9]+", sentence.lower()) if len(w) > 3}
        content_words.discard("geometra")
        if not content_words:
            continue
        overlap_ratio = len(content_words & chunk_words) / len(content_words)
        if overlap_ratio < 0.5:
            unfounded += 1

    passed = unfounded == 0
    score = 1.0 if checked == 0 else round(1 - unfounded / checked, 4)
    return _result("answer_faithfulness", passed, score, f"{unfounded} of {checked} sentences lack chunk overlap")


def answer_completeness(query: str, response: str) -> EvaluationResult:
    if "?" not in query:
        return _result("answer_completeness", True, None, "Query is not a question — not applicable")
    if _is_non_substantive(response):
        return _result("answer_completeness", True, None, "Fixed fallback/scope message — not applicable")
    word_count = len(response.split())
    passed = word_count > MIN_COMPLETENESS_WORDS
    return _result(
        "answer_completeness", passed, float(word_count),
        f"Response is {word_count} words (min {MIN_COMPLETENESS_WORDS})",
    )


def hallucination_rate(response: str, chunks: List[SourceChunk]) -> EvaluationResult:
    if _is_non_substantive(response):
        return _result("hallucination_rate", True, 0.0, "Fallback/scope response — no numbers to verify")
    chunk_numbers = set()
    for chunk in chunks:
        chunk_numbers.update(_extract_numbers(chunk.text))
    unlisted = [n for n in _extract_numbers(response) if n not in chunk_numbers and n not in ALLOWED_PRICE_NUMBERS]
    passed = len(unlisted) == 0
    return _result("hallucination_rate", passed, float(len(unlisted)), f"{len(unlisted)} unlisted numbers found: {unlisted}")


def source_citation_accuracy(response: str, chunks: List[SourceChunk]) -> EvaluationResult:
    if _is_non_substantive(response):
        return _result("source_citation_accuracy", True, None, "No citation required for fallback/scope response")
    match = CITATION_RE.search(response)
    if not match:
        return _result("source_citation_accuracy", False, None, "No [Source: ...] citation found in response")
    cited = [s.strip() for s in match.group(1).split(",")]
    valid_sections = {c.section for c in chunks}
    invalid = [s for s in cited if s not in valid_sections]
    passed = len(invalid) == 0
    return _result("source_citation_accuracy", passed, None, f"Cited {cited}; invalid: {invalid}")


def response_conciseness(response: str) -> EvaluationResult:
    word_count = len(response.split())
    passed = word_count <= 200
    return _result("response_conciseness", passed, float(word_count), f"Response is {word_count} words")


def fallback_trigger_rate(session_id: str) -> EvaluationResult:
    flags = get_session_message_flags(session_id)
    total = len(flags)
    if total == 0:
        return _result("fallback_trigger_rate", True, 0.0, "No prior messages in session")
    unknown_count = sum(1 for f in flags if f)
    ratio = round(unknown_count / total, 4)
    passed = ratio <= 0.15
    return _result("fallback_trigger_rate", passed, ratio, f"{unknown_count}/{total} prior turns were unknown ({ratio})")


def false_fallback_rate(chunks: List[SourceChunk], response: str) -> EvaluationResult:
    top1 = chunks[0].similarity_score if chunks else 0.0
    is_false_fallback = top1 >= LOW_CONFIDENCE_THRESHOLD and response.strip() == FALLBACK_MESSAGE
    passed = not is_false_fallback
    return _result("false_fallback_rate", passed, top1, "False fallback detected" if is_false_fallback else "No false fallback")


def response_latency(latency_ms: int) -> EvaluationResult:
    passed = latency_ms < 4000
    return _result("response_latency", passed, float(latency_ms), f"Latency {latency_ms}ms {'<' if passed else '>='} 4000ms")


def context_window_efficiency(input_tokens: int) -> EvaluationResult:
    passed = input_tokens < 3000
    return _result("context_window_efficiency", passed, float(input_tokens), f"Input tokens {input_tokens} {'<' if passed else '>='} 3000")


def price_accuracy(response: str) -> EvaluationResult:
    lower = response.lower()
    mentions_price = any(k in lower for k in ["price", "cost", "₹", "rupee", "per wall"])
    if not mentions_price:
        return _result("price_accuracy", True, None, "Response does not mention price — not applicable")
    passed = "399" in response or ("free" in lower and "3" in response)
    return _result("price_accuracy", passed, None, "Price matches FAQ" if passed else "Price does not match FAQ (expected 399 or free-plan-gives-3)")


def numerical_accuracy(response: str, chunks: List[SourceChunk]) -> EvaluationResult:
    if _is_non_substantive(response):
        return _result("numerical_accuracy", True, 0.0, "Fallback/scope response — no numbers to verify")
    chunk_numbers = set()
    for chunk in chunks:
        chunk_numbers.update(_extract_numbers(chunk.text))
    unlisted = [n for n in _extract_numbers(response) if n not in chunk_numbers and n not in ALLOWED_PRICE_NUMBERS]
    passed = len(unlisted) == 0
    return _result("numerical_accuracy", passed, float(len(unlisted)), f"{len(unlisted)} unlisted numbers: {unlisted}")


def tone_consistency(response: str) -> EvaluationResult:
    lowered = response.lower()
    found = [p for p in BANNED_PHRASES if p in lowered]
    passed = len(found) == 0
    return _result("tone_consistency", passed, None, "No banned phrases" if passed else f"Banned phrases found: {found}")


def multi_turn_coherence(session_id: str, turn_number: int) -> EvaluationResult:
    if turn_number <= 1:
        return _result("multi_turn_coherence", True, None, "First turn — not applicable")
    count = count_session_messages(session_id)
    passed = count >= turn_number - 1
    return _result("multi_turn_coherence", passed, float(count), f"Expected >= {turn_number - 1} prior messages, found {count}")


def unknown_q_detection(confidence_level: str, is_relevant: bool, response: str) -> EvaluationResult:
    if confidence_level != "unknown" or not is_relevant:
        return _result("unknown_q_detection", True, None, "Not applicable — not an on-topic unknown question")
    passed = response.strip() == FALLBACK_MESSAGE
    return _result("unknown_q_detection", passed, None, "Fallback correctly returned" if passed else "Fallback expected but not returned")


def out_of_scope_rejection(confidence_level: str, is_relevant: bool, response: str) -> EvaluationResult:
    if confidence_level != "unknown" or is_relevant:
        return _result("out_of_scope_rejection", True, None, "Not applicable — query is in-scope or similarity sufficient")
    passed = response.strip() == OUT_OF_SCOPE_MESSAGE
    return _result("out_of_scope_rejection", passed, None, "Scope rejection returned" if passed else "Scope rejection expected but not returned")


def sqlite_log_integrity(message_id: int, expected: Dict) -> EvaluationResult:
    row = get_message_by_id(message_id)
    if row is None:
        return _result("sqlite_log_integrity", False, None, "Message record not found after write")
    mismatches = []
    for key, value in expected.items():
        stored = row[key]
        if isinstance(value, bool):
            stored = bool(stored)
        if stored != value:
            mismatches.append(key)
    passed = len(mismatches) == 0
    return _result("sqlite_log_integrity", passed, None, "Re-read matches written record" if passed else f"Mismatched fields: {mismatches}")


def evaluate_core_metrics(
    query: str,
    response: str,
    chunks: List[SourceChunk],
    confidence_level: str,
    is_relevant: bool,
    session_id: str,
    turn_number: int,
    response_latency_ms: int,
    input_tokens: int,
) -> List[EvaluationResult]:
    """Metrics 1-16. Metric 17 (sqlite_log_integrity) runs separately after the DB write."""
    return [
        retrieval_precision(chunks),
        answer_faithfulness(response, chunks),
        answer_completeness(query, response),
        hallucination_rate(response, chunks),
        source_citation_accuracy(response, chunks),
        response_conciseness(response),
        fallback_trigger_rate(session_id),
        false_fallback_rate(chunks, response),
        response_latency(response_latency_ms),
        context_window_efficiency(input_tokens),
        price_accuracy(response),
        numerical_accuracy(response, chunks),
        tone_consistency(response),
        multi_turn_coherence(session_id, turn_number),
        unknown_q_detection(confidence_level, is_relevant, response),
        out_of_scope_rejection(confidence_level, is_relevant, response),
    ]
