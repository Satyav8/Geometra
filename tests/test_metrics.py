import uuid

from config import FALLBACK_MESSAGE, OUT_OF_SCOPE_MESSAGE
from database import ensure_session, write_message
from evaluation import metrics
from models import SourceChunk


def make_chunk(section="Pricing", text="We are priced at 399 per wall.", score=0.9):
    return SourceChunk(chunk_id="faq_010", section=section, text=text, similarity_score=score)


def test_retrieval_precision_pass_and_fail():
    high = metrics.retrieval_precision([make_chunk(score=0.9)])
    low = metrics.retrieval_precision([make_chunk(score=0.4)])
    assert high.passed is True
    assert low.passed is False


def test_answer_faithfulness_detects_ungrounded_sentence():
    chunks = [make_chunk(text="Geometra is priced at 399 per wall elevation.")]
    grounded = metrics.answer_faithfulness(
        "Geometra pricing is 399 per wall elevation. [Source: Pricing]", chunks
    )
    ungrounded = metrics.answer_faithfulness(
        "Geometra was founded by astronauts in space stations. [Source: Pricing]", chunks
    )
    assert grounded.passed is True
    assert ungrounded.passed is False


def test_answer_completeness_requires_length_for_questions():
    short = metrics.answer_completeness("What is Geometra?", "It's a tool.")
    long_enough = metrics.answer_completeness(
        "What is Geometra?",
        "Geometra is an image-to-CAD system that measures wall elevations directly from phone photos taken by any standard phone camera, offering high accuracy. [Source: Product Overview]",
    )
    not_a_question = metrics.answer_completeness("tell me about geometra", "It's a tool.")
    assert short.passed is False
    assert long_enough.passed is True
    assert not_a_question.passed is True


def test_hallucination_rate_flags_unlisted_numbers():
    chunks = [make_chunk(text="Accuracy is 99%+.")]
    clean = metrics.hallucination_rate("Accuracy is 99%+. [Source: Accuracy]", chunks)
    dirty = metrics.hallucination_rate("Accuracy is 87%. [Source: Accuracy]", chunks)
    assert clean.passed is True
    assert dirty.passed is False


def test_source_citation_accuracy():
    chunks = [make_chunk(section="Pricing")]
    valid = metrics.source_citation_accuracy("The price is 399. [Source: Pricing]", chunks)
    invalid = metrics.source_citation_accuracy("The price is 399. [Source: Accuracy]", chunks)
    missing = metrics.source_citation_accuracy("The price is 399.", chunks)
    assert valid.passed is True
    assert invalid.passed is False
    assert missing.passed is False


def test_response_conciseness():
    short = metrics.response_conciseness("Short response.")
    long_response = " ".join(["word"] * 250)
    long_result = metrics.response_conciseness(long_response)
    assert short.passed is True
    assert long_result.passed is False


def test_false_fallback_rate():
    high_conf_chunks = [make_chunk(score=0.9)]
    false_fallback = metrics.false_fallback_rate(high_conf_chunks, FALLBACK_MESSAGE)
    normal_response = metrics.false_fallback_rate(high_conf_chunks, "The price is 399. [Source: Pricing]")
    assert false_fallback.passed is False
    assert normal_response.passed is True


def test_response_latency():
    fast = metrics.response_latency(1200)
    slow = metrics.response_latency(5000)
    assert fast.passed is True
    assert slow.passed is False


def test_context_window_efficiency():
    small = metrics.context_window_efficiency(1500)
    large = metrics.context_window_efficiency(3500)
    assert small.passed is True
    assert large.passed is False


def test_price_accuracy():
    correct = metrics.price_accuracy("The price is 399 per wall. [Source: Pricing]")
    wrong = metrics.price_accuracy("The price is 500 per wall. [Source: Pricing]")
    not_applicable = metrics.price_accuracy("Geometra measures wall elevations. [Source: Product Overview]")
    assert correct.passed is True
    assert wrong.passed is False
    assert not_applicable.passed is True


def test_tone_consistency():
    clean = metrics.tone_consistency("The price is 399 per wall. [Source: Pricing]")
    hedging = metrics.tone_consistency("I think the price might be 399.")
    assert clean.passed is True
    assert hedging.passed is False


def test_unknown_q_detection():
    correct = metrics.unknown_q_detection("unknown", True, FALLBACK_MESSAGE)
    incorrect = metrics.unknown_q_detection("unknown", True, "Some made up answer.")
    not_applicable_confidence = metrics.unknown_q_detection("high", True, "Some answer. [Source: Pricing]")
    not_applicable_irrelevant = metrics.unknown_q_detection("unknown", False, OUT_OF_SCOPE_MESSAGE)
    assert correct.passed is True
    assert incorrect.passed is False
    assert not_applicable_confidence.passed is True
    assert not_applicable_irrelevant.passed is True


def test_out_of_scope_rejection():
    correct = metrics.out_of_scope_rejection("unknown", False, OUT_OF_SCOPE_MESSAGE)
    incorrect = metrics.out_of_scope_rejection("unknown", False, "It's sunny.")
    in_scope = metrics.out_of_scope_rejection("unknown", True, FALLBACK_MESSAGE)
    not_unknown = metrics.out_of_scope_rejection("high", False, "Anything")
    assert correct.passed is True
    assert incorrect.passed is False
    assert in_scope.passed is True
    assert not_unknown.passed is True


def test_multi_turn_coherence():
    session_id = str(uuid.uuid4())
    ensure_session(session_id)
    first_turn = metrics.multi_turn_coherence(session_id, 1)
    assert first_turn.passed is True

    write_message(
        session_id=session_id,
        turn_number=1,
        query="q1",
        response="r1",
        retrieved_chunk_ids=["faq_000"],
        similarity_scores=[0.9],
        confidence_level="high",
        is_unknown_question=False,
        response_latency_ms=100,
        input_tokens=10,
        output_tokens=10,
    )
    second_turn = metrics.multi_turn_coherence(session_id, 2)
    assert second_turn.passed is True

    missing_history = metrics.multi_turn_coherence(str(uuid.uuid4()), 3)
    assert missing_history.passed is False


def test_fallback_trigger_rate():
    session_id = str(uuid.uuid4())
    ensure_session(session_id)
    no_history = metrics.fallback_trigger_rate(session_id)
    assert no_history.passed is True

    for i in range(1, 6):
        write_message(
            session_id=session_id,
            turn_number=i,
            query=f"q{i}",
            response="r",
            retrieved_chunk_ids=[],
            similarity_scores=[],
            confidence_level="unknown",
            is_unknown_question=True,
            response_latency_ms=100,
            input_tokens=0,
            output_tokens=0,
        )
    heavy_unknown = metrics.fallback_trigger_rate(session_id)
    assert heavy_unknown.passed is False


def test_sqlite_log_integrity():
    session_id = str(uuid.uuid4())
    ensure_session(session_id)
    message_id = write_message(
        session_id=session_id,
        turn_number=1,
        query="What is Geometra?",
        response="Geometra is an image-to-CAD tool. [Source: Product Overview]",
        retrieved_chunk_ids=["faq_000"],
        similarity_scores=[0.9],
        confidence_level="high",
        is_unknown_question=False,
        response_latency_ms=800,
        input_tokens=100,
        output_tokens=50,
    )
    matching = metrics.sqlite_log_integrity(
        message_id,
        {
            "session_id": session_id,
            "turn_number": 1,
            "query": "What is Geometra?",
            "response": "Geometra is an image-to-CAD tool. [Source: Product Overview]",
            "confidence_level": "high",
            "is_unknown_question": False,
            "response_latency_ms": 800,
            "input_tokens": 100,
            "output_tokens": 50,
        },
    )
    mismatching = metrics.sqlite_log_integrity(message_id, {"response": "wrong text"})
    assert matching.passed is True
    assert mismatching.passed is False
