import os
import sys

BACKEND_DIR = os.path.join(os.path.dirname(__file__), "..", "backend")
sys.path.insert(0, os.path.abspath(BACKEND_DIR))

from config import FALLBACK_MESSAGE  # noqa: E402
from llm.guardrails import (  # noqa: E402
    check_fallback_leakage,
    check_numerical_hallucination,
    check_response_length,
    check_uncertainty_language,
)
from models import SourceChunk  # noqa: E402


def make_chunk(text, section="Pricing", score=0.9):
    return SourceChunk(chunk_id="faq_000", section=section, text=text, similarity_score=score)


def test_uncertainty_language_triggers_fallback():
    response, triggered = check_uncertainty_language("I think the price is 399.")
    assert triggered is True
    assert response == FALLBACK_MESSAGE


def test_uncertainty_language_allows_clean_response():
    response, triggered = check_uncertainty_language("The price is 399 per wall. [Source: Pricing]")
    assert triggered is False
    assert "399" in response


def test_uncertainty_language_allows_factual_tolerance_wording():
    # "might be off by X" describes real measurement variance (this is the FAQ's own
    # wording for accuracy), not the LLM hedging about whether it knows the answer.
    response, triggered = check_uncertainty_language(
        "Accuracy is 99%+. On a 1000mm length, it might be off by about 10mm. [Source: Accuracy]"
    )
    assert triggered is False


def test_response_length_truncates_over_200_words():
    long_response = " ".join(["word"] * 250)
    response, triggered = check_response_length(long_response)
    assert triggered is True
    assert response.endswith("[Response truncated for brevity]")
    assert len(response.split()) <= 205


def test_response_length_leaves_short_response_alone():
    response, triggered = check_response_length("Short answer here.")
    assert triggered is False
    assert response == "Short answer here."


def test_numerical_hallucination_flags_unfounded_number():
    chunks = [make_chunk("Accuracy is 99%+ with 10mm variance.")]
    response, triggered = check_numerical_hallucination(
        "The device weighs 500 grams. [Source: Pricing]", chunks
    )
    assert triggered is True
    assert "Warning" in response


def test_numerical_hallucination_allows_grounded_number():
    chunks = [make_chunk("We are priced at 399 per wall.")]
    response, triggered = check_numerical_hallucination(
        "The price is 399 per wall. [Source: Pricing]", chunks
    )
    assert triggered is False


def test_numerical_hallucination_allows_constants():
    chunks = [make_chunk("Free plan gives 3 walls.")]
    response, triggered = check_numerical_hallucination(
        "Accuracy is 99%+ and pricing is 399 per wall. [Source: Pricing]", chunks
    )
    assert triggered is False


def test_fallback_leakage_strips_prepended_commentary():
    leaked = (
        "Geometra doesn't mention this specific case in its documentation. " + FALLBACK_MESSAGE
    )
    response, triggered = check_fallback_leakage(leaked)
    assert triggered is True
    assert response == FALLBACK_MESSAGE


def test_fallback_leakage_leaves_clean_fallback_alone():
    response, triggered = check_fallback_leakage(FALLBACK_MESSAGE)
    assert triggered is False
    assert response == FALLBACK_MESSAGE


def test_fallback_leakage_leaves_normal_response_alone():
    response, triggered = check_fallback_leakage("The price is 399 per wall. [Source: Pricing]")
    assert triggered is False
    assert "399" in response
