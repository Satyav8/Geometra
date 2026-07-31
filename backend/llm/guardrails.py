import re
from typing import List, Tuple

from config import FALLBACK_MESSAGE, ALLOWED_PRICE_NUMBERS
from models import SourceChunk

BANNED_PHRASES = [
    "i think",
    "i believe",
    "probably",
    "i'm not sure",
    "i am not sure",
    "it seems",
    "it appears",
    "perhaps",
    "i suppose",
]
# "might be" was in the original spec's list but is dropped here: the FAQ's own official
# accuracy answer says the measurement "might be off by ~10mm" (real tolerance, not the
# LLM hedging), so it false-positived on one of the most common customer questions.

NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")
# Ordered-list markers ("1. ", "2. " at the start of a line) look like numbers to NUMBER_RE
# but aren't facts — strip them before extracting numbers, or every numbered-list answer
# false-positives the hallucination check.
LIST_MARKER_RE = re.compile(r"^\s*\d+\.\s+", re.MULTILINE)


def _extract_numbers(text: str) -> List[float]:
    text = LIST_MARKER_RE.sub("", text)
    return [float(n) for n in NUMBER_RE.findall(text)]


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def check_numerical_hallucination(
    response: str, chunks: List[SourceChunk]
) -> Tuple[str, bool]:
    """Returns (response, triggered)."""
    if response == FALLBACK_MESSAGE:
        return response, False

    chunk_numbers = set()
    for chunk in chunks:
        chunk_numbers.update(_extract_numbers(chunk.text))

    response_numbers = _extract_numbers(response)
    unfounded = [
        n
        for n in response_numbers
        if n not in chunk_numbers and n not in ALLOWED_PRICE_NUMBERS
    ]

    if unfounded:
        response = response + "\n\n[Warning: response contains unverified numbers]"
        return response, True

    return response, False


def check_response_length(response: str, max_words: int = 200) -> Tuple[str, bool]:
    """Returns (response, triggered)."""
    words = response.split()
    if len(words) > max_words:
        truncated = " ".join(words[:max_words]) + "... [Response truncated for brevity]"
        return truncated, True
    return response, False


def check_uncertainty_language(response: str) -> Tuple[str, bool]:
    """Returns (response, triggered)."""
    lowered = response.lower()
    for phrase in BANNED_PHRASES:
        if phrase in lowered:
            return FALLBACK_MESSAGE, True
    return response, False


def check_fallback_leakage(response: str) -> Tuple[str, bool]:
    """Rule 2 requires the fallback to be the ENTIRE response. If the model prepends its
    own commentary before the exact fallback sentence, strip everything but the fallback.
    Whitespace-normalized comparison: the model sometimes uses a newline where the fallback
    constant has a plain space, which would otherwise defeat a literal substring match."""
    if response.strip() == FALLBACK_MESSAGE:
        return response, False
    if _normalize_whitespace(FALLBACK_MESSAGE) in _normalize_whitespace(response):
        return FALLBACK_MESSAGE, True
    return response, False
