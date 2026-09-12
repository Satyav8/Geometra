"""Guards the test that decides whether a clarifying question was justified.

A 135-case production battery measured 12% of answers coming back as a clarifying question
on things the FAQ answers outright - "can you teach me how to take a picture", "the room is
dark so use extra lighting right", "should he be standing further back from the wall". Rule
2 in the Pass 2 prompt already ends with "clarifying a question you could already answer is
worse than just answering it", so this was never a missing instruction; it is the same
22k-prompt dilution seen throughout llm/two_pass.py.

Retrieval confidence cannot gate it. Measured, the two populations overlap almost exactly:
"can I measure it" scores 0.475 and "can you teach me how to take a picture" scores 0.483.
What separates them is whether the customer named a subject at all, which this tests
directly - a concrete question that drew a clarifying question gets one retry with the
blunt instruction that already works for the clarification cap.

Both halves are load-bearing. Too strict and genuinely ambiguous questions get a guessed
answer instead of a question; too loose and the 12% stays.

Pure local checks - no LLM, no network.
"""
import pytest

from llm.two_pass import is_genuinely_vague

# Named a subject. A clarifying question here is the bug - these must be retried.
CONCRETE = [
    # the cases measured in production
    "can you teach me how to take a picture",
    "the room is dark so use extra lighting for the photo right",
    "should he be standing further back from the wall",
    "place the marker and wall together in one photo",
    "the wall is XXX cm wide, does that matter",
    "my name is Randi and I need help measuring a wall",
    "we are a studio measuring walls for our customers",
    "I am from Lund Sweden, can I use Geometra",
    "can I measure the wall for my kids",
    # ordinary short questions that still name their subject
    "how do I print the marker at home",
    "what is the price",
    "which marker size",
]


@pytest.mark.parametrize("message", CONCRETE)
def test_concrete_question_is_not_treated_as_vague(message):
    assert is_genuinely_vague(message) is False, (
        f"concrete question would be allowed to draw a clarifying question: {message!r}"
    )


# Leans on a referent instead of naming a subject, or is too short to carry one. A
# clarifying question is the RIGHT answer here and must not be retried away.
VAGUE = [
    "can I measure it",
    "yeah the thing over there",
    "3000 2500 1800",
    "can I do that",
    "is it possible",
    "what about this one",
    "help",
    "I need that",
]


@pytest.mark.parametrize("message", VAGUE)
def test_vague_question_still_earns_a_clarification(message):
    assert is_genuinely_vague(message) is True, (
        f"genuinely vague message would be forced into a guessed answer: {message!r}"
    )
