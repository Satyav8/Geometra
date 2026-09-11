"""Guards the deterministic "Geometra can't measure that" categories.

Every behavioural bug found in this area had one root cause: a question no deterministic
layer claimed fell through to Pass 2, which judges it alone and over-refuses on
unusual-but-harmless input. Animals, celestial objects, weapons and printers were all
fixed by claiming the territory in code rather than by rewording the prompt.

A production smoke test found the next instance: "can I measure my friend standing by the
wall" got the hard SAFETY refusal, because "friend" was not in the living-thing list while
"person" and "dog" were. The person half of these tests holds that closed.

The other half guards the cost of closing it. The matcher only requires "measure" somewhere
and the noun somewhere - it never checks that the noun is what is being measured - so
naming a person as a BENEFICIARY ("measure this wall for my client") must keep answering
normally. That is ordinary studio work and the single most likely way this fix could make
the bot worse.

Pure local regex checks - no LLM, no network.
"""
import pytest

from llm.two_pass import find_definite_exclusion_reason

MUST_EXCLUDE = [
    # the case found in production
    ("can I measure my friend standing by the wall", "living thing"),
    ("can I measure my friend", "living thing"),
    # other ways a customer names a person in the room
    ("can I measure my brother", "living thing"),
    ("can I measure my wife", "living thing"),
    ("can I measure my neighbour", "living thing"),
    ("can I measure my colleague", "living thing"),
    ("can I measure a guest in the room", "living thing"),
    ("can I measure my mom", "living thing"),
    # the categories that already worked, re-checked so the new words didn't disturb them
    ("can I measure a person", "living thing"),
    ("can I measure my dog", "living thing"),
    ("can I measure a lion", "living thing"),
    ("can I measure a mannequin", None),          # handled by its own check, not here
    # only the trailing beneficiary phrase is removed, so the real subject survives
    ("can I measure my dog for my vet", "living thing"),
]


@pytest.mark.parametrize("message,expected_fragment", MUST_EXCLUDE)
def test_person_or_animal_is_excluded(message, expected_fragment):
    reason = find_definite_exclusion_reason(message)
    if expected_fragment is None:
        return
    assert reason is not None, f"should have been excluded but wasn't: {message!r}"
    assert expected_fragment in reason, (
        f"{message!r} excluded for the wrong reason: {reason!r}"
    )


MUST_NOT_EXCLUDE = [
    # a person named as the beneficiary - ordinary studio work
    "can I measure this wall for my client",
    "I need to measure a wall for my friend",
    "can I measure the wall for my kids",
    "measure the room for my boss",
    "we are a studio measuring walls for our customers",
    # room names that contain a person word - the long-standing lookahead
    "can I measure my kids room",
    "can I measure the baby nursery",
    "can I measure my bro's man cave",
    # the object is a thing, the person is incidental
    "can I measure a cage for my parrot",
    # plain product questions
    "how much does it cost to measure a wall",
    "can I measure my living room wall",
    "can I measure a kitchen cabinet",
]


@pytest.mark.parametrize("message", MUST_NOT_EXCLUDE)
def test_ordinary_measurement_request_is_not_excluded(message):
    assert find_definite_exclusion_reason(message) is None, (
        f"ordinary request wrongly refused as out of scope: {message!r}"
    )
