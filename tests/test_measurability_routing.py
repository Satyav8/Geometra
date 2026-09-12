"""Guards the routing around the measurability classifier (llm/measurability.py).

The classifier exists because extending _EXCLUDED_CATEGORIES by hand does not converge -
the categories are open-ended (every dish, every animal, every object), so there is always
a next miss, and each one reaches a customer as either a refusal for an innocent question
or a confidently wrong answer. Food was the case that made it obvious: Rule 8C names food
explicitly, yet "I want to measure lasagna" and "can I measure biryani" both got hard
SAFETY refusals from Pass 2.

What is tested here is the ROUTING, not the classifier's judgment - that is measured
separately against every category, because it is an LLM call and belongs in a measurement
run, not a unit test. Routing is what keeps the classifier cheap:

  - a question no deterministic cache can answer reaches it
  - a question either cache CAN answer never does, so the common path pays nothing

Pure local regex checks - no LLM, no network.
"""
import pytest

from llm.two_pass import (
    _is_measurability_question,
    find_definite_exclusion_reason,
    is_known_measurable,
)

# Open-ended things no regex should ever be asked to enumerate.
NEEDS_THE_CLASSIFIER = [
    "can I measure lasagna",
    "I want to measure biryani can I do that",
    "can I measure a dosa",
    "can I measure a violin",
    "can I measure a chandelier",
    "can I measure a tandoor",
    "can I measure a harmonium",
]


@pytest.mark.parametrize("message", NEEDS_THE_CLASSIFIER)
def test_unknown_object_reaches_the_classifier(message):
    assert _is_measurability_question(message) is True, (
        f"open-ended object left to Pass 2 instead of the classifier: {message!r}"
    )


# Anything either cache already knows must NOT pay for a call. This is the whole latency
# argument: these are the questions customers actually ask.
ALREADY_ANSWERED = [
    # in-scope cache
    "how much does it cost to measure a wall",
    "can I measure my living room wall",
    "can I measure the ceiling height of my room",
    "can I measure a wardrobe",
    "can I measure my kitchen cabinet",
    "can I measure a washbasin",
    # out-of-scope cache
    "can I measure my dog",
    "can I measure a knife",
    "can I measure a mannequin",
    "can I measure a black hole",
    "can I measure my wallet",
    "can I measure a ceramic vase",
    "can I measure food",
    # not a measurability question at all
    "how do I print the marker",
    "what size should the marker be",
    "how accurate are the measurements",
]


@pytest.mark.parametrize("message", ALREADY_ANSWERED)
def test_known_question_does_not_pay_for_a_call(message):
    assert _is_measurability_question(message) is False, (
        f"already-answerable question wasted a classifier call: {message!r}"
    )


# Rule 8C names food explicitly. The generic terms belong in the cache: the classifier
# reads a bare category word as naming no specific thing and answers "unclear", which sends
# the turn to Pass 2 - the exact path that refused lasagna as abuse.
GENERIC_FOOD = ["can I measure food", "can I measure a meal", "can I measure my lunch",
                "can I measure a snack", "can I measure dessert",
                "can I measure my breakfast", "can I measure a beverage"]


@pytest.mark.parametrize("message", GENERIC_FOOD)
def test_generic_food_is_excluded_without_a_call(message):
    reason = find_definite_exclusion_reason(message)
    assert reason is not None and "handheld" in reason, (
        f"generic food term not excluded deterministically: {message!r} -> {reason!r}"
    )


# A food court is a room, not a meal.
ROOMS_THAT_MENTION_FOOD = ["can I measure a food court", "can I measure the food hall"]


@pytest.mark.parametrize("message", ROOMS_THAT_MENTION_FOOD)
def test_food_court_is_still_a_room(message):
    assert find_definite_exclusion_reason(message) is None, (
        f"a room was excluded as food: {message!r}"
    )


def test_exclusions_outrank_the_in_scope_cache():
    """"can I measure my dog next to the wall" contains "wall", but the dog decides it."""
    message = "can I measure my dog next to the wall"
    assert is_known_measurable(message) is True          # the in-scope regex does match
    assert find_definite_exclusion_reason(message) is not None   # and is overruled
    assert _is_measurability_question(message) is False
