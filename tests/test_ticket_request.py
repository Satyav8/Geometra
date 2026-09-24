"""Guards Rule 2C: a ticket request with no problem described must draw a clarifying
question, never an answer.

Rule 2C has always said this - we cannot raise a ticket without knowing what it would be
about. Pass 2 applied it inconsistently. Measured against production on one build:

    "raise a ticket"                    clarified              correct
    "i want to raise a ticket"          answered with contact@geometra.in
    "I want to raise a ticket can i ?"  answered with contact@geometra.in

Same intent, three phrasings, two behaviours. The FAQ holds chunks that LOOK like answers
here ("how can I reach out to geometra team", "turnaround time if I raise a support
request"), so Rule 1's answer-from-context pull beat Rule 2C - the same 22k-prompt dilution
behind the print-shop and measurability fixes.

The must-not half is the constraint that stops this from being harmful: a request that
names a topic has to keep reaching Pass 2, which should engage with the actual problem.
Claiming those too would replace a real answer with a question, which is worse than the bug.

Pure local checks - no LLM, no network.
"""
import pytest

from llm.two_pass import is_bare_ticket_request

# No subject given. There is nothing to raise a ticket about yet.
BARE = [
    # the exact phrasings seen failing in production
    "I want to raise a ticket can i ?",
    "i want to raise a ticket",
    "raise a ticket",
    # other ways people ask for the same thing
    "can you raise a ticket for me",
    "I need a support ticket",
    "please open a ticket",
    "create a ticket",
    "I want to raise a complaint",
    "can I get a ticket raised",
    "i want a ticket to be created",
    "kindly log a support case",
]


@pytest.mark.parametrize("message", BARE)
def test_bare_request_asks_what_it_is_about(message):
    assert is_bare_ticket_request(message) is True, (
        f"bare ticket request left to Pass 2, which answers it with the support email "
        f"instead of asking: {message!r}"
    )


# A subject IS given. These belong to Pass 2 - Rule 2C says engage with the real problem.
NAMES_A_TOPIC = [
    "I want to raise a ticket regarding the refund policy",
    "raise a ticket because my measurements are inaccurate",
    "raise a ticket about the marker not sticking",
    "my scan failed, raise a ticket",
    "raise a ticket for my printer issue",
]


@pytest.mark.parametrize("message", NAMES_A_TOPIC)
def test_request_naming_a_topic_still_reaches_pass_2(message):
    assert is_bare_ticket_request(message) is False, (
        f"a described problem would be replaced by a clarifying question: {message!r}"
    )


# Nothing to do with raising a ticket.
UNRELATED = [
    "how much does Geometra cost",
    "can I measure a wall",
    "what is the turnaround time if a scan fails",
    "how do I contact support",
    "can I measure a ticket booth",
]


@pytest.mark.parametrize("message", UNRELATED)
def test_unrelated_messages_are_untouched(message):
    assert is_bare_ticket_request(message) is False, (
        f"unrelated message wrongly treated as a ticket request: {message!r}"
    )
