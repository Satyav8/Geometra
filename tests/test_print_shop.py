"""Guards the business's stated position on print shops.

The FAQ says it outright: "Geometra will not specify or recommend any particular print shop
or location in your area." That wording exists because pointing a customer at a specific
shop would make Geometra answerable for a print it never saw.

Pass 2 did not honour it. Asked "what is the nearest print shop I should go to" in
production it replied "Can you share your location or area so I can better assist you?" -
an offer to do the exact thing the business refuses - while the correct answer sat in the
retrieved context at 0.577 similarity. So the position is stated deterministically rather
than re-derived by the model on every turn.

The must-not half matters just as much: a customer whose print came out wrong is
troubleshooting, not shopping, and still needs Pass 2 and the retrieved instructions.

Pure local regex checks - no LLM, no network.
"""
import pytest

from llm.two_pass import is_print_shop_location_question

SEEKING_A_SHOP = [
    # the phrasing that failed in production
    "what is the nearest print shop I should go to",
    # asking for a recommendation, in the ways people actually ask
    "which print shop should I use",
    "can you recommend a print shop near me",
    "is there any good printing shop nearby",
    "find me a xerox shop to print this",
    "any stationery store near me that can print it",
    "can I take a printout from a shop somewhere",
    # asking where, without ever saying "shop"
    "where can I get the marker printed",
    "where should I go to print the marker",
    "where do I print the marker",
]


@pytest.mark.parametrize("message", SEEKING_A_SHOP)
def test_shop_question_gets_the_fixed_position(message):
    assert is_print_shop_location_question(message) is True, (
        f"print-shop question left to Pass 2, which offers to recommend one: {message!r}"
    )


NOT_SEEKING_A_SHOP = [
    # troubleshooting a print that already happened - needs the real instructions
    "the print shop printed it too small what do I do",
    "the shop gave me a blurry print, can I rescan",
    # ordinary printing questions
    "how do I print the marker at home",
    "can I print on A4 paper",
    "what size should the marker be printed at",
    "do I need a laser printer for the marker",
    "my printer is an Epson L3110, is that ok",
    # "where" questions about something other than printing - the print VERB is what
    # separates these, since they share both the question word and the product noun
    "where do I place the marker on a wall",
    "where should the marker go on the wall",
    "where are my measurements saved",
    # unrelated
    "how much does it cost per wall",
    "can I measure my living room wall",
]


@pytest.mark.parametrize("message", NOT_SEEKING_A_SHOP)
def test_other_questions_are_left_alone(message):
    assert is_print_shop_location_question(message) is False, (
        f"wrongly answered with the print-shop position: {message!r}"
    )
