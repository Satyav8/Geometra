"""Guards the gate that decides whether a message reaches the printer-type classifier.

A production smoke test found the original gate (a print-family word AND a model token)
missed four of eight real-model phrasings, because customers name hardware without ever
saying "print": "I have an HP DeskJet 2331, can I use it". Those fell to Pass 2, which
told a customer the TVS MSP 250 - a dot matrix printer - was an inkjet. Printing the
marker on a dot matrix produces a failed measurement, so that is a real cost, not cosmetic.

The gate is now generous on purpose (possession framing, or "marker" as printing context),
which is only safe because the classifier can answer "notprinter" and hand the turn back
to Pass 2 untouched. These tests hold both halves in place: the phrasings that must route,
and the ordinary traffic that must not pay for an extra call.

Pure local regex checks - no LLM, no network.
"""
import pytest

from llm.two_pass import mentions_printer_model

MUST_ROUTE = [
    # print-family word present (worked before, must keep working)
    "my printer is an Epson EcoTank L3250, is that ok",
    "I only have an Epson LX-310, can I print the marker",
    "is a Brother HL-L2321D okay for printing the marker",
    # "marker" as printing context
    "I have an HP LaserJet 1020, is it fine for the marker",
    "will a Canon PIXMA G3010 work for the marker",
    # possession framing, no printing word at all
    "I have an HP DeskJet 2331, can I use it",
    "I own a Canon imageCLASS LBP2900B, will it work",
    "we have a TVS MSP 250 star at the office, will it do",
    "I have a Pantum P2200W, is that suitable",
]


@pytest.mark.parametrize("message", MUST_ROUTE)
def test_named_model_reaches_the_classifier(message):
    assert mentions_printer_model(message) is True, (
        f"message naming a printer model was not routed to the classifier: {message!r}"
    )


MUST_NOT_ROUTE = [
    # ordinary product questions
    "how much does Geometra cost per wall",
    "can I measure my living room wall",
    "how many corners must be visible",
    "thank you so much",
    # dimensions read as model-shaped tokens if the shape test is too loose
    "my wall is 3000mm wide and 2500mm high",
    "the wall is XXX cm wide",
    "3000 2500 1800",
    # printing questions with no model named - these want the full retrieved
    # instructions, not a one-line type verdict
    "how do I print the marker at home",
    "what size should the marker be printed at",
    "can I print on A4 paper",
    # a shouting customer: every token has capitals, which must not read as a model
    "HOW DO I MEASRUE A WALL",
    # possession framing with nothing model-shaped nearby
    "we have 15 walls to measure this week",
    "I have a question about pricing",
    "we use Geometra for our studio",
    # regressions from the abuse-guard work, re-checked here because this gate also
    # inspects raw tokens
    "I am in a hotel room, can I measure the wall",
    "place the marker and wall together",
    "do we need separate logins for each team member",
]


@pytest.mark.parametrize("message", MUST_NOT_ROUTE)
def test_ordinary_message_does_not_reach_the_classifier(message):
    assert mentions_printer_model(message) is False, (
        f"ordinary message wrongly routed to the printer classifier: {message!r}"
    )
