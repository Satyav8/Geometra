"""Guards the profanity wordlist against false positives on ordinary customer messages.

Exists because of a real production bug: better-profanity strips spaces to catch
spaced-out evasion ("a s s" -> "ass"), so any term in the multilingual list can also be
formed by running consecutive ordinary words together. The Turkish term "amina" is exactly
what "am in a" collapses to, which meant every sentence containing that phrase - "I am in
a hotel room, can I measure the wall" - got a hard safety refusal.

These are pure local checks: no LLM, no moderation API, no network. Safe to run in the
normal suite, which never touches paid APIs.
"""
import pytest

from llm.two_pass import is_severe_slur

# Ordinary things a customer says. None of these may ever be treated as abuse.
LEGITIMATE_MESSAGES = [
    # the phrase that caused the original bug, in several natural forms
    "I am in a hotel room, can I measure the wall",
    "I am in a meeting room and want to measure it",
    "I am in a hurry, how fast is the scan",
    "I am in a rush to get this done",
    "I am in a different country, does it work",
    "I am in an office and need to measure a wall",
    "I am in a warzone right now and want to measure the wall",
    # general product questions
    "how much does it cost per wall",
    "can I measure my living room wall",
    "where do I place the marker on a wall",
    "how do I print the marker at home",
    "is a laser printer required",
    "the marker is not sticking to the wall",
    "can I get a refund for this scan",
    "my wall is 3 meters long and 2 meters high",
    "we are a studio and need bulk scans",
    "what is the price in india",
    "my team in india wants to use geometra",
    "can I measure a kitchen cabinet",
    "does it work in low light conditions",
    # names and places that overlap excluded-term reasoning
    "my name is Randi and I need help measuring a wall",
    "I am from Lund, Sweden",
    "can I measure the chutney jar shelf",
    # legitimate questions in Indian languages
    "marker ka size kya hona chahiye 3 meter wall ke liye",
    "main deewar kaise maap sakta hoon",
    "मैं दीवार कैसे माप सकता हूँ",
    "नेने गोडने एला कोलवali",
    "मी भिंत कशी मोजू शकतो",
]


@pytest.mark.parametrize("message", LEGITIMATE_MESSAGES)
def test_legitimate_message_is_not_flagged_as_abuse(message):
    assert is_severe_slur(message) is False, (
        f"ordinary customer message wrongly flagged as abuse: {message!r}"
    )


# Real abuse must still be caught - the guard above must not be satisfied by simply
# weakening the wordlist.
ABUSIVE_MESSAGES = [
    "you dirty niggggga get out",
    "abe madarchod bhenchod chutiya",
    "nuvvu oka lanja koduku",
    "आई घाल झवाड्या भडव्या",
    "தேவிடியா புண்டை",
    "orospu cocugu siktir git",
]


@pytest.mark.parametrize("message", ABUSIVE_MESSAGES)
def test_abuse_is_still_flagged(message):
    assert is_severe_slur(message) is True, (
        f"abusive message no longer flagged: {message!r}"
    )
