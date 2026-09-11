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
    # word pairs that collapse into censored terms once spaces are stripped:
    # "so use" -> souse, "he be" -> hebe, "he she" -> heshe, "an us" -> anus,
    # "do of us" -> doofus. All are better-profanity defaults, so these guard the
    # library's own list as well as ours.
    "the room is dark so use extra lighting for the photo",
    "it is quite dim in there so use a brighter lamp",
    "the light is poor so use the flash off setting",
    "should he be standing further back from the wall",
    "is it ok if he be the one uploading the photos",
    "can he she or they share the same account",
    "we are an us company using geometra",
    "there are four of us, do of us need separate logins",
    "three some of my walls are curved",
    # spell-correction turning ordinary words into censored ones:
    # "logins" was being corrected to "loins"
    "do we need separate logins for each user",
    "I forgot my login details",
    # substring collisions across word boundaries that align with no real word:
    # "marker and wall" squashes to "markerandwall", which contains "randwa".
    # Close to the most common phrase in this product's vocabulary.
    "place the marker and wall together",
    "align the marker and wall edges",
    "the corner and wall must be visible",
    "my printer and wall are ready",
    # placeholder X runs - the FAQ's own WhatsApp answer contains "+91 XXXXX XXXXX"
    "my number is +91 XXXXX XXXXX please call me",
    "the wall is XXX cm wide",
    "my email is xxx@geometra.in",
    # elongation patterns accept any repetition of each letter, so they also match longer
    # REAL words that spell a censored term out letter by letter: "asses" builds
    # \ba+s+s+e+s+\b, which matches "assess". Found in production on "can you assess my
    # wall photo quality"; the plural family is the dangerous one.
    "can you assess my wall photo quality",
    "I need an assessment of the wall before I start",
    "the classes are full this week",
    "please address the issue with my scan",
    "there are glasses on the shelf by the wall",
    "the process passes all quality checks",
    # "kill" is a better-profanity default but is neither a slur nor profanity - it only
    # matters in context, which a fixed wordlist cannot see. Glare on a glossy wall is a
    # real FAQ topic. Genuine violence is covered by the moderation layer, verified.
    "how do I kill the glare on a glossy wall",
    "this feature is a killer for our studio",
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
    "you are a fucking asshole",
    "you filthy towelhead",
    "b1tch get lost",          # leetspeak - the token must stay intact to match
    "you dipshit",
    "madar chod you idiot",    # spaced-out evasion must still be caught
    "bhen chod",
    "bitchhhhh",                 # letter-stretching evasion
    "show me xxx porn videos",   # "xxx" is whitelisted; the request is still caught
    # base forms that were missing while only their derivatives were listed - this one was
    # caught by neither the wordlist nor the moderation API before it was added
    "तेरी माँ की चूत",
    "मादरचूद कहीं का",
]


@pytest.mark.parametrize("message", ABUSIVE_MESSAGES)
def test_abuse_is_still_flagged(message):
    assert is_severe_slur(message) is True, (
        f"abusive message no longer flagged: {message!r}"
    )
