"""Guards language identification and the answer-language instruction.

TWO BUGS, REPORTED TOGETHER ON 2026-10-11, WITH DIFFERENT CAUSES

  "Can i meausre humayun's tomb"            English, answered in Hinglish
  "Kya mei iske saat diwaron ko maap..."    Hinglish, answered in English

The first was the detector. It counted dictionary words, and a typo ("meausre") plus a
proper noun ("humayun's") is two unknowns out of four - enough to call a plain English
question foreign, translate it, and render the reply in Hinglish.

No threshold fixes that. Measured across both populations:

    minimum unknown words   English wrongly flagged   Hinglish missed
              2                     4/16                   0/14
              3                     1/16                   3/14
              4                     0/16                   9/14

And the spellchecker cannot separate them either - it maps Hindi to English as readily as
typos: kahan->khan, maap->map, kaise->raise, all at distance 1.

The second was simpler: the Pass 2 prompt contained no instruction about language at all.
Every non-English answer was the model mirroring of its own accord, which it mostly did.

These tests cover the parts that are deterministic - the fast paths, the routing, and the
instruction text. The classifier's judgement is an LLM call, measured in a run (25/25 at
the time of writing), not asserted here.
"""
import pytest

from llm import language


def test_plain_english_costs_no_call(monkeypatch):
    """The affordability argument: ordinary traffic must not reach the classifier."""
    monkeypatch.setattr(language, "_ask", lambda text, fallback: pytest.fail(
        f"classifier called for plainly English text: {text!r}"))
    for text in ["how much does it cost per wall",
                 "can I measure my wardrobe",
                 "where can I get the marker printed",
                 "is the A4 marker ok for a 7m wall"]:
        assert language.identify(text) == language.ENGLISH


def test_non_latin_script_costs_no_call(monkeypatch):
    """Devanagari, Telugu and Tamil are settled by the bytes.

    Asking was not merely wasteful, it was wrong: the classifier labelled Telugu and
    Tamil as "hindi", which would have instructed Pass 2 to answer a Telugu speaker in
    Devanagari. The script alone gives the only instruction that matters - mirror it.
    """
    monkeypatch.setattr(language, "_ask", lambda text, fallback: pytest.fail(
        f"classifier called for non-Latin text: {text!r}"))
    for text in ["क्या मैं दीवार माप सकता हूँ", "నేను గోడను కొలవగలనా", "சுவரை அளக்க முடியுமா"]:
        assert language.identify(text) == "other"


def test_the_ambiguous_middle_is_asked(monkeypatch):
    """The reported failure lives here: a dictionary cannot tell this from Hinglish, so
    it must not be decided locally."""
    asked = []
    monkeypatch.setattr(language, "_ask",
                        lambda text, fallback: asked.append(text) or language.ENGLISH)
    language.identify("Can i meausre humayun's tomb")
    assert asked, "a typo-and-name message was decided without asking"


def test_a_failed_classifier_falls_back_to_english(monkeypatch):
    """English means no translation and no language note - exactly the behaviour before
    this module existed, which is the right thing to degrade to."""
    def _boom(prompt, text):
        raise RuntimeError("no network")

    monkeypatch.setattr(language, "call_llm", _boom)
    assert language.identify("kya main deewar maap sakta hun") in {"hinglish", "english"}
    assert language.identify("Can i meausre humayun's tomb") == language.ENGLISH


def test_very_short_messages_are_left_alone(monkeypatch):
    monkeypatch.setattr(language, "_ask", lambda text, fallback: pytest.fail("called"))
    assert language.identify("ok") == language.ENGLISH
    assert language.identify("") == language.ENGLISH
    assert language.identify("hi") == language.ENGLISH


def test_only_non_english_triggers_translation():
    assert language.needs_translation("english") is False
    for other in ("hinglish", "hindi", "other"):
        assert language.needs_translation(other) is True


def test_english_gets_no_language_instruction():
    """A note on every English turn would be prompt bloat for no gain, and this prompt is
    already measured as unreliable under its own weight."""
    assert language.answer_language_instruction("english") is None


@pytest.mark.parametrize("lang", ["hinglish", "hindi", "other"])
def test_non_english_gets_an_instruction_that_protects_numbers(lang):
    note = language.answer_language_instruction(lang)
    assert note, f"no instruction produced for {lang}"
    # Prices and specs must survive translation - a localized "Rs 199" that became
    # "Rs 200" would be worse than answering in the wrong language.
    assert "199" in note and "Geometra" in note


def test_hinglish_is_told_not_to_use_devanagari_or_english():
    """The specific trap: Hinglish is Latin script carrying Hindi, so "mirror the script"
    is ambiguous - it has to be said explicitly."""
    note = language.answer_language_instruction("hinglish")
    assert "Devanagari" in note and "English" in note
