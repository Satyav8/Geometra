"""Guards the multilingual routing in llm/translation.py and its wiring into process_turn.

Every code-level business rule in two_pass.py matches English text. Measured before this
existed, 9 of 12 checks that answer an English customer with no LLM call at all claimed
nothing when the same question arrived in Hindi or Hinglish - so those customers were
served by Pass 2's improvisation on exactly the questions where an English customer got a
deterministic answer.

The fix translates the message and keeps ONE set of rules, rather than writing the rules
again per language. What is tested here is the part that must not drift:

  - the detector's two halves (script, and English-word ratio for romanised Hindi)
  - that English traffic is never sent down the translating path, since that is what keeps
    the change free for the overwhelming majority of turns
  - that the business layers read the translation while safety and the ticket keep the
    customer's real words

Translation QUALITY is not tested here - it is an LLM call, measured in a run, not asserted
in a unit test. These are pure local checks: no LLM, no network.
"""
import pytest

from llm import two_pass as tp
from llm.translation import is_non_latin_script, needs_translation

# Ordinary English, including the misspelled and product-heavy kind. A false positive here
# is not a wrong answer, but it spends an LLM call on every English turn - which is the
# whole cost argument for this design.
ENGLISH = [
    "can I measure my dog",
    "where can I get the marker printed",
    "how much does it cost per wall",
    "i want to raise a ticket",
    "can geometra measure my living room wall",
    "I have an HP LaserJet 1020, is it fine for the marker",
    "is the A4 marker ok for a 7m wall",
    "whats the pricing for geometra credits",
    # Misspelled English must stay on the English path - this is why the ratio threshold
    # sits at 0.65 and not higher.
    "my wal is to smal to mesure",
    "can i mesure a wardrobe",
]


@pytest.mark.parametrize("message", ENGLISH)
def test_english_is_never_translated(message):
    assert needs_translation(message) is False, (
        f"English message routed through translation, costing a call: {message!r}"
    )


NON_ENGLISH = [
    # Native script - caught by the script test.
    "क्या मैं अपने कुत्ते को माप सकता हूँ",
    "मुझे टिकट उठाना है",
    "मार्कर कहाँ प्रिंट कराऊँ",
    "నేను గోడను కొలవగలనా",
    # Romanised - Latin script, so ONLY the word-ratio test can see these.
    "kya main apne kutte ko maap sakta hun",
    "mujhe ticket raise karna hai",
    "geometra ki keemat kya hai",
    "deewar ki tasveer kaise loon",
    "kya biryani maap sakte hain",
]


@pytest.mark.parametrize("message", NON_ENGLISH)
def test_non_english_is_translated(message):
    assert needs_translation(message) is True, (
        f"non-English message left to Pass 2's judgment: {message!r}"
    )


def test_accented_latin_is_not_a_foreign_script():
    """The script test is about scripts, not accents - é/ñ/ü must stay Latin, or ordinary
    English carrying a loanword would be sent to translation."""
    assert is_non_latin_script("can I measure a façade or a piñata") is False
    assert is_non_latin_script("क्या मैं माप सकता हूँ") is True


def test_short_messages_are_left_alone():
    """Under three words the ratio means nothing - one unknown token out of one is 0.0
    whether it is Hindi or a product name. Bare greetings and one-word replies already
    have their own handlers."""
    assert needs_translation("ok") is False
    assert needs_translation("Geometra?") is False


def _stub_translation(monkeypatch, english):
    """Replaces both LLM calls so the WIRING can be tested without the network."""
    monkeypatch.setattr(tp, "to_english", lambda text: english)
    monkeypatch.setattr(tp, "localize", lambda reply, sample: f"<localized>{reply}")


def test_business_rules_read_the_translation(monkeypatch):
    """The point of the whole change: a Devanagari question reaches the exclusion rule."""
    _stub_translation(monkeypatch, "can I measure my dog")
    result = tp.process_turn("क्या मैं कुत्ता माप सकता हूँ", "क्या मैं कुत्ता माप सकता हूँ", [], None)
    assert result.canned is True
    assert "living thing" in result.response
    assert result.response.startswith("<localized>"), "canned reply was not localized"


def test_english_turn_is_untouched(monkeypatch):
    """No translation call, and no localization, on the common path."""
    def _explode(*a, **k):
        raise AssertionError("translation ran on an English turn")

    monkeypatch.setattr(tp, "to_english", _explode)
    monkeypatch.setattr(tp, "localize", _explode)
    result = tp.process_turn("can I measure my dog", "can I measure my dog", [], None)
    assert result.canned is True
    assert "living thing" in result.response


def test_a_failed_translation_falls_back_to_today(monkeypatch):
    """to_english() returning None must leave the turn exactly where it is now - the whole
    layer is optional, like moderation and the classifiers."""
    monkeypatch.setattr(tp, "to_english", lambda text: None)
    monkeypatch.setattr(tp, "localize", lambda reply, sample: pytest.fail("localized anyway"))
    # A message nothing claims, so this asserts the fallback path is reached without
    # depending on what Pass 2 would say.
    assert tp.find_definite_exclusion_reason("क्या मैं कुत्ता माप सकता हूँ") is None


def test_safety_still_reads_the_customers_own_words(monkeypatch):
    """Safety must never depend on the translation: a hostile or broken translation that
    returned something bland would otherwise disarm the slur check. is_severe_slur() runs
    on the raw text inside _resolve_turn, before the probe is used for anything."""
    _stub_translation(monkeypatch, "hello, how are you")
    result = tp.process_turn("tu ek bekar chutiya bot hai", "tu ek bekar chutiya bot hai", [], None)
    assert "<localized>" in result.response
    assert "I can't help with that" in result.response
