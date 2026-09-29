"""Gives the deterministic layers an English view of a non-English message.

WHY THIS EXISTS, AND WHY IT IS NOT TWELVE MORE WORDLISTS

Every code-level business rule in two_pass.py matches English text: the exclusion
categories, the greeting/gratitude/ticket/print-shop caches, the printer gate, and the
measure-intent test that decides whether the measurability classifier runs at all.
Measured against their English equivalents, 9 of 12 checks that answer an English customer
deterministically claim nothing when the same question arrives in Hindi or Hinglish:

  "can I measure my dog"      -> exclusion, "it's a living thing"     (0 LLM calls)
  "क्या मैं अपने कुत्ते को माप सकता हूँ"  -> nothing; Pass 2 decides
  "kya main apne kutte ko maap sakta hun" -> nothing; Pass 2 decides

Pass 2 is genuinely multilingual, so those turns are answered - but by the same 22k-char
prompt that this codebase has repeatedly measured as unreliable at applying its own rules
(see llm/measurability.py). So a Hindi customer is served by the weakest layer in the
system, on exactly the questions where an English customer gets the strongest one.

The obvious fix is to translate the rules: Devanagari exclusion regexes, a Hinglish
greeting list, a romanised ticket pattern. That is the approach this codebase has already
rejected for exclusions, and it is worse here - it does not converge in ONE language, and
India has twenty-two official ones plus a romanised form of each.

So the message is translated once, and the rules stay in English. One small focused call,
the same isolation principle the printer and measurability classifiers are built on: the
model does the one job it is reliably good at, and the deterministic layers keep their
authority over what the answer actually is.

WHAT IS AND IS NOT TRANSLATED

  translated : the text the business rules READ (the "probe")
  never      : the text safety rules read - is_severe_slur() and the moderation API both
               already work across scripts and romanisation, and they run on the raw text
               BEFORE this, so a translation failure can never weaken them
  never      : the question stored on a ticket, which stays in the customer's own words
  never      : what Pass 2 receives - it answers in the customer's language, and handing
               it an English rewrite would make it reply in English

COST

Nothing on English traffic: needs_translation() is a free local test and the call only
happens when it says so. A false positive costs one ~1s call that returns the text
unchanged; a false negative leaves the turn exactly where it is today. Both directions
degrade to current behaviour, which is why this fails open everywhere.
"""
import re
from typing import Optional

from llm.client import call_llm
from rag.spelling import is_dictionary_word

# 0x24F is the end of Latin Extended-B, so accented Latin (é, ñ, ü, ş) still counts as
# Latin - only genuinely different scripts (Devanagari, Telugu, Tamil, Bengali, Gurmukhi,
# Kannada, Malayalam, Odia, Gujarati, Arabic, Cyrillic, CJK, Thai) trip this.
_LATIN_MAX = 0x24F

_WORD_RE = re.compile(r"[A-Za-z']+")

# Romanised Hindi is Latin script, so the script test above is blind to it - "kya main
# deewar maap sakta hun" looks like English to every byte-level check in this file. What
# separates it is that almost none of its words are English words, which is cheap to test
# with the spellchecker vocabulary this codebase already ships.
#
# Measured on English (including deliberately misspelled) against Hinglish:
#
#   worst English : 0.71  "my wal is to smal to mesure"   (typo-ridden, still English)
#   worst Hinglish: 0.60  "geometra marker kaise print karu"
#
# 0.65 sits in that gap, and this deliberately runs on the RAW text. Running it on the
# typo-corrected text was tried first, reasoning that correction would lift misspelled
# English away from the boundary - but it does the same thing to Hinglish, and far more
# damagingly: the spellchecker turns "mujhe ticket raise karna hai" into English-looking
# words and the ratio climbs over the threshold, so the message is never flagged. Measured,
# that single case went from correctly detected to missed. Raw text costs nothing here,
# because misspelled English clears the threshold on its own.
_ENGLISH_WORD_RATIO = 0.65

# Below this, a message is too short for the ratio to mean anything - "kya" is 1 of 1
# non-English words, but so is a bare product name. Short messages are left to the
# existing checks, which already handle bare greetings and one-word replies.
_MIN_WORDS_FOR_RATIO = 3


def is_non_latin_script(text: str) -> bool:
    """True when at least half the letters are outside the Latin range."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return False
    return sum(1 for c in letters if ord(c) > _LATIN_MAX) / len(letters) >= 0.5


def needs_translation(text: str) -> bool:
    """True when the English business rules cannot be expected to read this message.

    Free and local - no LLM call. Deliberately biased towards saying no: a miss costs
    nothing that isn't already being paid today, while a false yes spends a call.
    """
    if is_non_latin_script(text):
        return True
    words = [w for w in _WORD_RE.findall(text) if len(w) > 1]
    if len(words) < _MIN_WORDS_FOR_RATIO:
        return False
    english = sum(1 for w in words if is_dictionary_word(w.lower()))
    return english / len(words) < _ENGLISH_WORD_RATIO


_TO_ENGLISH_PROMPT = """You translate a customer support message into English.

Output ONLY the translation. No quotes, no explanation, no notes, no answer to it.

Rules:
- If the message is already English, output it back unchanged.
- Translate romanised Indian languages too (Hindi, Telugu, Tamil, Marathi, Bengali and
  others written in Latin letters), not only native scripts.
- Keep numbers, measurements, units, currency and product names exactly as written:
  199, Rs, A4, A5, 7m, 125 GSM, Geometra, LaserJet.
- Translate the message as it was actually written, including rudeness, insults or
  profanity. Do not soften it, clean it up, or decline. A blunt, accurate translation is
  required here - safety checks read this text, and a politely rewritten version of an
  abusive message would defeat them.
- Keep the customer's meaning and intent; do not add information or answer the question.
- Use the natural English a customer support desk would use, not a word-by-word rendering.
  "mujhe ticket uthana hai" is "I want to raise a ticket", never "I want to lift a ticket";
  "deewar ki tasveer" is "a photo of the wall". Idioms matter more than literal words - the
  translation is read by rules written for how English speakers actually phrase things."""


def to_english(text: str) -> Optional[str]:
    """The message in English, or None if the call failed or returned nothing usable.

    None means "carry on with the original text" - i.e. exactly the behaviour this module
    was added to improve on, never worse than it.
    """
    try:
        raw, _, _ = call_llm(_TO_ENGLISH_PROMPT, text)
    except Exception as e:
        print(f"[translation] to_english failed, using the original text: {e}")
        return None
    translated = raw.strip().strip('"').strip()
    return translated or None


_LOCALIZE_PROMPT = """You translate a support reply into the language the customer wrote in.

You are given the customer's message and a reply. Output ONLY the reply, rewritten in the
same language AND the same script the customer used - if they wrote Hindi in Latin letters,
answer in Hindi in Latin letters; if they wrote Devanagari, answer in Devanagari.

Rules:
- Keep every number, price, measurement, unit and product name exactly as written:
  199, Rs, A4, A5, 7m, 125 GSM, Geometra, LaserJet.
- Keep the meaning exact. Do not add, remove, soften or explain anything.
- Keep the same tone: warm, plain and direct.
- If the customer's message is already English, output the reply unchanged."""


def localize(reply: str, customer_message: str) -> str:
    """The canned English `reply`, rendered in the language `customer_message` is written in.

    Falls back to the English reply on any failure: a customer reading an English answer is
    a worse experience than a translated one, but a customer reading nothing is a bug.
    """
    if not reply.strip():
        return reply
    try:
        raw, _, _ = call_llm(
            _LOCALIZE_PROMPT,
            f"CUSTOMER MESSAGE:\n{customer_message}\n\nREPLY TO TRANSLATE:\n{reply}",
        )
    except Exception as e:
        print(f"[translation] localize failed, sending the English reply: {e}")
        return reply
    translated = raw.strip()
    return translated or reply
