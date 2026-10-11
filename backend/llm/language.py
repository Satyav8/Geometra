"""Decides what language a customer wrote in, and therefore what language to answer in.

WHY A CLASSIFIER AND NOT A WORDLIST

The first version of this was lexical: count how many words are in an English dictionary,
call it non-English below a threshold. It broke in both directions on 2026-10-11.

  "Can i meausre humayun's tomb"   English, answered in Hinglish
  "Kya mei iske saat diwaron..."   Hinglish, answered in English

The first is the detector's fault. A typo ("meausre") and a proper noun ("humayun's") are
two unknown words out of four, which was enough to call a plain English question foreign.
Names, places, product words and typos are ordinary content, not evidence of a language.

Tuning the threshold cannot fix it. Measured across both populations:

    minimum unknown words   English wrongly flagged   Hinglish missed
              2                     4/16                   0/14
              3                     1/16                   3/14
              4                     0/16                   9/14

Every setting trades one error for the other, because short Hinglish and typo-heavy
English look the same to a dictionary.

Nor can the spellchecker separate them, which was the obvious next idea - Hindi words
correct to English words exactly as readily as typos do:

    meausre -> measure (1)      kahan -> khan  (1)
    halp    -> help    (1)      maap  -> map   (1)
    mch     -> much    (1)      kaise -> raise (1)

So the lexical approach is finished, and this is the same shape as two problems this
codebase already solved - printer type and measurability - where enumeration could not
converge and one isolated question could. Language identification is something a model is
genuinely reliable at, and neither typos nor proper nouns trouble it.

COST

The classifier runs only on the ambiguous middle. Clearly-English text (the overwhelming
majority) and non-Latin script are both decided locally for free; only messages that the
cheap check cannot call confidently cost an LLM round trip.

WHAT IT IS USED FOR

Two things that were previously guessed:
  - whether to translate a message for the English business rules (llm/translation.py)
  - what language Pass 2 must answer in, which the prompt never actually specified -
    every non-English answer until now was the model mirroring by luck
"""
import re
from typing import Optional

from llm.client import call_llm
from rag.spelling import is_dictionary_word

_WORD_RE = re.compile(r"[A-Za-z']+")
_LATIN_MAX = 0x24F

# Above this share of dictionary words, a Latin-script message is English and no call is
# made. Deliberately high: the cost of being wrong here is the bug this module exists for,
# while the cost of asking is one small call.
_CONFIDENT_ENGLISH_RATIO = 0.80

# Below this, with enough words to mean anything, it is confidently NOT English - the
# classifier would only confirm what the dictionary already shows.
_CONFIDENT_FOREIGN_RATIO = 0.35
_MIN_WORDS = 3

ENGLISH = "english"

_PROMPT = """You identify what language a customer support message is written in.

Answer with ONE word, lowercase, from exactly this list:
  english    plain English, including misspelled English, English with names, places,
             brand names, product codes or abbreviations in it
  hinglish   Hindi (or another Indian language) written in Latin letters
  hindi      Hindi written in Devanagari script
  other      any other language, in any script

Judge the LANGUAGE, not the spelling. A message full of typos is still english. A message
containing an Indian name, city or brand is still english if the sentence itself is
English - "Can i meausre humayun's tomb" is english, "kya main deewar maap sakta hun" is
hinglish.

Answer with the single word and nothing else."""

_VALID = {ENGLISH, "hinglish", "hindi", "other"}


def is_non_latin_script(text: str) -> bool:
    """True when at least half the letters are outside the Latin range."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return False
    return sum(1 for c in letters if ord(c) > _LATIN_MAX) / len(letters) >= 0.5


def _english_word_ratio(text: str):
    words = [w for w in _WORD_RE.findall(text) if len(w) > 1]
    if not words:
        return None, 0
    known = sum(1 for w in words if is_dictionary_word(w.lower()))
    return known / len(words), len(words)


def identify(text: str) -> str:
    """The language of this message: "english", "hinglish", "hindi" or "other".

    Costs nothing for the common cases and one small call for the ambiguous middle.
    Any failure answers "english", which is the safe default: it means no translation and
    no language instruction, i.e. exactly the behaviour before this module existed.
    """
    if not text or not text.strip():
        return ENGLISH

    if is_non_latin_script(text):
        # No call: the script already settles everything that matters. Naming the
        # specific language would only be useful if the instruction differed per
        # language, and it does not - "answer in the same language and script the
        # customer used" covers Devanagari, Telugu, Tamil, Bengali and the rest.
        #
        # Asking was also measurably WRONG: the classifier labelled Telugu and Tamil as
        # "hindi", which would have told Pass 2 to answer a Telugu speaker in Devanagari.
        return "other"

    ratio, word_count = _english_word_ratio(text)
    if ratio is None or word_count < _MIN_WORDS:
        return ENGLISH
    if ratio >= _CONFIDENT_ENGLISH_RATIO:
        return ENGLISH
    if ratio <= _CONFIDENT_FOREIGN_RATIO:
        return _ask(text, fallback="hinglish")
    # The middle, where a dictionary genuinely cannot tell a typo from another language.
    return _ask(text, fallback=ENGLISH)


def _ask(text: str, fallback: str) -> str:
    try:
        raw, _, _ = call_llm(_PROMPT, text)
    except Exception as e:
        print(f"[language] identification failed, assuming {fallback}: {e}")
        return fallback
    answer = raw.strip().lower().strip(".").split()[0] if raw.strip() else ""
    return answer if answer in _VALID else fallback


def needs_translation(language: str) -> bool:
    """Whether the English business rules need a translated view of this message."""
    return language != ENGLISH


def answer_language_instruction(language: str) -> Optional[str]:
    """The instruction telling Pass 2 what language to answer in, or None for English.

    This exists because the Pass 2 prompt never said anything about language at all -
    "same language", "respond in", "reply in" appear nowhere in it. Non-English answers
    were the model mirroring the customer of its own accord, which it mostly did and
    sometimes did not: "Kya mei iske saat diwaron ko maap sakta hun?" came back in
    English on 2026-10-11.
    """
    if language == ENGLISH:
        return None
    described = {
        "hinglish": "Hindi written in Latin letters (Hinglish), the same way the customer "
                    "wrote - do NOT reply in Devanagari script and do NOT reply in English",
        # Reachable only if the classifier returns "hindi" for Latin-script input;
        # Devanagari input short-circuits to "other" above without a call.
        "hindi": "Hindi in Devanagari script, the same script the customer used",
        "other": "the same language AND the same script the customer wrote in",
    }[language]
    return (
        f"\nNOTE: the customer wrote in {language}. Write your entire answer in "
        f"{described}. Keep numbers, prices, measurements and product names exactly as "
        f"they are (199, A4, 125 GSM, Geometra, DXF) - translate the words around them, "
        f"not those.\n"
    )
