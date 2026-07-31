import re

from spellchecker import SpellChecker

# Product/technical terms that must never be "corrected" into an unrelated real word
# (verified against pyspellchecker's actual suggestions: geometra->geometry, dxf->of,
# a4/a5/a3->a, upi->up, aruco->truck, whatsapp->None).
DOMAIN_WORDS = {
    "geometra", "aruco", "whatsapp", "dxf", "chromadb", "qdrant",
    "sendgrid", "supabase", "groq", "sam",
}

# Below this length, corrections are more likely to mangle a legitimate short word/
# acronym than fix a real typo (pdf->pd, png->pig, app->ape, faq->far, otp->top, a4->a
# were all observed on words this short). Real typos we actually need to catch
# ("flooors", "warrenty", "acuracy") are comfortably longer than this.
MIN_LENGTH_TO_CORRECT = 5

_spell = SpellChecker()
_spell.word_frequency.load_words(DOMAIN_WORDS)

_TOKEN_RE = re.compile(r"[A-Za-z']+")


def correct_query(text: str) -> str:
    """Fixes obvious typos before retrieval/relevance matching so a misspelled word
    doesn't tank the embedding similarity or miss a keyword match. Conservative by
    design: only touches words the checker doesn't recognize, skips short words
    (protects acronyms/codes), and never touches known product/technical terms."""
    tokens = _TOKEN_RE.findall(text)
    candidates = {t.lower() for t in tokens if len(t) >= MIN_LENGTH_TO_CORRECT}
    if not candidates:
        return text

    unknown = _spell.unknown(candidates)
    if not unknown:
        return text

    corrected = text
    for word in tokens:
        lower = word.lower()
        if lower not in unknown:
            continue
        suggestion = _spell.correction(lower)
        if not suggestion or suggestion == lower:
            continue
        replacement = suggestion.capitalize() if word[:1].isupper() else suggestion
        corrected = re.sub(rf"\b{re.escape(word)}\b", replacement, corrected, count=1)

    return corrected
