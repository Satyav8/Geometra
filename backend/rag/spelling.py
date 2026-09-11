import re

from spellchecker import SpellChecker

# Product/technical terms that must never be "corrected" into an unrelated real word
# (verified against pyspellchecker's actual suggestions: geometra->geometry, dxf->of,
# a4/a5/a3->a, upi->up, aruco->truck, whatsapp->None, zepto->kept, blinkit->blanket).
# The zepto/blinkit case was a real production bug, not a theoretical one: a customer
# asking "can I get a marker print from zepto" got "zepto" silently corrected to "kept"
# before the LLM ever saw the query, which then answered a fabricated question about a
# nonexistent "Kept" product/feature instead of surfacing the FAQ's actual, correct
# guidance (avoid quick-commerce platforms for printing) - a hallucination caused entirely
# by this correction step, not the LLM's own judgment.
DOMAIN_WORDS = {
    "geometra", "aruco", "whatsapp", "dxf", "chromadb", "qdrant",
    "sendgrid", "supabase", "groq", "sam",
    "zepto", "blinkit", "instamart",
    # Printer brand/series names, for customers who type them in lowercase (the
    # internal-capitals rule below covers them when typed normally). Printing the marker
    # is a required step, so these come up constantly.
    "laserjet", "deskjet", "officejet", "ecotank", "pixma", "imageclass",
    "smarttank", "inktank", "epson", "canon", "brother", "kyocera", "ricoh",
    # Ordinary software vocabulary the dictionary doesn't know, which correction turns
    # into unrelated - and sometimes censored - words. "logins" became "loins", which is
    # in the profanity list, so "do we need separate logins" was flagged as abuse.
    "login", "logins", "signup", "signin", "signout", "logout", "username",
    "screenshot", "screenshots", "upload", "uploads", "uploading", "downloadable",
    "workspace", "workspaces", "webapp", "browser", "wifi", "megapixel",
    "megapixels", "rescan", "rescans", "reupload", "onboarding", "walkthrough",
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
    # A token carrying capitals anywhere but the first character is a product identifier,
    # not a typo - "LaserJet", "EcoTank", "PIXMA", "DeskJet", "imageCLASS". Left alone,
    # the checker rewrote real printer models into unrelated dictionary words and the LLM
    # answered about those instead: "HP LaserJet 1020" reached it as "HP Learjet 1020" (a
    # private jet) and it replied that it couldn't confirm whether a Learjet was a laser
    # printer; "Epson EcoTank" became "Ecotone", "Canon PIXMA" became "Pima". Same class
    # of bug as the zepto->kept case above, and unfixable by wordlist alone since no list
    # can enumerate every printer model a customer might own.
    #
    # Skipped only when the message isn't entirely uppercase, so a customer typing in
    # caps ("HOW DO I MEASRUE A WALL") still gets their genuine typos corrected.
    shouting = text.isupper()
    def _is_product_identifier(token: str) -> bool:
        return not shouting and any(c.isupper() for c in token[1:])

    tokens = _TOKEN_RE.findall(text)
    protected = {t.lower() for t in tokens if _is_product_identifier(t)}
    candidates = {
        t.lower() for t in tokens
        if len(t) >= MIN_LENGTH_TO_CORRECT and t.lower() not in protected
    }
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


def is_dictionary_word(word: str) -> bool:
    """True when this is an ordinary English word. Used by the profanity elongation check:
    a letter-stretching evasion ("bitchhhh", "niggggga") is never a real word, so a match
    that IS one is a collision rather than abuse."""
    return bool(_spell.known([word.lower()]))


def has_no_correction_candidates(word: str) -> bool:
    """True when pyspellchecker can't find ANY known word close to this one - a much
    stronger signal of pure gibberish (keyboard-mash like "ejfnlefnse") than merely being
    "unknown", since a real typo (e.g. "reciept") always has an obvious top candidate."""
    lower = word.lower()
    if _spell.known([lower]):
        return False
    return not _spell.candidates(lower)
