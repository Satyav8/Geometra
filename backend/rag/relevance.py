import re

# Tightened to Geometra-specific terms only. Generic words that used to be here
# ("data", "app", "web", "browser", "room", "image", "price", "cost", "sign", "png")
# were dropped because they false-positive on totally unrelated questions — precision
# matters more than recall here since this list also gates what reaches the team's
# escalation queue (Supabase).
FAQ_KEYWORDS = {
    "geometra", "wall", "walls", "elevation", "elevations", "marker", "markers",
    "photo", "photos", "measure", "measurement", "measurements",
    "accuracy", "accurate", "pricing", "refund", "privacy",
    "signup", "onboarding", "tutorial", "hardware", "printer",
    "dxf", "cad", "scan", "verify", "verification",
    "survey", "surveyor", "fabrication", "floor", "floors", "ceiling", "ceilings",
    "corner", "corners", "laser", "tape", "camera", "flash", "lighting",
    "shadow", "shadows", "whatsapp", "payment", "studio", "offline",
    "curved", "arch", "arches", "tilt", "tilted", "resolution",
}


def _contains_word(text: str, phrase: str) -> bool:
    """Word-boundary match — plain substring matching lets short/common words
    (e.g. "ty" inside "warranty", "cad" inside "arcade") false-positive."""
    return re.search(r"\b" + re.escape(phrase) + r"\b", text) is not None


def is_query_relevant(query: str) -> bool:
    query_lower = query.lower()
    return any(_contains_word(query_lower, kw) for kw in FAQ_KEYWORDS)


def compute_criticality(similarity_score: float) -> str:
    """How big a knowledge-base gap this represents — lower similarity to anything
    already in the FAQ means a more novel topic, which the team should prioritize."""
    if similarity_score < 0.10:
        return "high"
    if similarity_score < 0.20:
        return "medium"
    return "low"


GRATITUDE_PHRASES = {
    "thank you", "thanks", "thank u", "thankyou", "thanx", "ty",
    "much appreciated", "appreciate it", "appreciate you",
}


def is_gratitude(query: str) -> bool:
    query_lower = query.lower().strip().strip("!.,")
    return any(_contains_word(query_lower, phrase) for phrase in GRATITUDE_PHRASES)


# A fixed set of exact strings ("hi", "hey", "heyy"...) kept missing casual
# spelling variants one letter at a time ("heyaaa", "hiiii", "helloooo") - matched
# by pattern instead so any amount of letter-stretching on a known greeting root
# is covered, not just the handful of spellings someone thought to list.
_GREETING_RE = re.compile(
    r"^(?:h+i+|h+e+y+a*|h+e+l+o+|good\s+(?:morning|afternoon|evening))(?:\s+there)?$"
)


def is_greeting(query: str) -> bool:
    """Whole-message match only (unlike is_gratitude's "contains anywhere") - a greeting
    is much more likely than "thanks" to open a message that also has a real question in
    it (e.g. "Hi, how much does it cost?"), so only short-circuit when the WHOLE message
    is just a greeting, otherwise the real question would never get answered."""
    normalized = query.lower().strip().strip("!.,")
    return bool(_GREETING_RE.match(normalized))
