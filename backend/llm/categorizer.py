"""Assigns an FAQ question to one of the team's existing topic categories.

WHY THIS EXISTS

The category is what the customer sees in the "[Source: ...]" line under an answer. It was
restored to specific per-topic labels on 2026-09-29, because every row had collapsed to a
single generic "Geometra FAQ" and the citation stopped carrying information.

The FAQ spreadsheet the team actually edits has no Category column - it is Sno./Question/
Response/revised response. So the mirror has three choices: lose the labels, ask the team
to categorise ~195 rows by hand and keep doing it forever, or derive the label. This
derives it, with one small isolated call per question, following the same pattern as
llm/measurability.py and llm/printer_classifier.py: a narrow prompt, one job, no retrieved
context, no tone rules.

THE TAXONOMY IS THE TEAM'S, NOT A NEW ONE

CATEGORIES is exactly the eight labels already in use. The model picks from that list and
nothing else, so an automated FAQ cannot quietly invent a ninth label that then shows up in
a customer-facing citation.

COST, AND WHY THE CACHE IS A COMMITTED FILE

A sync re-reads all ~195 rows every time it runs. Classifying all of them per sync would
be absurd, so the mapping is cached by question text and the cache ships in the repo at
data/faq_categories.json. That makes the common sync cost zero calls, makes the labels
reviewable in a diff, and means a label can be corrected by editing one line rather than
re-running a classifier and hoping.

Render's filesystem is ephemeral, so a runtime-learned label does not survive a restart.
That is accepted rather than solved: the only cost is re-classifying the handful of rows
added since the last commit of the cache file, at roughly a cent per hundred. Persisting it
in Supabase would remove that, and would add a second source of truth for something a repo
file already versions better.
"""
import json
import os
import threading
from typing import Dict, Optional

from llm.client import call_llm

# The eight labels the team already uses, verbatim - including the inconsistent casing,
# which is theirs and is what the existing citations show.
CATEGORIES = [
    "How to take pictures",
    "Pasting marker",
    "Printing marker",
    "About geometra marker",
    "About the Geometra website",
    "how to upload",
    "purchasing of geometra",
    # Added 2026-10-08. The eight labels above predate the 101 rows the team added for the
    # app itself, and measured against them 68 of 195 rows (35%) fell into "others" - a
    # citation label that tells the customer nothing. These two cover what was actually
    # missing: the measuring workspace, and getting work back out.
    "Measuring and editing",
    "Exporting and saving",
    "others",
]

# Where an unclassifiable question lands. "others" is one of the team's own labels and
# already holds the genuinely miscellaneous rows, so a fallback is indistinguishable from
# a deliberate choice - which is correct here, since both mean "no better label fits".
FALLBACK = "others"

_CACHE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "faq_categories.json")

_lock = threading.Lock()
_cache: Optional[Dict[str, str]] = None

_PROMPT = """You file a customer-support FAQ question under exactly one topic category.

The categories, and what belongs in each:

  How to take pictures       - taking/capturing photos: lighting, angle, distance, framing,
                               panorama, close-ups, which corners must be visible, retakes
  Pasting marker             - physically attaching the marker to a surface: tape, folds,
                               creases, curved or multi-plane surfaces, damaging the wall,
                               holding it instead of sticking it
  Printing marker            - printing the marker: printer type, paper, GSM, scale,
                               orientation, print shops, where to download it to print
  About geometra marker      - the marker itself as an object: its sizes (A4/A5), choosing
                               a size, what it is for, where to get one
  About the Geometra website - using the website or app: projects, folders, the library,
                               home page, renaming, deleting, downloading, collaboration,
                               accounts, logging in, dialogs and error messages
  how to upload              - getting an image into Geometra: uploading, transferring from
                               phone to computer, messaging apps, file formats, upload
                               failures and incomplete uploads
  purchasing of geometra     - money and plans: price, credits, free plan, billing, buying
  Measuring and editing      - working on a drawing inside Geometra: planes, sections and
                               spaces, selecting and splitting surfaces, drawing or tracing
                               lines and curves, calibration and recalibration, units,
                               correcting a measurement that looks wrong, reviewing before
                               export
  Exporting and saving       - getting work out: export formats, vector vs raster, DXF,
                               PNG/PDF, saving, downloading, what happens to a project
                               after export
  others                     - anything that fits none of the above: accuracy, what the
                               product is, what it can or cannot measure, support contact

Answer with the category name EXACTLY as written above, and nothing else.

If two could fit, choose the one the customer's actual problem is about. "Where can I
download the marker to print it" is Printing marker, not About the Geometra website,
because the problem is printing. "Why did my upload fail" is how to upload, not About the
Geometra website, because the problem is uploading."""

_VALID = {c.lower(): c for c in CATEGORIES}


def _load_cache() -> Dict[str, str]:
    global _cache
    if _cache is None:
        try:
            with open(_CACHE_PATH, "r", encoding="utf-8") as f:
                _cache = json.load(f)
        except (OSError, ValueError):
            # No cache file yet, or it is unreadable. Starting empty is safe - every
            # lookup then classifies, which is the slow path, not a wrong one.
            _cache = {}
    return _cache


def _key(question: str) -> str:
    return " ".join(question.lower().split())


def save_cache() -> None:
    """Writes the cache back, sorted, so a diff of this file reads as a list of labels
    rather than a reshuffle. Best-effort: a read-only filesystem must not fail an ingest."""
    try:
        with _lock:
            data = dict(_load_cache())
        os.makedirs(os.path.dirname(_CACHE_PATH), exist_ok=True)
        with open(_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(dict(sorted(data.items())), f, indent=2, ensure_ascii=False)
            f.write("\n")
    except OSError as e:
        print(f"[categorizer] could not write the category cache: {e}")


def classify_category(question: str) -> str:
    """The category for a question, from the cache when known, else one LLM call.

    Never raises and never returns a label outside CATEGORIES: a failed call or an
    unexpected answer both fall back to "others", because a slightly wrong citation label
    is not a reason to fail an FAQ ingest.
    """
    key = _key(question)
    cache = _load_cache()
    if key in cache:
        return cache[key]

    try:
        raw, _, _ = call_llm(_PROMPT, question)
        answer = _VALID.get(raw.strip().strip(".").lower(), FALLBACK)
    except Exception as e:
        print(f"[categorizer] classification failed for {question[:60]!r}: {e}")
        return FALLBACK

    with _lock:
        cache[key] = answer
    return answer


def seed(mapping: Dict[str, str]) -> int:
    """Pre-loads known question -> category pairs, skipping any label that is not one of
    the team's own. Used to carry the hand-assigned categories across so they are never
    re-derived (and so a classifier can never overrule a human's choice)."""
    cache = _load_cache()
    added = 0
    with _lock:
        for question, category in mapping.items():
            canonical = _VALID.get(str(category).strip().lower())
            if canonical and _key(question) not in cache:
                cache[_key(question)] = canonical
                added += 1
    return added
