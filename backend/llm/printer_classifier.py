"""Focused printer-type classifier.

Rule 8E's policy depends on knowing whether a named printer model is laser, inkjet or dot
matrix. Pass 2 is allowed to work that out from general knowledge, but measured against 12
real models common in the Indian market it only managed 10/12 - on the Epson FX-890 and the
Canon imageCLASS LBP2900 it answered "I don't have specific information about that model".
That is Rule 8E-compliant ("say so rather than guessing") but useless to the customer, and
the consequence is real: printing the marker on a dot matrix produces a failed measurement,
not just a poor experience.

The task itself isn't hard - it was just competing with the entire 22k-character Pass 2
prompt. Asked in isolation, the same model gets it right: measured 15/15 correct across
three runs each (12 real models plus three controls), fully self-consistent, at ~1.3s per
call versus ~5s for the Pass 2 path it replaces.

Deliberately NOT a lookup table of model prefixes. A hardcoded series list would need
permanent maintenance and would still miss models, whereas this is the model's own world
knowledge asked cleanly. The three control cases confirm it doesn't invent an answer to
look useful: a message with no printer named, an invented model ("Zorblax QT-9900") and an
ordinary marker question all returned "unknown" on every run.
"""
from typing import Optional

from llm.client import call_llm

CLASSIFIER_PROMPT = """You identify printer hardware types.

The user message may mention a printer model. Reply with EXACTLY one lowercase word and
nothing else:

laser      - the model is a laser printer
inkjet     - the model is an inkjet printer (including ink tank / EcoTank style)
dotmatrix  - the model is a dot matrix / impact printer
unknown    - no printer model is named, or you genuinely cannot identify the model

Do not explain. Do not add punctuation. One word only."""

_VALID = {"laser", "inkjet", "dotmatrix", "unknown"}


def classify_printer_type(message: str) -> Optional[str]:
    """Returns "laser" / "inkjet" / "dotmatrix" / "unknown", or None if the call failed.

    None and "unknown" are handled differently by the caller: "unknown" is a real answer
    (the model genuinely couldn't identify it, so the customer gets the explain-the-three-
    types reply), while None means the call itself broke and the turn should fall through
    to Pass 2 exactly as it did before this module existed - the same fail-open convention
    as llm/moderation.py.
    """
    try:
        raw, _, _ = call_llm(CLASSIFIER_PROMPT, message)
    except Exception as e:
        print(f"[printer] classification failed, falling through to Pass 2: {e}")
        return None
    answer = raw.strip().lower().strip(".")
    return answer if answer in _VALID else "unknown"
