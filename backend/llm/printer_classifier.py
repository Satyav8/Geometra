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
ordinary marker question all returned "unknown" on every run (now "notprinter" - see
classify_printer_type for why that distinction earns its keep).
"""
from typing import Optional

from llm.client import call_llm

CLASSIFIER_PROMPT = """You identify printer hardware types.

The user message may mention a printer model. Reply with EXACTLY one lowercase word and
nothing else:

laser      - the model is a laser printer
inkjet     - the model is an inkjet printer (including ink tank / EcoTank style)
dotmatrix  - the model is a dot matrix / impact printer
unknown    - a printer is named, but you genuinely cannot identify which type it is
notprinter - the message does not name any printer hardware at all

Do not explain. Do not add punctuation. One word only."""

_VALID = {"laser", "inkjet", "dotmatrix", "unknown", "notprinter"}


def classify_printer_type(message: str) -> Optional[str]:
    """Returns "laser"/"inkjet"/"dotmatrix"/"unknown"/"notprinter", or None if the call
    failed.

    The caller treats all four non-type answers differently:

      "unknown"    a printer IS named but couldn't be identified - the customer gets the
                   explain-the-three-types reply, which is still useful to them.
      "notprinter" no printer in the message at all - falls through to Pass 2 untouched.
                   This separation is what lets the routing gate be generous: a message
                   that merely looks like it names hardware costs one small call and then
                   behaves exactly as it did before, instead of getting printer guidance
                   it never asked for.
      None         the call itself broke - falls through to Pass 2, the same fail-open
                   convention as llm/moderation.py.
    """
    try:
        raw, _, _ = call_llm(CLASSIFIER_PROMPT, message)
    except Exception as e:
        print(f"[printer] classification failed, falling through to Pass 2: {e}")
        return None
    answer = raw.strip().lower().strip(".")
    return answer if answer in _VALID else "notprinter"
