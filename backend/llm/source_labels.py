"""One place that decides what a source citation looks like, and removes it.

WHY THIS IS ITS OWN MODULE

Citations were removed on 2026-10-09 and reported as fixed twice, and twice they came back
in front of customers. The investigation on 2026-10-10 found two separate reasons, and
this module exists because of the second one.

The first reason was deployment: every leaking reply was produced by a build that predates
the fix (identifiable because it logs the old `source_citation_accuracy` metric), while
Render had been serving the fixed build since Oct 9. That is not a code problem.

The second reason IS a code problem, and it is why the fix was fragile enough to look
broken: three different places each had their own idea of what a citation looks like, and
all three recognised exactly one spelling - "[Source:", capital S, singular, no inner
space.

    the strip in two_pass.clean_leaked_artifacts   removed it
    CITATION_RE in evaluation/metrics.py            measured it
    the prompt rule                                 forbade it

Measured against fourteen forms a model plausibly writes, the strip removed two. The other
twelve - "[Sources:", "[source:", "[ Source: ]", "(Source: ...)", a bare "Source: X" line,
"**Source:**" - went straight to the customer. Worse, the leak METRIC shared the same blind
spot, so a drifted spelling would not have been logged as a failure either: the monitoring
could not see the thing it was monitoring for.

So the definition lives here once, and everything imports it. If the model invents a
fifteenth spelling, there is exactly one line to change and the strip, the metric and the
tests all move together.

WHY THE TWO PATTERNS ARE DIFFERENT

A bracketed or parenthesised label is unambiguous - no ordinary sentence contains
"[Source: x]" - so it is removed anywhere it appears.

A bare "Source: x" is NOT unambiguous. "Geometra needs a good light source: a window works
well" is a legitimate sentence. So the bare form is only removed when it begins a line or
ends the reply, which is where a citation goes and where that sentence never is.
"""
import re

# Bracketed or parenthesised, anywhere in the text. Tolerates plural, any casing, markdown
# asterisks, and spaces around the brackets and the colon.
_DELIMITED = re.compile(
    r"\s*[\[\(]\s*\*{0,2}\s*sources?\s*\*{0,2}\s*:[^\]\)\n]{0,160}[\]\)]",
    re.IGNORECASE,
)

# Bare, but only as its own line - see the module docstring for why this one is narrow.
_BARE_LINE = re.compile(
    r"(?:^|\n)[ \t]*\*{0,2}\s*sources?\s*\*{0,2}\s*:[^\n]{0,160}",
    re.IGNORECASE,
)


def contains_source_label(text: str) -> bool:
    """True if a customer would see a source citation in this text."""
    return bool(_DELIMITED.search(text or "") or _BARE_LINE.search(text or ""))


def find_source_label(text: str):
    """The matched citation, for logging what leaked and in which form."""
    match = _DELIMITED.search(text or "") or _BARE_LINE.search(text or "")
    return match.group(0).strip() if match else None


def strip_source_labels(text: str) -> str:
    """Removes every source citation, in any of its forms.

    Safe to call more than once and on text that has none - which matters, because it is
    deliberately called at more than one layer (inside Pass 2 handling, and again at the
    single point where a reply leaves the API) rather than trusting any single one.
    """
    if not text:
        return text
    cleaned = _DELIMITED.sub("", text)
    cleaned = _BARE_LINE.sub("", cleaned)
    # Removing a trailing citation can leave a dangling blank line or doubled spaces.
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()
