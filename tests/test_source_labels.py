"""Guards the citation strip against every form, not just the one that was reported.

HISTORY, BECAUSE IT IS THE WHOLE POINT OF THIS FILE

Citations were removed on 2026-10-09 and declared fixed twice. They reached customers
again both times. The investigation on 2026-10-10 found two causes.

One was deployment: the leaking replies came from a build predating the fix, identified by
the old metric name stamped on them. No test can prevent that, which is why /health now
reports the running commit.

The other was this: three places each defined a citation independently, and all three
recognised exactly one spelling - "[Source:", capital S, singular, no inner space. The
strip removed 2 of 14 plausible forms. The leak METRIC had the same blind spot, so a
drifted spelling would not even have been logged. And verification used a BROADER pattern
than the strip, so "0 leaks in 45 replies" meant "those 45 did not drift", while being
reported as "drift is handled".

So these tests assert the fourteen forms directly. A model that invents a new one should
break this file before it reaches a customer.
"""
import pytest

from llm.source_labels import (
    contains_source_label,
    find_source_label,
    strip_source_labels,
)

# Every form a model plausibly writes. The plural ones are not hypothetical: the prompt
# rule that created this behaviour literally said "If multiple sections are used, cite
# all", which is an invitation to write "Sources:".
CITATION_FORMS = [
    "The price is 199. [Source: Pricing]",
    "The price is 199. [Source: Pasting marker, About geometra marker]",
    "The price is 199. [Sources: Pricing]",
    "The price is 199. [Sources: Pricing, Accuracy]",
    "The price is 199. [source: Pricing]",
    "The price is 199. [SOURCE: Pricing]",
    "The price is 199. [ Source: Pricing ]",
    "The price is 199. [Source : Pricing]",
    "The price is 199. (Source: Pricing)",
    "The price is 199. (Sources: Pricing)",
    "The price is 199.\nSource: Pricing",
    "The price is 199.\nSources: Pricing, Accuracy",
    "The price is 199.\n**Source:** Pricing",
    "The price is 199.\n*Source: Pricing*",
    # the empty label actually observed in production logs on 2026-10-10
    "The price is 199. [Source: ]",
]


@pytest.mark.parametrize("text", CITATION_FORMS)
def test_every_form_is_stripped(text):
    cleaned = strip_source_labels(text)
    assert "source" not in cleaned.lower(), f"citation survived the strip: {cleaned!r}"
    assert "199" in cleaned, "the strip removed the actual answer"


@pytest.mark.parametrize("text", CITATION_FORMS)
def test_every_form_is_detected(text):
    assert contains_source_label(text) is True
    assert find_source_label(text), "nothing reported for logging"


# The cost of a broad pattern is over-stripping. These are sentences a support bot could
# legitimately write, where the word "source" is content rather than a citation.
LEGITIMATE = [
    "Geometra needs a good light source: a window works well for even lighting.",
    "The marker is the reference source for every measurement.",
    "Check that your light source is not creating glare on the wall.",
    "Where did you source the paper from?",
    "The accuracy depends on the source image resolution.",
]


@pytest.mark.parametrize("text", LEGITIMATE)
def test_ordinary_sentences_are_untouched(text):
    assert strip_source_labels(text) == text, "legitimate text was mangled"
    assert contains_source_label(text) is False


def test_a_trailing_citation_does_not_leave_a_dangling_blank_line():
    """The visible symptom if the strip is careless - the answer ends in whitespace and
    the bubble renders with an empty gap at the bottom."""
    cleaned = strip_source_labels("Here is the answer.\n\n[Source: Pricing]")
    assert cleaned == "Here is the answer."


def test_multiple_citations_in_one_reply_all_go():
    text = "First point. [Source: A] Second point. (Sources: B, C)\nSource: D"
    cleaned = strip_source_labels(text)
    assert "source" not in cleaned.lower()
    assert "First point." in cleaned and "Second point." in cleaned


def test_stripping_is_idempotent():
    """It is deliberately applied at more than one layer, so running twice must be safe."""
    once = strip_source_labels("The price is 199. [Source: Pricing]")
    assert strip_source_labels(once) == once


def test_empty_and_missing_text_are_safe():
    assert strip_source_labels("") == ""
    assert strip_source_labels(None) is None
    assert contains_source_label("") is False


def test_the_metric_and_the_strip_share_one_definition():
    """The blind spot that let this recur unlogged: the metric recognised less than the
    strip removed, so a drifted spelling was invisible in production logging."""
    from evaluation import metrics

    for text in CITATION_FORMS:
        result = metrics.citation_not_leaked(text, [])
        assert result.passed is False, f"metric missed a form the strip removes: {text!r}"
    for text in LEGITIMATE:
        assert metrics.citation_not_leaked(text, []).passed is True
