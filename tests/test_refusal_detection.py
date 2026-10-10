"""Guards refuses_the_subject(), which decides whether a wrong refusal gets retried.

WHY THIS MATTERS MORE THAN IT LOOKS

This is not only a test helper. process_turn() uses it to decide whether Pass 2 has
contradicted a "measurable" verdict and must be asked again. If it fails to recognise a
refusal, the retry never fires and the whole measurable-verdict fix silently does nothing
on that turn.

That is exactly what happened. The first version cancelled a denial whenever an
affirmative appeared ANYWHERE later in the reply, so this:

    "No, Geometra cannot measure a staircase wall. The staircase itself is excluded
     from measurement, and while the wall next to it can be measured..."

was read as "not a refusal" - the aside about a different wall overrode the verdict. The
fix measured 7/7 locally and then failed in production on precisely these replies, because
the retry was never reached.

The rule now is the first sentence, because English puts the verdict there, and a genuine
"no, but" keeps both halves in one sentence.
"""
import pytest

from llm.two_pass import refuses_the_subject

# Replies that refuse the thing the customer asked about. The last three are the
# production answers that the previous version let through.
REFUSALS = [
    "Unfortunately, Geometra isn't able to measure that since it's a living thing.",
    "No, Geometra cannot measure the staircase.",
    "Geometra cannot measure a swimming pool, as it is a liquid feature.",
    "No, Geometra cannot measure a staircase wall. The staircase itself is excluded from "
    "measurement, and while the wall next to it can be measured, this one cannot.",
    "No, Geometra cannot measure a wall that has a mirror on it. Mirrors are considered "
    "reflective surfaces, which fall under the exclusions. You can measure other walls.",
    "No, Geometra is designed specifically for measuring interior walls. Since your "
    "garage wall is external, it cannot be measured.",
]


@pytest.mark.parametrize("reply", REFUSALS)
def test_a_refusal_is_recognised(reply):
    assert refuses_the_subject(reply) is True, (
        "a refusal was not recognised, so the measurable-verdict retry would not fire"
    )


# Replies that ANSWER. The first is the one that matters most: it denies something, then
# affirms the actual subject in the same sentence. Treating it as a refusal would trigger
# a pointless retry on a reply that is already correct and helpful.
ANSWERS = [
    "Geometra cannot measure the staircase itself, but you can measure the wall next to it.",
    "Yes, you can measure the wall as long as at least three corners are visible.",
    "Geometra can measure the wall as long as it meets the requirements. However, the "
    "mirror itself cannot be measured.",
    "It sounds like the vases are blocking the corners. As long as three corners are "
    "visible, you can measure the wall.",
    "To measure a wall, place the marker flat against the surface and take one photo.",
]


@pytest.mark.parametrize("reply", ANSWERS)
def test_an_answer_is_not_mistaken_for_a_refusal(reply):
    assert refuses_the_subject(reply) is False, "an answer was misread as a refusal"


def test_the_whole_reply_is_not_scanned_for_affirmatives():
    """The precise regression. A denial in sentence one and an affirmative in sentence
    three is a refusal; the old version called it an answer."""
    reply = ("No, Geometra cannot measure that wall. Reflective surfaces are excluded. "
             "Other walls in the room can be measured normally.")
    assert refuses_the_subject(reply) is True


def test_a_contrast_inside_one_sentence_is_an_answer():
    """The mirror image of the case above, and why the rule is per-sentence rather than
    "ignore affirmatives entirely"."""
    reply = "Geometra cannot measure the mirror, but you can measure the wall behind it."
    assert refuses_the_subject(reply) is False


def test_empty_and_short_replies_are_safe():
    assert refuses_the_subject("") is False
    assert refuses_the_subject("Yes") is False
    assert refuses_the_subject("No, that cannot be measured") is True
