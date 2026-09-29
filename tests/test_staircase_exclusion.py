"""Guards the staircase exclusion, and the word that nearly broke it.

Staircases are not measurable (confirmed 29-09-2026). The exclusion originally matched
"steps?" too, which looked reasonable and was badly wrong: in this product "steps" almost
always means INSTRUCTIONS. Because the category only fires when the message also contains
"measure", the collision landed on exactly the phrasing customers use to ask how to measure
something - a real customer asking "I want to clearly measure a wall help with the steps to
do that" was told Geometra cannot measure staircases.

Six variants were refused before the fix. "steps" is gone from the regex; a genuine "can I
measure the steps" reaches the measurability classifier instead, which returns its own
stairs verdict (verified).

Same shape as every collision this project has hit - "dude" in "man cave", "randwa" in
"marker and wall", "asses" in "assess", "kill" in "kill the glare". A word that reads as
unambiguous in isolation turns out to be common ordinary vocabulary in context.

Pure local regex checks - no LLM, no network.
"""
import pytest

from llm.two_pass import find_definite_exclusion_reason

STAIRS = [
    "can I measure the staircase",
    "can we measure the staircase?",
    "can I measure stairs",
    "can I measure a stairway",
    "can I measure a stairwell",
    "can I measure a flight of stairs",
    "can we measure staircases",
]


@pytest.mark.parametrize("message", STAIRS)
def test_staircase_is_excluded(message):
    reason = find_definite_exclusion_reason(message)
    assert reason is not None, f"staircase question not excluded: {message!r}"
    assert "staircase" in reason, f"excluded for the wrong reason: {reason!r}"


# "steps" as instructions. Every one of these was refused as a staircase before the fix.
INSTRUCTIONS = [
    "I want to clearly measure a wall help with the steps to do that",
    "what are the steps to measure a wall",
    "can you give me step by step instructions to measure",
    "measure a wall - what are the steps involved",
    "steps to measure my bedroom wall",
    "how do I measure a wall, step by step",
    "what steps should I follow to measure a wardrobe",
]


# A staircase WALL is an ordinary measurable wall, and "the wall next to the staircase" is
# one of the most natural ways to describe one. Both were refused, because the category
# fires on the word appearing anywhere rather than on it being the subject. When a surface
# Rule 8 covers is also named, the regex defers to the measurability classifier.
WALL_BESIDE_STAIRS = [
    "can I measure the wall next to the staircase",
    "can I measure a staircase wall",
    "can I measure the wall along the stairway",
    "can I measure the ceiling above the stairs",
]


@pytest.mark.parametrize("message", WALL_BESIDE_STAIRS)
def test_a_wall_beside_stairs_is_not_refused_as_a_staircase(message):
    assert find_definite_exclusion_reason(message) is None, (
        f"an ordinary wall was refused because stairs were mentioned: {message!r}"
    )


@pytest.mark.parametrize("message", INSTRUCTIONS)
def test_steps_meaning_instructions_is_not_a_staircase(message):
    assert find_definite_exclusion_reason(message) is None, (
        f"a request for instructions was refused as a staircase: {message!r}"
    )
