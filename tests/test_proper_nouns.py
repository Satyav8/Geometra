"""Guards the spellchecker against inventing words the customer never typed.

Three production failures on 2026-10-11, one root cause - the checker cannot tell a
proper noun from a typo, and reaches two edits away to find a "correction":

    "Can i meausre humayun's tomb"          -> "Can i measure human's tomb"
    "can I measure Rajesh's office wall"    -> answered about "Radish's office wall"
    "is Geometra ok for my Godrej wardrobe" -> answered about a "Gore wardrobe"

The first was the serious one, and its shape is worth remembering: humayun became human,
so Pass 2 was handed "measure human's tomb" and refused it under the SAFETY rule covering
human remains. The rule was right. The input was fabricated. A customer asking about a
famous monument was told their question was not respectful - and nothing in the safety
layer was at fault.

Two defences, both of which only ever REJECT a correction, so neither can introduce a new
wrong one:
  a leading capital mid-sentence marks a name, place or brand
  a correction must be one edit away; two edits is usually a different word
"""
import pytest

from rag.spelling import correct_query

# Each entry: the message, and the word that must survive untouched.
PROPER_NOUNS = [
    ("can I measure Rajesh office wall", "Rajesh"),
    ("is Geometra ok for my Godrej wardrobe", "Godrej"),
    ("can I measure the Taj Mahal wall", "Mahal"),
    ("does it work in Bengaluru", "Bengaluru"),
    ("I am measuring Priya kitchen cabinet", "Priya"),
    ("my Samsung phone camera is fine right", "Samsung"),
]


@pytest.mark.parametrize("message,name", PROPER_NOUNS)
def test_names_and_brands_are_not_corrected(message, name):
    assert name in correct_query(message), (
        f"{name!r} was rewritten - the customer would be answered about a word they "
        f"never typed: {correct_query(message)!r}"
    )


def test_the_monument_case_end_to_end():
    """The exact message that was refused as being about human remains. "humayun" must
    survive, and the real typo next to it must still be fixed."""
    corrected = correct_query("Can i meausre humayun's tomb")
    assert "humayun" in corrected, f"humayun -> something else: {corrected!r}"
    assert "human's" not in corrected, f"still fabricating 'human': {corrected!r}"
    assert "measure" in corrected, f"the genuine typo stopped being fixed: {corrected!r}"


# Real typos, which must still be corrected - the fix is worthless if it switches
# correction off.
TYPOS = [
    ("can i mesure a wall", "measure"),
    ("what is the warrenty", "warranty"),
    ("the flooors are uneven", "floors"),
    ("Measrue the wall please", "Measure"),       # first word: no capital protection
    ("HOW DO I MEASRUE A WALL", "easure"),        # shouting: protection is disabled
]


@pytest.mark.parametrize("message,expected", TYPOS)
def test_genuine_typos_are_still_fixed(message, expected):
    assert expected.lower() in correct_query(message).lower(), (
        f"correction stopped working: {message!r} -> {correct_query(message)!r}"
    )


def test_two_edit_corrections_are_rejected():
    """Where the damage lived. These are all two edits away, and all turn one word into
    an unrelated one rather than fixing a slip."""
    for message, must_not_appear in [
        ("my Godrej wardrobe", "gore"),
        ("humayun tomb", "human"),
    ]:
        out = correct_query(message).lower()
        assert must_not_appear not in out, f"{message!r} -> {out!r}"


def test_product_identifiers_still_protected():
    """The existing internal-capitals rule must keep working - printer models were being
    rewritten into unrelated words ("LaserJet" -> "Learjet") before it existed."""
    assert "LaserJet" in correct_query("I have an HP LaserJet 1020")
    assert "EcoTank" in correct_query("my Epson EcoTank prints fine")
    assert "zepto" in correct_query("can I get a marker print from zepto")
