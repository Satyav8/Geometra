import os
import sys

BACKEND_DIR = os.path.join(os.path.dirname(__file__), "..", "backend")
sys.path.insert(0, os.path.abspath(BACKEND_DIR))

import pytest  # noqa: E402

from rag.spelling import correct_query  # noqa: E402


@pytest.mark.parametrize("typo,expected_word", [
    ("Can I measure flooors with geometra", "floors"),
    ("Is there a warrenty on the marker", "warranty"),
    ("How acurate is the measurement", "accurate"),
])
def test_corrects_real_typos(typo, expected_word):
    result = correct_query(typo)
    assert expected_word in result.lower()


@pytest.mark.parametrize("query", [
    "Can I measure floors with geometra",
    "What is Geometra?",
    "How much does it cost per wall?",
    "Can I get a refund?",
])
def test_leaves_correct_spelling_alone(query):
    assert correct_query(query) == query


@pytest.mark.parametrize("term", [
    "Is Geometra suitable for my wall?",
    "Do you support DXF export?",
    "Is there WhatsApp support?",
    "What is the A4 marker size?",
    "Can I pay using UPI?",
])
def test_never_mangles_domain_or_short_terms(term):
    # These previously would have been "corrected" into unrelated words:
    # geometra->geometry, dxf->of, whatsapp->None, a4->a, upi->up
    result = correct_query(term)
    assert result == term
