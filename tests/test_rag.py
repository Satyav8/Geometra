import pytest

from rag.relevance import compute_criticality, is_gratitude, is_query_relevant
from rag.retriever import retrieve

# 20 test queries covering all FAQ sheet categories + unknown/out-of-scope questions.
# Categories come from the live Google Sheet (see rag/ingestor.py), not the old static file.
TEST_QUERIES = [
    ("What is Geometra?", "Basic"),
    ("How does the measurement process work step by step?", "Basic"),
    ("Do I need any special hardware or a laser scanner?", "Basic"),
    ("Do I have to download an app to use Geometra?", "Basic"),
    ("What file formats do I get as output?", "Basic"),
    ("Is Geometra a floor-plan tool or an elevations tool?", "Basic"),
    ("How accurate is Geometra in millimeters?", "Basic"),
    ("Has Geometra been tested against a laser measure?", "Basic"),
    ("What's the largest wall Geometra can measure?", "Basic"),
    ("What is the marker and why do I need it?", "Marker"),
    ("How do I print the marker correctly?", "Marker"),
    ("Can I reuse the same marker multiple times?", "Marker"),
    ("Where do I place the marker on the wall?", "Marker"),
    ("How many corners of the wall need to be visible in the photo?", " Capturing the Photo"),
    ("Can I measure a whole room at once?", " Capturing the Photo"),
    ("How much does Geometra cost per wall?", " Pricing & Plans"),
    ("What does the free plan include?", " Pricing & Plans"),
    ("Can I get a refund if a scan fails?", " Pricing & Plans"),
    ("Is my uploaded photo data stored securely?", " Data Privacy & Security"),
    ("How do I sign up for Geometra?", " Getting Started & Support"),
]

UNKNOWN_QUERIES = [
    "What's the weather like in Mumbai today?",
    "Can you recommend a good pizza recipe?",
]


@pytest.mark.parametrize("query,expected_section", TEST_QUERIES)
def test_retrieval_returns_expected_section(query, expected_section):
    chunks, confidence_level = retrieve(query)
    assert len(chunks) > 0
    assert confidence_level in ("high", "low")
    sections = [c.section.strip() for c in chunks]
    assert expected_section.strip() in sections


@pytest.mark.parametrize("query", UNKNOWN_QUERIES)
def test_out_of_domain_query_is_low_confidence_or_unknown(query):
    chunks, confidence_level = retrieve(query)
    assert confidence_level in ("low", "unknown")


@pytest.mark.parametrize("query", ["Thank you", "thanks!", "Okay thank you", "thanks so much", "TY"])
def test_gratitude_detected(query):
    assert is_gratitude(query) is True


@pytest.mark.parametrize("query", [
    "What is Geometra?",
    "How much does it cost?",
    "Does the marker come with a warranty against fading over time?",
    "Can I get a certainty guarantee on accuracy?",
])
def test_gratitude_not_falsely_detected(query):
    assert is_gratitude(query) is False


@pytest.mark.parametrize("query", [
    "What is the marker made of?",
    "Does the DXF file have annotations?",
    "Is there WhatsApp support?",
])
def test_relevant_query_detected(query):
    assert is_query_relevant(query) is True


@pytest.mark.parametrize("query", [
    "What's the weather like in Mumbai today?",
    "Can you recommend a good pizza recipe?",
    "What's a good app for tracking my data?",
    "Is there an arcade near a decade-old shopping mall?",
    "Can you search for architecture jobs nearby?",
])
def test_irrelevant_query_not_falsely_detected(query):
    assert is_query_relevant(query) is False


def test_criticality_buckets():
    assert compute_criticality(0.05) == "high"
    assert compute_criticality(0.15) == "medium"
    assert compute_criticality(0.25) == "low"
