import pytest

from rag.relevance import compute_criticality, is_gratitude, is_query_relevant
from rag.retriever import retrieve

# 20 test queries checked against a distinctive substring of their known-correct answer,
# not the FAQ's category/section label. The team has renamed the category taxonomy twice
# now (Title Case -> lowercase, "ID"/"Category" -> "SlnoSlno"/"category ") without changing
# the underlying answers, so asserting on category names kept going stale for reasons
# unrelated to retrieval quality. Answer content is far more stable than category naming.
TEST_QUERIES = [
    ("What is Geometra?", "exact measurements"),
    ("How does the measurement process work step by step?", "Stick our marker"),
    ("Do I need any special hardware or a laser scanner?", "laser printer"),
    ("Do I have to download an app to use Geometra?", "No app to download"),
    ("What file formats do I get as output?", "DXF"),
    ("Is Geometra a floor-plan tool or an elevations tool?", "elevation"),
    ("How accurate is Geometra in millimeters?", "10 mm"),
    ("Has Geometra been tested against a laser measure?", "laser measurements"),
    ("What's the largest wall Geometra can measure?", "marker size"),
    ("What is the marker and why do I need it?", "reference point"),
    ("How do I print the marker correctly?", "portrait"),
    ("Can I reuse the same marker multiple times?", "again and again"),
    ("Where do I place the marker on the wall?", "anywhere on the wall"),
    ("How many corners of the wall need to be visible in the photo?", "3 corners"),
    ("Can I measure a whole room at once?", "One wall at a time"),
    ("How much does Geometra cost per wall?", "399"),
    ("What does the free plan include?", "3 wall elevations"),
    ("Can I get a refund if a scan fails?", "refund policy"),
    ("Is my uploaded photo data stored securely?", "privacy and data policy"),
    ("How do I sign up for Geometra?", "Google"),
]

UNKNOWN_QUERIES = [
    "What's the weather like in Mumbai today?",
    "Can you recommend a good pizza recipe?",
]


@pytest.mark.parametrize("query,expected_substring", TEST_QUERIES)
def test_retrieval_returns_expected_answer(query, expected_substring):
    chunks, confidence_level = retrieve(query)
    assert len(chunks) > 0
    assert confidence_level in ("high", "low")
    combined_text = " ".join(c.text for c in chunks)
    assert expected_substring.lower() in combined_text.lower()


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
