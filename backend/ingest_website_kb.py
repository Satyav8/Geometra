"""One-off ingestion of new-to-the-bot content scraped from geometra.in (home/about/pricing
pages) into the isolated `geometra_website` Qdrant collection - see website_kb.py for why
this is a separate collection from the production FAQ one.

Only includes facts NOT already covered by the existing FAQ sheet. The website's own /faq
page (54 questions) was checked and found to be the same content as the already-ingested
Google Sheet (near-identical question wording throughout), so it was not re-scraped.

Run: ./venv/Scripts/python.exe ingest_website_kb.py
"""
import sys

sys.path.insert(0, ".")
from rag.embedder import embed_batch
import website_kb

# (section_name, question, answer) - answers paraphrase geometra.in's own wording, not
# invented. Source pages: https://geometra.in/, /about, /pricing (fetched 2026-08-19).
ROWS = [
    (
        "About Geometra — Company Website",
        "What is Geometra's vision or mission?",
        "Geometra's vision is that a 10-year-old with average intelligence should be able "
        "to produce more accurate CAD drawings than the best professional surveyor in the "
        "market, and do it in a tenth of the time.",
    ),
    (
        "About Geometra — Company Website",
        "Does Geometra use AI to calculate measurements?",
        "No. Every measurement comes from geometry and the marker's known real-world scale, "
        "solved with trigonometry, not a black-box AI guess. The same photo always produces "
        "the same answer.",
    ),
    (
        "About Geometra — Company Website",
        "Is Geometra's measurement accuracy really the best in the industry?",
        "Geometra states it delivers 99%+ accuracy in indoor space measurement, which it "
        "describes on its website as the highest accuracy in the world for this kind of "
        "measurement.",
    ),
    (
        "About Geometra — Company Website",
        "Does Geometra have a social media presence, like Instagram?",
        "Yes, Geometra is on Instagram at instagram.com/geometra.in.",
    ),
]


def main():
    texts = [f"Q: {q}\nA: {a}" for _, q, a in ROWS]
    chunk_ids = [website_kb.stable_chunk_id(section, text) for (section, _, _), text in zip(ROWS, texts)]
    metadatas = [{"section_name": section} for section, _, _ in ROWS]

    embeddings = embed_batch(texts)
    website_kb.upsert_chunks(chunk_ids, embeddings, texts, metadatas)

    print(f"Ingested {len(ROWS)} chunks into '{website_kb.WEBSITE_COLLECTION}'.")
    print(f"Collection now has {website_kb.count()} points total.")


if __name__ == "__main__":
    main()
