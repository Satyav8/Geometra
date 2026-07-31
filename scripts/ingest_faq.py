import os
import sys

BACKEND_DIR = os.path.join(os.path.dirname(__file__), "..", "backend")
sys.path.insert(0, os.path.abspath(BACKEND_DIR))

from rag.ingestor import ingest_faq_from_sheet  # noqa: E402

if __name__ == "__main__":
    n = ingest_faq_from_sheet()
    print(f"Ingested {n} chunks into ChromaDB collection geometra_faq (source: Google Sheet)")
