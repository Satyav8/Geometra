import os
import sys

BACKEND_DIR = os.path.join(os.path.dirname(__file__), "..", "backend")
sys.path.insert(0, os.path.abspath(BACKEND_DIR))

from rag.ingestor import ingest_faq_from_sheet  # noqa: E402

if __name__ == "__main__":
    stats = ingest_faq_from_sheet()
    print(
        f"FAQ ingestion: {stats['total']} chunks total "
        f"({stats['added_or_changed']} added/changed, {stats['unchanged']} unchanged, "
        f"{stats['removed']} removed)"
    )
