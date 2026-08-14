import csv
import io
import re

import requests
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import FAQ_SHEET_CSV_URL
from rag import vectorstore
from rag.embedder import embed_batch

SECTION_HEADER_RE = re.compile(r"^##\s*Section:\s*(.+?)\s*$", re.MULTILINE)


def _split_into_sections(raw_text: str):
    matches = list(SECTION_HEADER_RE.finditer(raw_text))
    sections = []
    for i, match in enumerate(matches):
        section_name = match.group(1).strip()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(raw_text)
        section_text = raw_text[start:end].strip()
        sections.append((section_name, section_text))
    return sections


def _store_chunks(rows) -> int:
    """rows: iterable of (section_name, combined_qa_text). Chunks, embeds, and stores in ChromaDB."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400, chunk_overlap=60, add_start_index=True
    )

    chunk_texts = []
    chunk_metadatas = []
    chunk_ids = []
    counter = 0

    for section_name, text in rows:
        docs = splitter.create_documents([text], metadatas=[{"section_name": section_name}])
        for doc in docs:
            chunk_id = f"faq_{counter:03d}"
            start = doc.metadata.get("start_index", 0)
            end = start + len(doc.page_content)
            chunk_texts.append(doc.page_content)
            chunk_metadatas.append(
                {
                    "chunk_id": chunk_id,
                    "section_name": section_name,
                    "char_start": start,
                    "char_end": end,
                }
            )
            chunk_ids.append(chunk_id)
            counter += 1

    embeddings = embed_batch(chunk_texts)
    vectorstore.recreate_and_store(chunk_ids, embeddings, chunk_texts, chunk_metadatas)
    return len(chunk_ids)


def fetch_sheet_rows(csv_url: str = FAQ_SHEET_CSV_URL):
    """Downloads the FAQ Google Sheet and yields (category, question, answer) for every usable row."""
    response = requests.get(csv_url, timeout=30)
    response.raise_for_status()
    # requests defaults text/csv to Latin-1 when no charset is declared in the response
    # headers, mangling multi-byte UTF-8 characters (em dashes, curly quotes, etc.) from
    # the sheet. Google's CSV export is UTF-8, so decode it explicitly.
    reader = csv.reader(io.StringIO(response.content.decode("utf-8")))

    header = None
    for row in reader:
        if row and any(c.strip().lower() == "question" for c in row):
            header = [c.strip() for c in row]
            break

    if header is None:
        raise ValueError("Could not find header row (expected a 'Question' column) in the FAQ sheet")

    # Case-insensitive, position-independent lookup: the team has renamed/reordered
    # columns before (e.g. "ID" -> "SlnoSlno", "Category" -> "category ", moved to a
    # different position) without changing which columns are actually needed, so match
    # by lowercase name rather than exact position/casing.
    col = {name.lower(): idx for idx, name in enumerate(header)}
    required = {"category", "question", "draft answer", "updated response"}
    missing = required - set(col.keys())
    if missing:
        raise ValueError(f"FAQ sheet is missing expected columns: {missing}")

    rows = []
    for row in reader:
        if len(row) <= col["question"]:
            continue
        category = row[col["category"]].strip() if col["category"] < len(row) else ""
        question = row[col["question"]].strip() if col["question"] < len(row) else ""
        draft_answer = row[col["draft answer"]].strip() if col["draft answer"] < len(row) else ""
        updated_answer = row[col["updated response"]].strip() if col["updated response"] < len(row) else ""
        answer = updated_answer or draft_answer

        if not category or not question or not answer:
            continue

        rows.append((category, question, answer))

    return rows


def ingest_faq_from_sheet(csv_url: str = FAQ_SHEET_CSV_URL) -> int:
    sheet_rows = fetch_sheet_rows(csv_url)
    rows = [
        (category, f"Q: {question}\nA: {answer}")
        for category, question, answer in sheet_rows
    ]
    return _store_chunks(rows)


def ingest_faq(faq_path: str) -> int:
    """Legacy path: ingest from a local plain-text FAQ file (## Section: <name> headers)."""
    with open(faq_path, "r", encoding="utf-8") as f:
        raw_text = f.read()
    sections = _split_into_sections(raw_text)
    return _store_chunks(sections)
