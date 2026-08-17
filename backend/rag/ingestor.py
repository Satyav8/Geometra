import csv
import hashlib
import io
import re
import uuid

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


def _stable_chunk_id(section_name: str, text: str) -> str:
    """Deterministic UUID derived from a chunk's own content - identical content always
    maps to the same id, so "this chunk is unchanged since last ingestion" is just "this
    id already exists in the vector store", with no separate tracking needed. Also a
    valid Qdrant point id (Qdrant requires an unsigned int or a UUID string)."""
    digest = hashlib.md5(f"{section_name}::{text}".encode("utf-8")).digest()
    return str(uuid.UUID(bytes=digest))


def _chunk_rows(rows):
    """rows: iterable of (section_name, combined_qa_text). Returns [(chunk_id, section_name, text)]."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=60)
    chunks = []
    for section_name, text in rows:
        docs = splitter.create_documents([text], metadatas=[{"section_name": section_name}])
        for doc in docs:
            chunk_id = _stable_chunk_id(section_name, doc.page_content)
            chunks.append((chunk_id, section_name, doc.page_content))
    return chunks


def _store_chunks(rows) -> dict:
    """Only embeds/uploads chunks that are new or whose content changed since the last
    ingestion, and removes chunks whose source row no longer exists - a full re-ingest
    used to re-embed the entire FAQ every time regardless of what actually changed,
    which cost real (if small) OpenAI credits on every server boot in production."""
    chunks_raw = _chunk_rows(rows)
    # Dedupe by id: identical content (e.g. a literally duplicated row in the sheet)
    # hashes to the same id, and a vector store upsert rejects the same id appearing
    # twice in one call.
    seen = set()
    chunks = []
    for c in chunks_raw:
        if c[0] not in seen:
            chunks.append(c)
            seen.add(c[0])

    new_ids = {c[0] for c in chunks}
    existing_ids = vectorstore.get_all_ids()

    ids_to_add = new_ids - existing_ids
    ids_to_remove = existing_ids - new_ids

    to_embed = [c for c in chunks if c[0] in ids_to_add]
    if to_embed:
        chunk_ids = [c[0] for c in to_embed]
        texts = [c[2] for c in to_embed]
        metadatas = [{"chunk_id": c[0], "section_name": c[1]} for c in to_embed]
        embeddings = embed_batch(texts)
        vectorstore.upsert_chunks(chunk_ids, embeddings, texts, metadatas)

    if ids_to_remove:
        vectorstore.delete_chunks(list(ids_to_remove))

    return {
        "total": len(chunks),
        "added_or_changed": len(ids_to_add),
        "removed": len(ids_to_remove),
        "unchanged": len(chunks) - len(ids_to_add),
    }


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


def ingest_faq_from_sheet(csv_url: str = FAQ_SHEET_CSV_URL) -> dict:
    sheet_rows = fetch_sheet_rows(csv_url)
    rows = [
        (category, f"Q: {question}\nA: {answer}")
        for category, question, answer in sheet_rows
    ]
    return _store_chunks(rows)


def ingest_faq(faq_path: str) -> dict:
    """Legacy path: ingest from a local plain-text FAQ file (## Section: <name> headers)."""
    with open(faq_path, "r", encoding="utf-8") as f:
        raw_text = f.read()
    sections = _split_into_sections(raw_text)
    return _store_chunks(sections)
