"""Mirrors the FAQ spreadsheet into the vector store, safely enough to run unattended.

WHY THIS IS NOT JUST ingest_faq_current()

That function already does the work: it fetches rows and replaces the FAQ chunks. The
problem is what happens when it runs on a trigger nobody is watching.

Sheet-as-source has already caused real damage in this project. fetch_faq_rows() carries
the note: an ingest run against the sheet silently replaced deliberate content updates,
twice, minutes after they went in, with nothing to show it had happened. That was a HUMAN
running a script. Putting the same operation behind an automatic trigger makes it faster
and less visible, so the destructive paths have to be closed first.

Three things this adds:

  a floor on row count   - the mirror is authoritative, so a fetch that comes back nearly
                           empty would delete the FAQ. A sheet temporarily broken, a tab
                           renamed, a permission revoked mid-edit, someone clearing rows
                           to re-paste them: all plausible, all cost the whole knowledge
                           base. Below FAQ_MIN_ROWS nothing is written and the caller is
                           told why.
  a content hash         - the trigger fires on every edit, including ones that change no
                           FAQ content (a comment, a formatting change, an unrelated
                           column). Re-running the ingest then costs embedding calls for
                           no reason. Unchanged content is a no-op.
  a lock                 - two triggers overlapping would have two writers replacing the
                           same collection. The second caller is told it is busy rather
                           than queued, because the first run is already picking up the
                           same sheet state.

A malformed fetch is already safe without any of this: fetch_sheet_rows() raises if it
cannot find a "Question" header, so a Google sign-in HTML page or an error page aborts
before anything is written. The floor exists for the case that parses fine and is simply
missing content.
"""
import hashlib
import threading
import time
from typing import Optional

from config import FAQ_MIN_ROWS
from llm.categorizer import classify_category, save_cache
from rag import vectorstore
from rag.ingestor import _store_chunks, fetch_faq_rows

# Non-blocking: a caller that arrives mid-sync is told to come back, not queued behind it.
# Queuing would mean a second identical pass over the same sheet state.
_SYNC_LOCK = threading.Lock()

# Last successful sync, for the content-hash short-circuit. In-memory on purpose: a
# restart simply means the next trigger does one real ingest, which is correct and cheap.
# Persisting it would add a way for the store and the recorded hash to disagree.
_last_hash: Optional[str] = None
_last_synced_at: Optional[float] = None
_last_row_count: Optional[int] = None


def _content_hash(rows) -> str:
    """Identifies the FAQ content, not the spreadsheet. Ordering is included because
    chunk order affects nothing downstream but reordering is also not a content change
    worth an ingest - hashing the rows as given keeps this honest and cheap."""
    h = hashlib.md5()
    for category, question, answer in rows:
        h.update(f"{category}\x00{question}\x00{answer}\x1e".encode("utf-8"))
    return h.hexdigest()


def sync_faq(force: bool = False) -> dict:
    """Pulls the FAQ source and mirrors it into the vector store.

    Returns a dict with a "status" of:
      synced     - content changed and the store now matches the source
      unchanged  - the source is byte-identical to the last successful sync; nothing written
      busy       - another sync is in progress; nothing written
      mismatch   - the write reported success but the store disagrees; treat as a failure
      refused    - a safety check failed; nothing written, "reason" says which

    Never raises for an expected failure - the caller is an HTTP endpoint and a scheduled
    job, and both need a status they can report rather than a traceback. A genuinely
    unexpected error (the vector store being down mid-write) still propagates, because
    that is not something to paper over with a 200.
    """
    global _last_hash, _last_synced_at, _last_row_count

    if not _SYNC_LOCK.acquire(blocking=False):
        return {"status": "busy", "reason": "a sync is already running"}
    try:
        try:
            rows = list(fetch_faq_rows())
        except Exception as e:
            # Covers the whole malformed-source family: a sign-in page, a 404, a renamed
            # tab, a missing Question column. Nothing has been written at this point.
            return {"status": "refused", "reason": f"could not read the FAQ source: {e}"}

        if len(rows) < FAQ_MIN_ROWS:
            return {
                "status": "refused",
                "reason": (
                    f"source returned {len(rows)} rows, below the floor of {FAQ_MIN_ROWS} - "
                    "refusing to mirror what looks like a broken or emptied source. "
                    "Raise FAQ_MIN_ROWS if the FAQ is genuinely this small now."
                ),
                "rows": len(rows),
            }

        # The spreadsheet the team edits has no Category column, but the category is what
        # the customer sees in the "[Source: ...]" line. Derived here, from the team's own
        # eight labels, and cached by question so an unchanged FAQ costs no calls at all.
        # See llm/categorizer.py.
        needing_category = sum(1 for category, _, _ in rows if not category.strip())
        if needing_category:
            rows = [
                (category.strip() or classify_category(question), question, answer)
                for category, question, answer in rows
            ]
            save_cache()

        # Hashed AFTER categorisation, because the category is part of a chunk's identity
        # (_stable_chunk_id is md5("section::text")) - correcting a label by hand in the
        # cache file has to be able to trigger a real re-ingest.
        content = _content_hash(rows)
        if content == _last_hash and not force:
            return {
                "status": "unchanged",
                "rows": len(rows),
                "last_synced_at": _last_synced_at,
            }

        stats = _store_chunks([
            (category, f"Q: {question}\nA: {answer}")
            for category, question, answer in rows
        ])

        # Verified against the store rather than trusting the write. _store_chunks reports
        # what it INTENDED to write; it has been observed returning a clean "118 added"
        # while the collection stayed at its old size, which for an unattended trigger is
        # the worst failure available - the bot serves stale content and the job goes
        # green. The count is the one thing that can be checked independently and cheaply.
        try:
            in_store = vectorstore.count()
        except Exception as e:
            in_store = None
            print(f"[faq_sync] could not verify the store count: {e}")

        if in_store is not None and in_store != stats["total"]:
            return {
                "status": "mismatch",
                "reason": (
                    f"wrote {stats['total']} chunks but the store reports {in_store} - "
                    "the sync was not applied and the bot may be serving stale content"
                ),
                "expected": stats["total"],
                "in_store": in_store,
            }

        _last_hash, _last_synced_at, _last_row_count = content, time.time(), len(rows)
        return {
            "status": "synced",
            "verified_in_store": in_store,
            "rows": len(rows),
            "chunks": stats["total"],
            "added_or_changed": stats["added_or_changed"],
            "unchanged": stats["unchanged"],
            "removed": stats["removed"],
            "categories_filled": needing_category,
        }
    finally:
        _SYNC_LOCK.release()


def last_sync() -> dict:
    """What the process knows about the most recent successful sync. Reported by the
    endpoint so a silently-failing trigger is visible without reading logs."""
    return {
        "last_synced_at": _last_synced_at,
        "rows": _last_row_count,
        "content_hash": _last_hash,
    }
