"""Ingests the FAQ into the vector store. Run by hand, and on every boot.

WHY THIS GOES THROUGH sync_faq() RATHER THAN ingest_faq_current()

This script is not only a manual tool. The Render start command runs it on every deploy
and every cold start - which was discovered on 2026-10-10 from a production log line
("FAQ ingestion: 215 chunks total") that main.py cannot produce, after I had wrongly
stated that startup did no ingestion. It does; it just does it from the start command
rather than the application.

That makes this the most frequently executed ingest path there is, and until now it was
the only one with no safety at all. ingest_faq_current() fetches and writes: no row-count
floor, no content hash, no check that the write landed. Every protection added in
rag/faq_sync.py was reachable only through the /admin/reingest endpoint.

Two things follow from that, both real:

  a near-empty source would delete the FAQ. The floor exists because a source can parse
  perfectly and still come back short - a renamed tab, a half-finished paste, a
  permission change mid-edit. On this path there was nothing to catch it.

  a transient fetch failure would take the service down. FAQ_SOURCE is now the Google
  Sheet, so this reaches out to Google on every boot. An exception here fails the start
  command, and the bot does not come up at all - for a problem that has nothing to do
  with the bot.

So this now calls sync_faq(), which refuses rather than destroys, verifies the write
against the store, and reports a status instead of raising. A failed sync leaves the
PREVIOUS contents in place and lets the service start: stale FAQ answers are a far better
failure than no bot, and the status is printed loudly enough to see in the deploy log.

A refusal does NOT fail the process by default. That default is deliberate: this runs
unattended on every boot, and the safe behaviour must not depend on someone remembering to
set an environment variable. Set FAQ_INGEST_STRICT=1 for a non-zero exit when a person or
a CI job wants one. The store is untouched either way - the only question is whether the
caller wants an exit code, and the caller that cannot tolerate one is the unattended one.
"""
import os
import sys

BACKEND_DIR = os.path.join(os.path.dirname(__file__), "..", "backend")
sys.path.insert(0, os.path.abspath(BACKEND_DIR))

from rag.faq_sync import sync_faq  # noqa: E402

# Opt IN to a non-zero exit. Default is to carry on, so a slow or briefly unreachable
# spreadsheet cannot stop the service starting.
STRICT = os.getenv("FAQ_INGEST_STRICT", "").lower() in ("1", "true", "yes")

if __name__ == "__main__":
    result = sync_faq(force=True)
    status = result.get("status")

    if status == "synced":
        print(
            f"FAQ ingestion: {result['chunks']} chunks total "
            f"({result['added_or_changed']} added/changed, {result['unchanged']} unchanged, "
            f"{result['removed']} removed), verified {result.get('verified_in_store')} "
            f"in store, from {result['rows']} rows"
        )
    elif status == "unchanged":
        print(f"FAQ ingestion: no change ({result['rows']} rows), nothing written")
    else:
        # refused / mismatch / busy. The store still holds whatever it held before, which
        # is why this is survivable: the bot serves the previous FAQ rather than nothing.
        print(
            f"FAQ ingestion {status.upper()}: {result.get('reason')}\n"
            f"  The vector store was NOT modified and still serves its previous contents.",
            file=sys.stderr,
        )
        if STRICT:
            sys.exit(1)
        print("  Continuing anyway - a FAQ that is briefly stale is better than a service "
              "that will not start. Set FAQ_INGEST_STRICT=1 to exit non-zero instead.")
