"""Guards the automated FAQ mirror (rag/faq_sync.py) and its endpoint (routers/admin.py).

The mirror is authoritative - it removes chunks the source no longer has - and it runs on
a trigger nobody is watching. That combination is why these tests are about the REFUSALS
rather than the happy path: an ingest that works is visible in the bot's answers, while an
ingest that quietly destroys the knowledge base is not.

The specific history being defended against is in fetch_faq_rows()' own docstring: a sheet
ingest silently replaced deliberate content updates twice, minutes after they went in,
with nothing to show it had happened. Automating the same operation makes that faster, so
the floor, the hash and the lock exist before the trigger does.

No network and no embedding calls - the source and the store are both stubbed.
"""
import pytest
from fastapi.testclient import TestClient

import config
from rag import faq_sync


@pytest.fixture(autouse=True)
def _clean_sync_state(monkeypatch):
    """Each test starts with no remembered sync, so the hash short-circuit is explicit
    rather than inherited from whichever test ran first."""
    monkeypatch.setattr(faq_sync, "_last_hash", None)
    monkeypatch.setattr(faq_sync, "_last_synced_at", None)
    monkeypatch.setattr(faq_sync, "_last_row_count", None)


def _rows(n, answer="an answer"):
    return [("Printing marker", f"question {i}?", answer) for i in range(n)]


def _stub_source(monkeypatch, rows):
    monkeypatch.setattr(faq_sync, "fetch_faq_rows", lambda: iter(rows))


def _stub_store(monkeypatch, calls, store_count=None):
    """Stubs the writer AND the verifier.

    sync_faq reads the store's own count back to confirm a write actually landed, so a test
    that stubs only the writer would trip that check against the real local store.
    store_count=None means "the store agrees with what was written" (the happy path);
    passing a number simulates a write that reported success without sticking.
    """
    def _store(chunks):
        calls.append(chunks)
        return {"total": len(chunks), "added_or_changed": len(chunks),
                "unchanged": 0, "removed": 0}
    monkeypatch.setattr(faq_sync, "_store_chunks", _store)

    class _FakeStore:
        @staticmethod
        def count():
            if store_count is not None:
                return store_count
            return len(calls[-1]) if calls else 0

    monkeypatch.setattr(faq_sync, "vectorstore", _FakeStore)


def test_a_healthy_source_is_mirrored(monkeypatch):
    calls = []
    _stub_source(monkeypatch, _rows(99))
    _stub_store(monkeypatch, calls)
    result = faq_sync.sync_faq()
    assert result["status"] == "synced"
    assert result["rows"] == 99
    assert len(calls) == 1


def test_a_nearly_empty_source_is_refused_without_writing(monkeypatch):
    """The core safety property. A source that parses fine but lost its rows - a renamed
    tab, a cleared sheet mid-paste - must not be allowed to delete the knowledge base."""
    calls = []
    _stub_source(monkeypatch, _rows(3))
    _stub_store(monkeypatch, calls)
    result = faq_sync.sync_faq()
    assert result["status"] == "refused"
    assert "below the floor" in result["reason"]
    assert calls == [], "a refused sync still wrote to the vector store"


def test_an_unreadable_source_is_refused_without_writing(monkeypatch):
    """A Google sign-in page, a 404, a revoked permission. fetch_sheet_rows() raises on
    these; the mirror must turn that into a reported refusal, not a traceback and not an
    empty write."""
    calls = []

    def _boom():
        raise ValueError("Could not find header row (expected a 'Question' column)")

    monkeypatch.setattr(faq_sync, "fetch_faq_rows", _boom)
    _stub_store(monkeypatch, calls)
    result = faq_sync.sync_faq()
    assert result["status"] == "refused"
    assert "could not read" in result["reason"]
    assert calls == []


def test_unchanged_content_costs_nothing(monkeypatch):
    """The trigger fires on formatting changes and comments too. Re-embedding then is pure
    waste, so identical content must not reach the store a second time."""
    calls = []
    _stub_source(monkeypatch, _rows(99))
    _stub_store(monkeypatch, calls)
    assert faq_sync.sync_faq()["status"] == "synced"
    second = faq_sync.sync_faq()
    assert second["status"] == "unchanged"
    assert len(calls) == 1, "unchanged content was ingested again"


def test_force_overrides_the_hash(monkeypatch):
    """For drift the hash cannot see - the store changed under us, or a restart lost the
    hash mid-change."""
    calls = []
    _stub_source(monkeypatch, _rows(99))
    _stub_store(monkeypatch, calls)
    faq_sync.sync_faq()
    assert faq_sync.sync_faq(force=True)["status"] == "synced"
    assert len(calls) == 2


def test_an_edited_row_is_a_change(monkeypatch):
    """The hash must track content, not row count - editing an answer in place leaves the
    count identical, and that is the most common edit the team makes."""
    calls = []
    _stub_source(monkeypatch, _rows(99))
    _stub_store(monkeypatch, calls)
    faq_sync.sync_faq()
    _stub_source(monkeypatch, _rows(99, answer="a corrected answer"))
    assert faq_sync.sync_faq()["status"] == "synced"
    assert len(calls) == 2


def test_a_concurrent_sync_is_told_to_wait(monkeypatch):
    """Two triggers overlapping would have two writers replacing the same collection."""
    calls = []
    _stub_source(monkeypatch, _rows(99))
    _stub_store(monkeypatch, calls)
    faq_sync._SYNC_LOCK.acquire()
    try:
        result = faq_sync.sync_faq()
    finally:
        faq_sync._SYNC_LOCK.release()
    assert result["status"] == "busy"
    assert calls == []


# --- the endpoint -----------------------------------------------------------------------
# Imported inside the tests so the module-level INGEST_TOKEN read happens after the
# monkeypatch, the same way it would on a service that boots with the env var set.

def _client(monkeypatch, token):
    monkeypatch.setattr(config, "INGEST_TOKEN", token)
    import routers.admin as admin
    monkeypatch.setattr(admin, "INGEST_TOKEN", token)
    import main
    return TestClient(main.app)


def test_endpoint_is_disabled_when_no_token_is_configured(monkeypatch):
    """The default posture. This endpoint spends money and replaces the knowledge base, so
    "no token set" must mean "unreachable", never "open"."""
    client = _client(monkeypatch, "")
    response = client.post("/admin/reingest")
    assert response.status_code == 503
    assert "disabled" in response.json()["detail"].lower()


def test_endpoint_rejects_a_missing_or_wrong_token(monkeypatch):
    client = _client(monkeypatch, "a-real-token-value")
    assert client.post("/admin/reingest").status_code == 401
    assert client.post(
        "/admin/reingest", headers={"X-Ingest-Token": "guess"}
    ).status_code == 401


def test_endpoint_syncs_with_the_right_token(monkeypatch):
    monkeypatch.setattr(faq_sync, "sync_faq", lambda force=False: {"status": "synced", "rows": 99})
    import routers.admin as admin
    monkeypatch.setattr(admin, "sync_faq", lambda force=False: {"status": "synced", "rows": 99})
    client = _client(monkeypatch, "a-real-token-value")
    response = client.post("/admin/reingest", headers={"X-Ingest-Token": "a-real-token-value"})
    assert response.status_code == 200
    assert response.json()["status"] == "synced"


def test_a_refusal_is_reported_as_a_failure_not_a_quiet_200(monkeypatch):
    """A scheduled job that gets a 200 records a success. A safety refusal means the bot is
    now serving older content than the sheet shows, which has to surface as a failure."""
    import routers.admin as admin
    monkeypatch.setattr(
        admin, "sync_faq",
        lambda force=False: {"status": "refused", "reason": "source returned 2 rows"},
    )
    client = _client(monkeypatch, "a-real-token-value")
    response = client.post("/admin/reingest", headers={"X-Ingest-Token": "a-real-token-value"})
    assert response.status_code == 409
    assert "refused" in str(response.json())


def test_the_floor_is_configured_well_below_the_live_row_count():
    """Guards both ends of the floor: high enough that an emptied source trips it, low
    enough that ordinary editing of a ~99-row FAQ never does."""
    assert 10 <= config.FAQ_MIN_ROWS <= 90


def test_a_write_that_does_not_stick_is_reported_as_a_failure(monkeypatch):
    """The failure this verification exists for, observed on 2026-10-08: a Qdrant sync
    returned a clean "118 added" while the collection stayed at its previous size for
    several minutes. For an unattended trigger that is the worst available outcome - the
    bot serves stale content and the scheduled job goes green - so the store's own count
    is read back and a disagreement is a failure, not a success.
    """
    calls = []
    _stub_source(monkeypatch, _rows(99))
    _stub_store(monkeypatch, calls, store_count=116)
    result = faq_sync.sync_faq()
    assert result["status"] == "mismatch"
    assert result["in_store"] == 116
    assert "stale" in result["reason"]


def test_a_mismatch_does_not_record_a_successful_sync(monkeypatch):
    """A failed write must not update the content hash, or the next trigger would decide
    there is nothing to do and the store would stay stale indefinitely."""
    calls = []
    _stub_source(monkeypatch, _rows(99))
    _stub_store(monkeypatch, calls, store_count=116)
    faq_sync.sync_faq()
    assert faq_sync.last_sync()["content_hash"] is None
    # And the next attempt must actually retry rather than short-circuit.
    _stub_store(monkeypatch, calls)
    assert faq_sync.sync_faq()["status"] == "synced"


def test_an_unreadable_store_count_does_not_block_the_sync(monkeypatch):
    """Verification is a check, not a dependency: if the count call itself fails, the write
    already happened and the sync must still be reported as done."""
    calls = []
    _stub_source(monkeypatch, _rows(99))
    _stub_store(monkeypatch, calls)

    class _Broken:
        @staticmethod
        def count():
            raise RuntimeError("vector store unreachable")

    monkeypatch.setattr(faq_sync, "vectorstore", _Broken)
    result = faq_sync.sync_faq()
    assert result["status"] == "synced"
    assert result["verified_in_store"] is None
