"""The endpoint the FAQ spreadsheet's own trigger calls when a row is added or changed.

This is the only write endpoint in the service, and it does two expensive things: it spends
OpenAI embedding calls, and it replaces the live knowledge base. So unlike /chat it is shut
by default - with INGEST_TOKEN unset it returns 503 and does nothing. There is no mode in
which this is reachable without a secret.

Why a shared secret and not real auth: the caller is a Google Apps Script and a scheduled
job, neither of which has a user to authenticate. Proper auth arrives with the wider
integration (the user's standing decision); a constant-time token comparison is the right
weight for "a script proves it is the script". It is not an authorization model and is not
pretending to be one - anyone holding the token can trigger a re-ingest.
"""
import secrets

from fastapi import APIRouter, Header, HTTPException, Request
from typing import Optional

from config import INGEST_TOKEN
from rag.faq_sync import last_sync, sync_faq
from rate_limiter import limiter

router = APIRouter(prefix="/admin")


def _require_token(provided: Optional[str]) -> None:
    if not INGEST_TOKEN:
        raise HTTPException(
            status_code=503,
            detail="Re-ingest is disabled: INGEST_TOKEN is not configured on this service.",
        )
    # compare_digest rather than == so a wrong token cannot be narrowed down by timing.
    if not provided or not secrets.compare_digest(provided, INGEST_TOKEN):
        raise HTTPException(status_code=401, detail="Invalid or missing X-Ingest-Token.")


# Rate limited despite needing a token: a trigger misconfigured into a loop (an Apps Script
# edit trigger that reacts to its own write, say) would otherwise hammer the vector store
# and the embedding API with a valid credential. The content hash in faq_sync already makes
# repeats cheap; this bounds them even when content really is changing every time.
@router.post("/reingest")
@limiter.limit("10/minute;60/hour")
def reingest(
    request: Request,
    x_ingest_token: Optional[str] = Header(default=None),
    force: bool = False,
):
    """Mirrors the FAQ source into the vector store. Safe to call repeatedly - unchanged
    content is a no-op (see rag/faq_sync.py).

    force=true skips the content-hash short-circuit, for the case where the store and the
    source have drifted for a reason the hash cannot see (a manual edit to the vector DB,
    or a restart that lost the hash while the store was mid-change).
    """
    _require_token(x_ingest_token)
    result = sync_faq(force=force)
    # "refused" is a safety stop and "mismatch" means the write did not stick. Neither is
    # a server fault, but the caller has to be able to tell both apart from success without
    # parsing prose - 409 so a scheduled job's failure shows up as a failure, not a quiet 200.
    if result["status"] in ("refused", "mismatch"):
        raise HTTPException(status_code=409, detail=result)
    return result


@router.get("/faq-status")
def faq_status(x_ingest_token: Optional[str] = Header(default=None)):
    """What this process knows about the last successful sync. Behind the same token: it
    is operational detail, and it is the thing to check when the team says a new FAQ has
    not shown up - a silently failing trigger looks exactly like a stale last_synced_at."""
    _require_token(x_ingest_token)
    return last_sync()
