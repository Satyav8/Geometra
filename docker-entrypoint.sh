#!/bin/sh
# The production start command, reproduced verbatim.
#
# On Render this service starts with backend/ as its working directory and:
#
#     python ../scripts/ingest_faq.py && uvicorn main:app --host 0.0.0.0 --port $PORT
#
# Both halves matter and the order matters.
#
# The ingest is not a deployment chore - it is how the knowledge base gets into the vector
# store at all. FAQ_SOURCE=sheet means the FAQ lives in a Google Sheet, so every deploy and
# every cold start re-reads it. Skip it and the container comes up healthy and answers "I
# can only help with questions about Geometra" to every on-topic question, because
# retrieval has nothing to retrieve. That exact failure has already cost this project an
# afternoon, on a teammate's laptop with an empty Chroma store.
#
# It is also safe to run on every boot, which is what makes this acceptable as an
# entrypoint: scripts/ingest_faq.py goes through rag/faq_sync.py, which skips the write
# entirely when the sheet's content hash is unchanged, refuses a source that comes back
# suspiciously short rather than emptying the index, and verifies what it wrote against
# the store. A sheet that is briefly unreachable prints a loud refusal and lets the service
# start on the previous contents - a stale FAQ beats no bot.
set -e

# "&&" in the start command means a failed ingest must stop the boot. set -e is that.
python ../scripts/ingest_faq.py

# exec, so uvicorn replaces this shell as PID 1 and receives SIGTERM directly. Without it
# the shell holds PID 1, swallows the stop signal, and in-flight answers are cut off when
# the platform's grace period runs out and the container is killed.
exec uvicorn main:app \
    --host 0.0.0.0 \
    --port "${PORT:-8000}" \
    --workers "${WEB_CONCURRENCY:-1}"
