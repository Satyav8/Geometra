# S.A.M backend — build context is backend/, not the repo root:
#   docker build -t sam-backend ./backend
#
# The app's imports are flat ("from config import ...", "from routers import chat"), so
# backend/ has to be the working directory, not a package inside one.
#
# Two stages. The build stage carries a C toolchain because a few dependencies still
# compile from source on slim images; the runtime stage does not, which keeps the compiler
# and its headers out of anything that faces the internet.

# ----------------------------------------------------------------------------- build
FROM python:3.10-slim AS build

# gcc/g++ for packages without manylinux wheels; the rest is what those builds need.
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential \
 && rm -rf /var/lib/apt/lists/*

# Dependencies are installed into their own prefix so the runtime stage can take the whole
# tree in one COPY without dragging the toolchain along.
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1

# requirements.txt alone first: this layer is only rebuilt when dependencies change, so
# ordinary code edits reuse the cached install instead of recompiling everything.
COPY requirements.txt /tmp/requirements.txt
RUN pip install --prefix=/install -r /tmp/requirements.txt

# ----------------------------------------------------------------------------- runtime
FROM python:3.10-slim AS runtime

# PYTHONUNBUFFERED is not cosmetic here: without it, logs sit in a buffer and a crash can
# take the very lines that explain it down with it. Learned the hard way chasing a run
# whose output never appeared.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY --from=build /install /usr/local

# Application code last, so a code change invalidates only this layer.
COPY . /app

# Runs as a non-root user: nothing here needs to write to the filesystem (state lives in
# Supabase, vectors in Qdrant), so the container has no reason to hold root.
RUN useradd --create-home --uid 10001 sam \
 && chown -R sam:sam /app
USER sam

# Documentation only - the real port comes from $PORT below, which is what Render, Fly and
# Cloud Run all inject.
EXPOSE 8000
ENV PORT=8000

# One worker by default, deliberately. /chat is a sync def, so Starlette already gives it
# a 40-thread pool per worker - that is the concurrency, and it is why llm/two_pass.py
# sizes its moderation pool to 40. Extra workers multiply memory and give each its own
# separate moderation pool, so raise WEB_CONCURRENCY only alongside a bigger instance.
ENV WEB_CONCURRENCY=1

# No curl on slim, so the check runs in Python. Start period is generous because the first
# request also opens the Qdrant and Supabase connections.
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD python -c "import os,urllib.request,sys; \
sys.exit(0 if urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\",8000)}/health', timeout=8).status == 200 else 1)"

# Shell form so $PORT and $WEB_CONCURRENCY are expanded at start time rather than baked in.
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT} --workers ${WEB_CONCURRENCY}
