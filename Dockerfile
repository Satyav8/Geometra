# S.A.M - the Geometra support chatbot, packaged to run the way production runs.
#
#   docker build -t sam-backend --build-arg GIT_COMMIT=$(git rev-parse --short HEAD) .
#   docker run --env-file .env.production -p 8000:8000 sam-backend
#
# Build context is the REPOSITORY ROOT, not backend/. That is not a style choice: the
# production start command is
#
#     python ../scripts/ingest_faq.py && uvicorn main:app --host 0.0.0.0 --port $PORT
#
# run with backend/ as the working directory, so the FAQ ingest that happens on every
# deploy and every cold start lives OUTSIDE backend/. An image built from backend/ alone
# cannot reproduce it. The repository layout is therefore kept intact inside the image
# (/app/backend, /app/scripts) and the working directory is /app/backend - which is also
# what the flat imports ("from config import ...", "from routers import chat") require.
#
# Two stages. The build stage carries a C toolchain because a few dependencies still
# compile from source on slim images; the runtime stage does not, which keeps the compiler
# and its headers out of anything that faces the internet.

# ----------------------------------------------------------------------------- build
# 3.11.9 is what backend/.python-version pins, which is what the production host installs.
# Pinned to the patch here as well, deliberately: "runs exactly like production" is the
# whole point of this image, and 3.11-slim would drift away from it on its own schedule.
FROM python:3.11.9-slim AS build

# gcc/g++ for packages without manylinux wheels; the rest is what those builds need.
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential \
 && rm -rf /var/lib/apt/lists/*

# Dependencies are installed into their own prefix so the runtime stage can take the whole
# tree in one COPY without dragging the toolchain along.
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1

# requirements.txt alone first: this layer is only rebuilt when dependencies change, so
# ordinary code edits reuse the cached install instead of recompiling everything.
COPY backend/requirements.txt /tmp/requirements.txt
RUN pip install --prefix=/install -r /tmp/requirements.txt

# ----------------------------------------------------------------------------- runtime
FROM python:3.11.9-slim AS runtime

# PYTHONUNBUFFERED is not cosmetic here: without it, logs sit in a buffer and a crash can
# take the very lines that explain it down with it. The startup banner and the FAQ
# ingestion line are the two things anyone debugging this container will look for first.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

COPY --from=build /install /usr/local

# Application code last, so a code change invalidates only these layers.
COPY backend/ /app/backend/
COPY scripts/ /app/scripts/
COPY docker-entrypoint.sh /app/docker-entrypoint.sh

WORKDIR /app/backend

# Which commit is answering. config.py normally reads RENDER_GIT_COMMIT, falling back to
# .git/HEAD - an image has neither, so /health would report "unknown" and the version
# marker would be useless in exactly the place it was added for. Pass it at build time:
#   --build-arg GIT_COMMIT=$(git rev-parse --short HEAD)
ARG GIT_COMMIT=unknown
ENV BUILD_VERSION=${GIT_COMMIT}

# sed before chmod: this repository is developed on Windows, and a checkout with
# core.autocrlf=true gives the entrypoint CRLF line endings. The container then fails with
# the famously unhelpful "exec /app/docker-entrypoint.sh: no such file or directory",
# because the kernel reads the shebang as "/bin/sh\r". .gitattributes pins it to LF as
# well; this is the belt behind that brace.
#
# Runs as a non-root user: nothing here needs to write to the filesystem (state lives in
# Supabase, vectors in Qdrant), so the container has no reason to hold root.
RUN sed -i 's/\r$//' /app/docker-entrypoint.sh \
 && chmod +x /app/docker-entrypoint.sh \
 && useradd --create-home --uid 10001 sam \
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

# No curl on slim, so the check runs in Python. The start period covers the FAQ ingest the
# entrypoint runs before uvicorn binds the port: against the live Google Sheet and OpenAI
# embeddings that is normally 20-40s, and a container is not unhealthy while it is doing
# the thing it is supposed to do on boot.
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
  CMD python -c "import os,urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:' + os.environ.get('PORT','8000') + '/health', timeout=8).status == 200 else 1)"

# A script rather than a shell-form CMD so that uvicorn can be exec'd into PID 1 and
# actually receive SIGTERM. Under "sh -c 'a && b'" the shell is PID 1, uvicorn is its
# child, and a stop signal reaches the shell only - the platform waits out its grace
# period and then kills the container, cutting off in-flight answers.
ENTRYPOINT ["/app/docker-entrypoint.sh"]
