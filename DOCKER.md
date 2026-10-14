# Running S.A.M in Docker

This image runs the chatbot the way the live service runs it: same Python version, same
start sequence, same configuration, same refusals when something is missing. The point of
the exercise was not "put it in a container" but "put the *production* system in a
container", so the differences from the live deployment are listed at the bottom rather
than glossed over.

---

## 1. Build and run

```bash
cp .env.production.example .env.production      # then fill in the seven secrets
docker build -t sam-backend --build-arg GIT_COMMIT=$(git rev-parse --short HEAD) .
docker run --env-file .env.production -p 8000:8000 sam-backend
```

Or with compose, which does both:

```bash
GIT_COMMIT=$(git rev-parse --short HEAD) docker compose up --build
```

The build context is the **repository root**, not `backend/`. That matters - see section 3.

### Confirming it works

```bash
curl http://localhost:8000/health
```

```json
{"status":"ok","version":"a9ca881","app_env":"production","database":"ok",
 "database_backend":"supabase","vector_db":"ok","vector_db_backend":"qdrant"}
```

Read every field, not just `status`:

- **`version`** is the commit the container was built from. It exists because replies
  carrying a long-fixed bug once turned out to be coming from a build that predated the
  fix, while the dashboard showed the fixed build as live. "Which build is answering?"
  should be one request, not a forensic exercise.
- **`app_env`** must say `production`. If it says `development` the strict startup
  checks never ran, and the container may be on throwaway backends.
- **`database_backend` / `vector_db_backend`** must say `supabase` and `qdrant`. Any
  other value means conversations are being written to a file inside the container and
  will vanish with it.

Then ask it something, which is the only check that proves retrieval is actually wired up:

```bash
curl -X POST http://localhost:8000/chat -H 'Content-Type: application/json' -d '{"session_id":"11111111-1111-1111-1111-111111111111","query":"how much does it cost per wall","turn_number":1}'
```

A correct answer mentions 199. The failure worth knowing by sight is an on-topic question
answered with *"I can only help with questions about Geometra"* - that is an empty vector
store, not a broken bot. See section 3.

---

## 2. Configuration

`.env.production.example` is the complete list, with each variable explained next to it.
Seven of them are secrets and are the only things not already filled in:

| Secret | What it is |
|---|---|
| `OPENAI_API_KEY` | Answers (gpt-4o-mini) **and** embeddings (text-embedding-3-small) |
| `QDRANT_URL`, `QDRANT_API_KEY` | The vector store holding the FAQ |
| `SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY` | Conversations, tickets, evaluation logs |
| `RESEND_API_KEY` | Sends the ticket email to the support inbox |
| `INGEST_TOKEN` | Shared secret for `POST /admin/reingest` |

**The container is meant to refuse to start.** With `APP_ENV=production`,
`backend/startup_checks.py` asserts that every backend is the durable shared one and that
every credential it needs is present, and raises with *all* the problems listed at once:

```
Refusing to start: 2 configuration problem(s).
  - APP_ENV=production but DATABASE_BACKEND='sqlite' (expected 'supabase') - SQLite lives
    on the container's own disk and is destroyed on every restart and redeploy.
  - APP_ENV=production but RESEND_API_KEY is empty - a confirmed support ticket would be
    written to the database with nobody notified.
```

That is the feature, not a hurdle. Every one of those states would otherwise let the
service start, pass its health check, and fail silently - which is how this project lost
an afternoon to a SQLite database that was being destroyed on every redeploy.

---

## 3. What the container does on boot

The entrypoint reproduces the production start command exactly:

```sh
python ../scripts/ingest_faq.py && uvicorn main:app --host 0.0.0.0 --port $PORT
```

**The ingest is not a deployment chore - it is how the knowledge base gets loaded at
all.** `FAQ_SOURCE=sheet` means the FAQ lives in a Google Sheet, so every boot re-reads
it. Skip it and the container comes up healthy and tells every customer "I can only help
with questions about Geometra", because retrieval has nothing to retrieve.

This is also why the build context is the repository root. `scripts/ingest_faq.py` sits
*outside* `backend/`, so an image built from `backend/` alone - as the previous Dockerfile
was - physically cannot run the start command production runs. The repository layout is
kept intact inside the image (`/app/backend`, `/app/scripts`) with `/app/backend` as the
working directory, which is also what the flat imports require.

Running the ingest on every boot is safe because it goes through `rag/faq_sync.py`, which

- **skips the write entirely** when the sheet's content hash is unchanged, so a restart
  with no FAQ edits costs one HTTP request and writes nothing;
- **refuses** a source returning fewer than `FAQ_MIN_ROWS` rows rather than emptying the
  index - the mirror is authoritative and deletes what the sheet no longer has, so a
  half-loaded fetch would otherwise destroy the knowledge base;
- **verifies** what it wrote against the store's own count, because a Qdrant write once
  reported success without persisting;
- **never fails the boot** on a refusal. A briefly unreachable spreadsheet prints loudly
  and the service starts on the previous contents: a stale FAQ beats no bot. Set
  `FAQ_INGEST_STRICT=1` if you want a non-zero exit instead.

Expect the first boot to take **20-40 seconds** before the port is open. The container is
not unhealthy during it, which is what the 120s health check start period is for.

---

## 4. Keeping the FAQ live

Edits to the Google Sheet reach the bot by two independent paths, both of which need
`INGEST_TOKEN` to be the **same value** in three places - the container, the sheet's Apps
Script, and the GitHub Actions secret:

1. **The sheet's own edit trigger** (`scripts/sheet_onchange_trigger.gs`) calls
   `POST /admin/reingest` within a couple of minutes of an edit.
2. **An hourly GitHub Action** (`.github/workflows/faq_sync.yml`) calls the same endpoint
   as a safety net, because that trigger can fail invisibly: authorisation expires,
   someone removes it while tidying, Google drops the trigger.

If `INGEST_TOKEN` is unset the endpoint returns **503 and is disabled entirely**, which is
the right default - it spends money on embedding calls and replaces the live knowledge
base - but it also means new FAQ rows silently stop going live. Check it:

```bash
curl -s -o /dev/null -w '%{http_code}\n' -X POST http://localhost:8000/admin/reingest -H "X-Ingest-Token: $INGEST_TOKEN"
```

`200` is working (synced, unchanged or busy - all fine). `401` means the token drifted,
`409` means a safety check refused and the bot is now behind the sheet, `503` means no
token is configured.

---

## 5. The frontend

Not in this image, and not in production either: it is a Vite build served as static
files, not a running process. To point a local frontend at the container:

```bash
cd frontend && VITE_API_URL=http://localhost:8000 npm run dev
```

The browser will then be refused by CORS, and correctly so - under `APP_ENV=production`
the startup checks reject a `CORS_ORIGINS` containing localhost, because that almost
always means the real origin was never filled in. For UI work, run the backend with the
development `.env` instead of `.env.production`; for a production container, put the real
frontend origin in `CORS_ORIGINS`.

---

## 6. Differences from the live deployment

An honest list, so nothing surprises you later:

| | Live service | This image |
|---|---|---|
| Host | Render, native Python build | any Docker host |
| Python | 3.11.9, from `backend/.python-version` | 3.11.9, pinned to the patch |
| Start command | `python ../scripts/ingest_faq.py && uvicorn ...` | identical, via the entrypoint |
| Version marker | `RENDER_GIT_COMMIT`, set automatically | `--build-arg GIT_COMMIT`, **you must pass it** |
| Workers | 1 | 1 |
| Stop signal | Render sends SIGTERM to uvicorn | `exec` in the entrypoint, so uvicorn is PID 1 |
| Idle behaviour | free tier sleeps; first request can take ~50s | stays up |

The version marker is the one that needs you to do something. Omit the build argument and
`/health` honestly reports `"version":"unknown"`, which defeats the purpose of having it.

---

## 7. Handover notes

The code in this repository is complete and is what production runs. What does **not**
transfer with it are the accounts the service depends on - all of them are currently on
personal credentials, and every one needs to move to a Geometra-owned account with the
old key revoked rather than left live:

- **OpenAI** - answers and embeddings. Billing sits here.
- **Qdrant Cloud** - the FAQ vector store.
- **Supabase** - conversations, tickets, evaluation logs. The service-role key bypasses
  row-level security, so it is the single most sensitive value in the file.
- **Resend** - ticket emails. `RESEND_FROM_EMAIL` currently uses Resend's shared test
  domain; a verified Geometra domain would be the first improvement.
- **Google Sheet** - the FAQ source of truth. Needs to be owned by, and shared with, the
  people who will be editing it.
- **Render**, and the **GitHub repository** including the `INGEST_TOKEN` and `INGEST_URL`
  Actions secrets.

Rotate every key during the handover regardless of whether you think it leaked. It is one
afternoon now against an unbounded liability later.
