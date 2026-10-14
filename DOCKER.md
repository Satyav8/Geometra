# Running S.A.M in Docker

This runs the chatbot the way the live service runs it: same Python version, same start
sequence, same configuration, same refusals when something is missing. The point of the
exercise was not "put it in a container" but "put the *production* system in a container",
so the differences from the live deployment are listed at the bottom rather than glossed
over.

Two containers: the **API** and the **chat UI** in front of it. If you already have the
accounts the service depends on, section 1 is the whole job. If you are standing those
up from scratch, do section 3 first.

> **This file is the authoritative copy.** A shared runbook covering the same ground
> exists for handing to people who are not working in the repository, but it is a
> convenience copy: it sits outside version control, so it cannot be reviewed alongside a
> change and will drift. When the two disagree, this one is right, because it is the one
> that travels with the code it describes.

---

## 1. Build and run

```bash
cp .env.production.example .env.production      # then fill in the seven secrets
GIT_COMMIT=$(git rev-parse --short HEAD) docker compose up --build
```

Open <http://localhost:8080> and talk to it.

The UI and the API answer on **one origin**: nginx in the frontend container serves the
built React app and forwards `/chat` to the backend over the compose network. The browser
never makes a cross-origin request, so CORS never has to be configured to see the thing
work — which matters because the backend refuses to start with a localhost entry in
`CORS_ORIGINS` when `APP_ENV=production`, ruling out the usual local workaround.

The API is published on port 8000 as well, so you can curl it directly.

### Just the API, without compose

```bash
docker build -t sam-backend --build-arg GIT_COMMIT=$(git rev-parse --short HEAD) .
docker run --env-file .env.production -p 8000:8000 sam-backend
```

The build context is the **repository root**, not `backend/`. That matters — see section 4.

### Confirming it works

```bash
curl http://localhost:8000/health
```

```json
{"status":"ok","version":"a9ca881","app_env":"production","database":"ok",
 "database_backend":"supabase","vector_db":"ok","vector_db_backend":"qdrant"}
```

Read every field, not just `status`. Each rules out a failure that otherwise stays
invisible.

| Field | Must say | If it says something else |
|---|---|---|
| `version` | the commit you built from | `unknown` means you left out the `GIT_COMMIT` build argument |
| `app_env` | `production` | `development` means the strict startup checks never ran |
| `database_backend` | `supabase` | conversations are going to a file inside the container and will vanish with it |
| `vector_db_backend` | `qdrant` | retrieval will return nothing |
| `database`, `vector_db` | `ok` | the service was reached but is not answering — usually a wrong or expired credential |

Then ask it something, which is the only check that proves retrieval is wired up. A
correct answer to "how much does it cost per wall" mentions ₹199. The failure worth
knowing by sight is an on-topic question answered with *"I can only help with questions
about Geometra"* — that is an empty vector store, not a broken bot. See section 4.

---

## 2. Configuration

`.env.production.example` is the complete list, with each variable explained next to it.
Seven are secrets and are the only things not already filled in:

| Secret | What it is |
|---|---|
| `OPENAI_API_KEY` | Answers (gpt-4o-mini) **and** embeddings (text-embedding-3-small) |
| `QDRANT_URL`, `QDRANT_API_KEY` | The vector store holding the FAQ |
| `SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY` | Conversations, tickets, evaluation logs |
| `RESEND_API_KEY` | Sends the ticket email to the support inbox |
| `INGEST_TOKEN` | Shared secret for `POST /admin/reingest` |

**The container is meant to refuse to start.** With `APP_ENV=production`,
`backend/startup_checks.py` asserts that every backend is the durable shared one and that
every credential it needs is present, and raises with *all* the problems at once:

```
Refusing to start: 2 configuration problem(s).
  - APP_ENV=production but DATABASE_BACKEND='sqlite' (expected 'supabase') - SQLite lives
    on the container's own disk and is destroyed on every restart and redeploy.
  - APP_ENV=production but RESEND_API_KEY is empty - a confirmed support ticket would be
    written to the database with nobody notified.
```

That is the feature, not a hurdle. Every one of those states would otherwise let the
service start, pass its health check, and fail silently — which is how this project lost
an afternoon to a SQLite database being destroyed on every redeploy.

---

## 3. Standing up the services from scratch

Skip this if you were handed working credentials. Do it in this order; only the first one
has a step that is easy to miss.

### Supabase — and the one step nothing warns you about

Create a project, then **run `backend/data/supabase_deploy_schema.sql` once in the SQL
Editor**. The application does *not* create its own tables: `init_db()` is deliberately a
no-op on this backend, because schema changes belong in one reviewed file rather than in
whatever the app happened to do on boot.

Skip it and the container starts, `/health` reports `"database":"error"`, and every answer
still works — until the first ticket, which has nowhere to go. That file is idempotent and
supersedes `supabase_main_schema.sql` and `supabase_escalated_questions.sql`, which were
written in two phases and left the session-state columns unapplied; running it again on an
older project is the intended way to bring it up to date.

Then copy the project URL and the **service role** key (Project Settings → API) into
`.env.production`. Service role, not anon: it bypasses row-level security, which is why it
belongs only in the server's environment.

### Qdrant

Create a cluster and copy its URL and API key. There is nothing to create inside it — the
collection is made automatically on the first ingest, at the vector size the configured
embedding backend produces.

### OpenAI

One key, used for both answers and embeddings. It needs billing enabled; the free trial
tier will rate-limit the FAQ ingest.

### Resend

Create an API key. `RESEND_FROM_EMAIL` defaults to Resend's shared test domain, which can
only send to the account's own verified address — fine for testing, not for customers.
Verifying a Geometra domain is the first improvement worth making.

### The FAQ spreadsheet

Share it as **"anyone with the link can view"**, or the ingest cannot read it. Then, to
make edits reach the bot automatically, install the trigger:

1. In the sheet: **Extensions → Apps Script**
2. Paste `scripts/sheet_onchange_trigger.gs` over `Code.gs`, and save
3. **Project Settings → Script properties**: add `INGEST_TOKEN`, the same value as in
   `.env.production`. A script property rather than a constant in the file, because anyone
   with edit access to the sheet can read the code and that token triggers paid work.
4. Edit `ENDPOINT` at the top of the script to point at your deployment
5. **Triggers → Add trigger**: function `onFaqChange`, source "From spreadsheet", type
   "On change"
6. Run `onFaqChange` once by hand to accept the authorisation prompt

### GitHub Actions (optional but recommended)

`.github/workflows/faq_sync.yml` re-ingests hourly as a safety net behind that trigger.
Add two repository secrets: `INGEST_URL` (your `/admin/reingest` URL) and `INGEST_TOKEN`
(the same value again). Without them the workflow fails loudly, which is the intent.

---

## 4. What the container does on boot

The entrypoint reproduces the production start command exactly:

```sh
python ../scripts/ingest_faq.py && uvicorn main:app --host 0.0.0.0 --port $PORT
```

**The ingest is not a deployment chore — it is how the knowledge base gets loaded at
all.** `FAQ_SOURCE=sheet` means the FAQ lives in a Google Sheet, so every boot re-reads
it. Skip it and the container comes up healthy and tells every customer "I can only help
with questions about Geometra", because retrieval has nothing to retrieve.

This is also why the build context is the repository root. `scripts/ingest_faq.py` sits
*outside* `backend/`, so an image built from `backend/` alone — as the previous Dockerfile
was — physically cannot run the start command production runs. The repository layout is
kept intact inside the image (`/app/backend`, `/app/scripts`) with `/app/backend` as the
working directory, which is also what the flat imports require.

Running the ingest on every boot is safe because it goes through `rag/faq_sync.py`, which

- **skips the write entirely** when the sheet's content hash is unchanged, so a restart
  with no FAQ edits costs one HTTP request and writes nothing;
- **refuses** a source returning fewer than `FAQ_MIN_ROWS` rows rather than emptying the
  index — the mirror is authoritative and deletes what the sheet no longer has, so a
  half-loaded fetch would otherwise destroy the knowledge base;
- **verifies** what it wrote against the store's own count, because a Qdrant write once
  reported success without persisting;
- **never fails the boot** on a refusal. A briefly unreachable spreadsheet prints loudly
  and the service starts on the previous contents: a stale FAQ beats no bot. Set
  `FAQ_INGEST_STRICT=1` if you want a non-zero exit instead.

Expect the first boot to take **20–40 seconds** before the port is open. The container is
not unhealthy during it, which is what the 120s health check start period is for, and why
the UI container waits for the API to be *healthy* rather than merely started.

---

## 5. Keeping the FAQ live

Edits to the spreadsheet reach the bot by two independent paths, both needing
`INGEST_TOKEN` to be the **same value** in three places — the container, the sheet's Apps
Script, and the GitHub Actions secret:

1. **The sheet's own change trigger** calls `POST /admin/reingest` once edits have stopped
   for 90 seconds, so a half-typed row is never ingested.
2. **An hourly GitHub Action** calls the same endpoint as a safety net, because that
   trigger can fail invisibly: authorisation expires, someone removes it while tidying,
   Google drops the trigger.

If `INGEST_TOKEN` is unset the endpoint returns **503 and is disabled entirely**, which is
the right default — it spends money on embedding calls and replaces the live knowledge
base — but it also means new FAQ rows silently stop going live. Check it:

```bash
curl -s -o /dev/null -w '%{http_code}\n' -X POST http://localhost:8000/admin/reingest -H "X-Ingest-Token: $INGEST_TOKEN"
```

`200` is working (synced, unchanged or busy — all fine). `401` means the token drifted,
`409` means a safety check refused and the bot is now behind the sheet, `503` means no
token is configured.

---

## 6. The chat UI

`docker compose up` builds and serves it; there is nothing to configure. Vite compiles the
React app, nginx serves the static files and proxies the API paths to the backend
container, and the result answers on <http://localhost:8080>.

To work on the UI instead, run it against a **development** backend:

```bash
cd frontend && npm install && npm run dev
```

The Vite dev server proxies `/chat` to `localhost:8000`, so this is same-origin too and
needs no CORS entry. Set `VITE_API_URL` only when the UI really is served from a different
origin than the API — which is how the Render deployment works, and the only reason that
variable still exists.

---

## 7. Differences from the live deployment

An honest list, so nothing surprises you later:

| | Live service | This stack |
|---|---|---|
| Host | Render, native Python build | any Docker host |
| Python | 3.11.9, from `backend/.python-version` | 3.11.9, pinned to the patch |
| Start command | `python ../scripts/ingest_faq.py && uvicorn ...` | identical, via the entrypoint |
| Version marker | `RENDER_GIT_COMMIT`, set automatically | `--build-arg GIT_COMMIT`, **you must pass it** |
| UI | separate Render static site, cross-origin, needs `CORS_ORIGINS` | nginx container, same origin, CORS unused |
| Workers | 1 | 1 |
| Stop signal | Render sends SIGTERM to uvicorn | `exec` in the entrypoint, so uvicorn is PID 1 |
| Idle behaviour | free tier sleeps; first request can take ~50s | stays up |

The version marker is the one that needs you to do something. Omit the build argument and
`/health` honestly reports `"version":"unknown"`, which defeats the purpose of having it.

---

## 8. Day to day

With compose: `docker compose logs -f`, `stop`, `start`, `down`, `ps`.

What needs which action:

- **Changed an environment value** — restart. No rebuild.
- **Changed the code** — rebuild with a fresh `GIT_COMMIT`, then recreate.
- **Changed a FAQ row in the spreadsheet** — nothing. The trigger pushes it, the hourly
  Action catches what the trigger missed, and any restart re-reads the sheet anyway.

Nothing on either container needs backing up. They hold no state — conversations live in
Supabase, the knowledge base in Qdrant — so deleting and recreating them loses nothing.

---

## 9. Handover notes

The code in this repository is complete and is what production runs. What does **not**
transfer with it are the accounts the service depends on — all of them are currently on
personal credentials, and every one needs to move to a Geometra-owned account with the old
key revoked rather than left live:

- **OpenAI** — answers and embeddings. Billing sits here.
- **Qdrant Cloud** — the FAQ vector store.
- **Supabase** — conversations, tickets, evaluation logs. The service-role key bypasses
  row-level security, so it is the single most sensitive value in the file.
- **Resend** — ticket emails, currently on Resend's shared test domain.
- **Google Sheet** — the FAQ source of truth. Needs to be owned by, and shared with, the
  people who will be editing it.
- **Render**, and the **GitHub repository** including the `INGEST_TOKEN` and `INGEST_URL`
  Actions secrets.

Rotate every key during the handover regardless of whether you think it leaked. It is one
afternoon now against an unbounded liability later.
