# Geometra Support Chatbot (S.A.M)

RAG-based customer support chatbot for Geometra, an image-to-CAD wall measurement
product. FastAPI + React, with a two-pass LLM flow over a Google Sheet knowledge base.

**Two ways to run it, and they are deliberately different.**

| | What it uses | Start here |
| --- | --- | --- |
| **Production** | OpenAI, Qdrant, Supabase, the live FAQ sheet | **[DOCKER.md](DOCKER.md)** — one command |
| **Development** | Groq, local Chroma, SQLite, a committed FAQ snapshot | the rest of this file |

The backend enforces the split in both directions. It refuses to start as production on
development backends, and refuses to start as development on the production ones — a
laptop writing into the live customer database, or re-ingesting over the live FAQ index,
has cost this project two afternoons. `ALLOW_PRODUCTION_BACKENDS=1` opts out when you
genuinely mean it.

## Running everything (production configuration)

```bash
cp .env.production.example .env.production      # fill in the seven secrets
GIT_COMMIT=$(git rev-parse --short HEAD) docker compose up --build
```

Then open <http://localhost:8080>. [DOCKER.md](DOCKER.md) covers the secrets, standing up
the services from scratch, what the container does on boot, and what to do when it
refuses to start.

## Development setup

### Backend

```bash
cp .env.example .env         # then fill in GROQ_API_KEY
cd backend
python -m venv venv
venv\Scripts\activate        # Windows; source venv/bin/activate elsewhere
pip install -r requirements.txt
python database.py                # initializes SQLite tables
python ../scripts/ingest_faq.py   # builds the local Chroma knowledge base
uvicorn main:app --reload --port 8000
```

Do not skip the ingest. Without it the vector store is empty and every on-topic question
comes back "I can only help with questions about Geometra" — the bot looks broken when
only the knowledge base is missing.

Python 3.10–3.12. Production runs 3.11.9 (`backend/.python-version`); langchain 0.2.x
pins numpy<2.0, which has no 3.13 wheel on Windows.

### Frontend

```bash
cd frontend
npm install
npm run dev
```

It opens on <http://localhost:5173> and proxies API calls to `localhost:8000`, so the
browser sees one origin and CORS never has to be configured. Set `VITE_API_URL` only when
the UI is served from a different origin than the API, as on Render.

### Tests

```bash
cd backend
python -m pytest ../tests
```

**Run them from `backend/`, not the repository root.** `CHROMA_PERSIST_DIR` is a relative
path, so the working directory decides which vector store opens; from the root you get an
empty one and 20 retrieval tests fail with `assert 0 > 0`, which looks exactly like a real
regression. 507 pass, 1 skipped.

## How it fits together

- `backend/llm/two_pass.py` — the two-pass flow: pass 1 understands and reformulates, pass
  2 answers from retrieved context. Deterministic layers short-circuit ahead of both.
- `backend/rag/faq_sync.py` — mirrors the FAQ sheet into the vector store, refusing rather
  than destroying when the source looks wrong.
- `backend/startup_checks.py` — why the service refuses to start on a misconfiguration.
- `backend/routers/admin.py` — `POST /admin/reingest`, called by the sheet's trigger and
  an hourly GitHub Action.
- `SYSTEM_OVERVIEW.md` — the longer architectural write-up, if it was shared with you.

## Environment

`.env.example` → `.env` for development; `.env.production.example` → `.env.production`
for the real thing. Both are templates: every secret in them is blank. Never commit
either filled-in file — `.gitignore` covers anything `.env`-shaped.
