# Geometra Pre-Prototype Chatbot

RAG-based customer support chatbot for Geometra (image-to-CAD wall measurement product).
Pre-prototype stack: FastAPI + ChromaDB (local) + SQLite + React + GPT-4o Mini.

See `Geometra_Claude_Code_Spec.docx` (project root, one level up) for the full technical specification.

## Running it the way production runs it

The instructions below set up a **development** environment - SQLite, a local Chroma
store, local embeddings - which is deliberately not what the live service uses. To run
the production system, see **[DOCKER.md](DOCKER.md)**: one image, the same start sequence
as the live deploy, and the handover notes for the accounts it depends on.

## Backend setup

```
cd backend
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
python database.py           # initializes SQLite tables
python ../scripts/ingest_faq.py   # builds the ChromaDB knowledge base
uvicorn main:app --reload --port 8000
```

Note: dependencies require Python 3.10-3.12 (langchain 0.2.x pins numpy<2.0, which has no
Python 3.13 wheel on Windows). Use `py -3.10` or `py -3.12` if you have multiple Python
versions installed.

## Frontend setup

```
cd frontend
npm install
npm run dev
```

## Environment

Copy `.env.example` to `.env` and fill in your LLM API key. Never commit `.env`.
