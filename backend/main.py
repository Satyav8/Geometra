from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import CORS_ORIGINS, CHROMA_PERSIST_DIR, CHROMA_COLLECTION, DATABASE_BACKEND
from database import init_db, check_health
from routers import chat, session, unknown

app = FastAPI(title="Geometra Pre-Prototype Chatbot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat.router)
app.include_router(session.router)
app.include_router(unknown.router)


@app.on_event("startup")
def startup():
    init_db()


@app.get("/health")
def health():
    db_status = "ok" if check_health() else "error"

    chroma_status = "ok"
    try:
        import chromadb
        from chromadb.config import Settings

        client = chromadb.PersistentClient(
            path=CHROMA_PERSIST_DIR,
            settings=Settings(anonymized_telemetry=False),
        )
        client.get_or_create_collection(CHROMA_COLLECTION)
    except Exception:
        chroma_status = "error"

    return {"status": "ok", "chroma": chroma_status, "database": db_status, "database_backend": DATABASE_BACKEND}


@app.get("/ingest-status")
def ingest_status():
    from chromadb.config import Settings
    import chromadb

    client = chromadb.PersistentClient(
        path=CHROMA_PERSIST_DIR, settings=Settings(anonymized_telemetry=False)
    )
    collection = client.get_or_create_collection(CHROMA_COLLECTION)
    return {"total_chunks": collection.count(), "collection": CHROMA_COLLECTION}
