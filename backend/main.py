from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import CORS_ORIGINS, DATABASE_BACKEND, VECTOR_DB_BACKEND
from database import init_db, check_health
from rag import vectorstore
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
    vector_status = "ok" if vectorstore.check_health() else "error"

    return {
        "status": "ok",
        "database": db_status,
        "database_backend": DATABASE_BACKEND,
        "vector_db": vector_status,
        "vector_db_backend": VECTOR_DB_BACKEND,
    }


@app.get("/ingest-status")
def ingest_status():
    return {"total_chunks": vectorstore.count(), "backend": VECTOR_DB_BACKEND}
