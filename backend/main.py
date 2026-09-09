from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from config import CORS_ORIGINS, DATABASE_BACKEND, VECTOR_DB_BACKEND
from database import init_db, check_health
from rag import vectorstore
from rate_limiter import limiter
from routers import chat, session, unknown

app = FastAPI(title="Geometra Pre-Prototype Chatbot")

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

# CORS added last so it ends up outermost (Starlette applies middleware in reverse
# registration order) - a rate-limited 429 still needs CORS headers, or the browser
# reports it to the caller as an opaque network error instead of a readable 429.
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
