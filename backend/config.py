import os
from dotenv import load_dotenv

load_dotenv()

# LLM
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")

# RAG — local embedding model (no API key / credits required)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMBEDDING_DIM = 384  # sentence-transformers/all-MiniLM-L6-v2 output size
TOP_K_CHUNKS = int(os.getenv("TOP_K_CHUNKS", "5"))
MIN_SIMILARITY_SCORE = float(os.getenv("MIN_SIMILARITY_SCORE", "0.30"))
LOW_CONFIDENCE_THRESHOLD = float(os.getenv("LOW_CONFIDENCE_THRESHOLD", "0.60"))

# Vector DB — "chroma" (default, local dev/tests) or "qdrant" (production). Same
# reasoning as DATABASE_BACKEND: local dev/tests stay fast and offline.
VECTOR_DB_BACKEND = os.getenv("VECTOR_DB_BACKEND", "chroma")
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_db")
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "geometra_faq")

# Qdrant Cloud — used only when VECTOR_DB_BACKEND=qdrant.
QDRANT_URL = os.getenv("QDRANT_URL", "")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "geometra_faq")

# FAQ knowledge base — live Google Sheet is the source of truth (team keeps adding rows)
FAQ_SHEET_ID = os.getenv("FAQ_SHEET_ID", "1dkd0Qj-6kTc72eXk0UCGFi47RrEP0fiKRAK-jPrFMtA")
FAQ_SHEET_GID = os.getenv("FAQ_SHEET_GID", "1861055441")
FAQ_SHEET_CSV_URL = (
    f"https://docs.google.com/spreadsheets/d/{FAQ_SHEET_ID}/export?format=csv&gid={FAQ_SHEET_GID}"
)

# Evaluation
MIN_COMPLETENESS_WORDS = int(os.getenv("MIN_COMPLETENESS_WORDS", "8"))

# Database — "sqlite" (default, local dev/tests) or "supabase" (production).
# Kept separate on purpose: local dev/test runs should never write into the real
# Supabase project (see the escalated_questions mixing issue we hit earlier).
DATABASE_BACKEND = os.getenv("DATABASE_BACKEND", "sqlite")
SQLITE_DB_PATH = os.getenv("SQLITE_DB_PATH", "./geometra_chat.db")

# Supabase — same project already used for escalated_questions. Also the main
# database (sessions/messages/evaluation_logs/unknown_questions) when
# DATABASE_BACKEND=supabase.
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")

# Resend — sends the automated ticket email (with full conversation transcript) to
# SUPPORT_EMAIL whenever a question is escalated. Free tier can send to the account's
# own verified email without domain verification, which is why RESEND_FROM_EMAIL
# defaults to Resend's shared test domain rather than a custom one.
RESEND_API_KEY = os.getenv("RESEND_API_KEY", "")
RESEND_FROM_EMAIL = os.getenv("RESEND_FROM_EMAIL", "S.A.M <onboarding@resend.dev>")

# Server
BACKEND_PORT = int(os.getenv("BACKEND_PORT", "8000"))
FRONTEND_PORT = int(os.getenv("FRONTEND_PORT", "5173"))
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",")

# Bot identity
BOT_NAME = "S.A.M"
BOT_FULL_NAME = "S.A.M (Simple Answering Machine)"

# Fixed numeric constants referenced by the system prompt / guardrails
ALLOWED_PRICE_NUMBERS = {399, 3, 99}

# The email itself is delivered as a clickable button by the frontend (see support_email
# field on ChatResponse) — kept out of this text so it isn't duplicated as plain text too.
FALLBACK_MESSAGE = (
    "I don't have enough information about the question that you have asked. "
    "You can contact our support team through email."
)

OUT_OF_SCOPE_MESSAGE = "I can only help with questions about Geometra."

# Shown instead of the generic fallback once a ticket has actually been raised for the
# customer's question (see routers/chat.py) — ticket_number is filled in at request time.
TICKET_RAISED_MESSAGE = (
    "I don't have enough information about the question that you have asked. "
    "I've raised a support ticket ({ticket_number}) for our team — they'll get back "
    "to you within 12-24 hours."
)

GRATITUDE_MESSAGE = "Most welcome! I'm here to assist you if there are any further doubts."

# After this many turns, nudge the customer to confirm resolution / escalate to a human.
ESCALATION_TURN_THRESHOLD = int(os.getenv("ESCALATION_TURN_THRESHOLD", "6"))
SUPPORT_EMAIL = os.getenv("SUPPORT_EMAIL", "satyav8.geometra@gmail.com")
CHECK_IN_MESSAGE = (
    "\n\nHas this resolved your query? If it hasn't, please contact our support "
    "team through email and we'll help you further."
)
