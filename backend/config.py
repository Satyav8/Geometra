import os
from dotenv import load_dotenv

# override=True: .env must always win over stray inherited shell/OS environment
# variables of the same name (e.g. a leftover OPENAI_API_KEY from another project
# on this machine) - otherwise dotenv silently keeps the shell's value instead.
load_dotenv(override=True)

# LLM
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
# llama-3.3-70b-versatile is decommissioned by Groq on 2026-08-16 - switched the local
# dev default to Groq's recommended replacement. Production is unaffected (runs
# LLM_PROVIDER=openai/gpt-4o-mini via Render env vars, set independently of this default).
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")

# RAG — embeddings. "local" (default, local dev/tests, no API key needed) or "openai"
# (production, text-embedding-3-small). Kept separate for the same reason as every other
# *_BACKEND flag here: local dev/tests should never spend real API credits.
EMBEDDING_BACKEND = os.getenv("EMBEDDING_BACKEND", "local")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
# 384 = MiniLM's output size, 1536 = text-embedding-3-small's default output size. Used
# only to create the Qdrant collection at the right size — must match EMBEDDING_BACKEND.
EMBEDDING_DIM = 1536 if EMBEDDING_BACKEND == "openai" else 384
# Raised 6->15: gives the LLM enough of the FAQ at once to synthesize an answer across
# multiple entries (e.g. "must I stick the marker before photographing?" isn't answered
# by any single FAQ row, but is answerable by combining several). Verified against a
# batch of adversarial/multi-fact questions with no hallucination or injection failures,
# plus the original 20-query retrieval set and the local MiniLM suite (81/81, no
# regression). See llm/prompts.py's [Source: X] labeling fix, needed once this many
# chunks are in context.
TOP_K_CHUNKS = int(os.getenv("TOP_K_CHUNKS", "15"))
# Verified against real OpenAI embeddings too: on-topic scores ranged 0.48-0.88 (avg
# 0.71), out-of-domain scores were 0.19-0.22 - the existing 0.30/0.60 cutoffs still sit
# in a clean gap, so no threshold change was needed when switching embedding backends.
MIN_SIMILARITY_SCORE = float(os.getenv("MIN_SIMILARITY_SCORE", "0.30"))
LOW_CONFIDENCE_THRESHOLD = float(os.getenv("LOW_CONFIDENCE_THRESHOLD", "0.60"))
# Two-pass flow's fast-path scope gate: keyword hit OR similarity >= this value counts as
# in-scope, no LLM spent otherwise. Loosened from MIN_SIMILARITY_SCORE (0.30) after testing
# on backend/try_it_yourself.py found 0.30 alone rejected some legitimate on-topic phrasings
# that only had a keyword match, not a strong embedding match.
FAST_PATH_SIMILARITY_THRESHOLD = float(os.getenv("FAST_PATH_SIMILARITY_THRESHOLD", "0.15"))

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
# The email button shown alongside this is a manual backup only (a mailto: draft the
# customer can optionally send themselves) — the ticket and team notification already
# happened automatically before this message is even shown, so the wording here must not
# imply the customer needs to click anything for the ticket to exist.
TICKET_RAISED_MESSAGE = (
    "I've created a support ticket ({ticket_number}) on your behalf for the Geometra "
    "team — they'll get back to you within 12-24 hours. If you'd also like to email us "
    "directly, you can use the button below."
)

GRATITUDE_MESSAGE = "Most welcome! I'm here to assist you if there are any further doubts."

GREETING_MESSAGE = "Hello! How may I help you?"

# After this many turns, nudge the customer to confirm resolution / escalate to a human.
ESCALATION_TURN_THRESHOLD = int(os.getenv("ESCALATION_TURN_THRESHOLD", "6"))
SUPPORT_EMAIL = os.getenv("SUPPORT_EMAIL", "satyav8.geometra@gmail.com")
CHECK_IN_MESSAGE = (
    "\n\nHas this resolved your query? If it hasn't, please contact our support "
    "team through email and we'll help you further."
)

# Two-pass flow messages — ported from backend/try_it_yourself.py (Testing branch) after
# extensive manual testing. See llm/two_pass.py for how each of these is triggered.
SAFETY_REFUSAL_MESSAGE = (
    "I can't help with that. This chat is here for genuine, respectful questions about "
    "using Geometra to measure interior spaces and objects - happy to help if you have "
    "one of those."
)

MANNEQUIN_EXCLUSION_MESSAGE = (
    "Unfortunately, Geometra isn't able to measure that - mannequins, statues, dolls, and "
    "stuffed toys fall under representations of a living thing, which are outside what "
    "Geometra supports. I'd be happy to help with anything else in the room you'd like "
    "measured!"
)

EXCLUDED_ITEM_MESSAGE_TEMPLATE = (
    "Unfortunately, Geometra isn't able to measure that since {reason}. I'd be happy to "
    "help with anything else in the room you'd like measured!"
)

TICKET_OFFER_MESSAGE = (
    "That's a fair question, and I'd rather not guess and risk giving you the "
    "wrong answer. I don't have that specific detail available to me right now, "
    "but I can raise a support ticket so our team follows up with you directly "
    "with an accurate answer. Would you like me to do that? (yes/no)"
)

# Separate wording for the deterministic escalation checks in llm/two_pass.py - those
# trigger on a REQUEST ("please raise a ticket," "I already tried, no response"), not an
# unanswered QUESTION, so TICKET_OFFER_MESSAGE's "that's a fair question... I don't have
# that detail" doesn't fit - there's no question being dodged.
TICKET_ESCALATION_MESSAGE = (
    "I hear you, and I don't want to keep going back and forth without getting you "
    "real help. I can raise a support ticket so our team follows up with you directly. "
    "Would you like me to go ahead? (yes/no)"
)

TICKET_DECLINED_MESSAGE = (
    "No problem, I won't raise a ticket. Let me know if there's anything else I can help with!"
)

CLARIFY_DECLINE_PROMPT_MESSAGE = "No worries — what would you like me to clarify or help with instead?"

FILLER_RESPONSE_MESSAGE = "No worries! Let me know whenever you have a question about Geometra."
