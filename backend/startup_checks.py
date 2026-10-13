"""Refuses to start on a configuration that would appear to work and silently lose data.

Every *_BACKEND setting in config.py defaults to the local-development choice - sqlite,
chroma, local embeddings. That is right for a laptop and dangerous in a container: a
deployment that forgets its environment variables starts perfectly, reports healthy, writes
every conversation to a SQLite file on the container's own disk, and loses all of it on the
next restart. Nothing errors, because SQLite genuinely works.

That is the same shape as the two silent failures found in production this week (a wrong
Supabase key that passed the health check, and a moderation pool that quietly stopped
checking under load). The fix in all three cases is the same: make the broken state loud.

Two kinds of check, deliberately separated:

  CONSISTENCY - always enforced. If you select a backend, you must supply what it needs.
                Catches "DATABASE_BACKEND=supabase" with no key, on a laptop or in prod.

  PRODUCTION  - only when APP_ENV=production. Asserts the backends are the shared, durable
                ones. This is the check that catches a container started with no
                environment at all, and it is opt-in so local development and the test
                suite are untouched.
"""
from typing import List

import config


class ConfigurationError(RuntimeError):
    """Raised at startup for a configuration that cannot work as intended."""


# Which credential each selectable backend requires to function at all.
_REQUIREMENTS = {
    ("DATABASE_BACKEND", "supabase"): [
        ("SUPABASE_URL", config.SUPABASE_URL),
        ("SUPABASE_SERVICE_ROLE_KEY", config.SUPABASE_SERVICE_ROLE_KEY),
    ],
    ("VECTOR_DB_BACKEND", "qdrant"): [
        ("QDRANT_URL", config.QDRANT_URL),
        ("QDRANT_API_KEY", config.QDRANT_API_KEY),
    ],
    ("EMBEDDING_BACKEND", "openai"): [
        ("OPENAI_API_KEY", config.OPENAI_API_KEY),
    ],
}

_LLM_KEYS = {
    "openai": ("OPENAI_API_KEY", config.OPENAI_API_KEY),
    "groq": ("GROQ_API_KEY", config.GROQ_API_KEY),
    "gemini": ("GEMINI_API_KEY", config.GEMINI_API_KEY),
    "anthropic": ("ANTHROPIC_API_KEY", config.ANTHROPIC_API_KEY),
}

# What a deployment must be pointed at. These are the backends that survive a restart;
# the defaults do not.
_PRODUCTION_BACKENDS = {
    "DATABASE_BACKEND": ("supabase", config.DATABASE_BACKEND,
                         "SQLite lives on the container's own disk and is destroyed on "
                         "every restart and redeploy"),
    "VECTOR_DB_BACKEND": ("qdrant", config.VECTOR_DB_BACKEND,
                          "Chroma reads data/chroma_db, which is excluded from the image "
                          "by .dockerignore - retrieval would return nothing"),
    "EMBEDDING_BACKEND": ("openai", config.EMBEDDING_BACKEND,
                          "the Qdrant collections are 1536-dimension OpenAI vectors; "
                          "local embeddings are 384 and every query would fail"),
}


def _consistency_problems() -> List[str]:
    problems = []
    for (setting, selected), needed in _REQUIREMENTS.items():
        if getattr(config, setting) != selected:
            continue
        for name, value in needed:
            if not value:
                problems.append(
                    f"{setting}={selected} but {name} is empty - that backend cannot "
                    f"connect without it."
                )

    provider = config.LLM_PROVIDER
    if provider in _LLM_KEYS:
        name, value = _LLM_KEYS[provider]
        if not value:
            problems.append(f"LLM_PROVIDER={provider} but {name} is empty.")
    else:
        problems.append(
            f"LLM_PROVIDER={provider!r} is not one of {sorted(_LLM_KEYS)}."
        )
    return problems


def _production_problems() -> List[str]:
    problems = []
    for setting, (expected, actual, why) in _PRODUCTION_BACKENDS.items():
        if actual != expected:
            problems.append(
                f"APP_ENV=production but {setting}={actual!r} (expected {expected!r}) - "
                f"{why}."
            )
    if not config.RESEND_API_KEY:
        problems.append(
            "APP_ENV=production but RESEND_API_KEY is empty - a confirmed support ticket "
            "would be written to the database with nobody notified."
        )
    localhost = [o for o in config.CORS_ORIGINS if "localhost" in o or "127.0.0.1" in o]
    if localhost:
        problems.append(
            f"APP_ENV=production but CORS_ORIGINS still contains {localhost} - the real "
            f"frontend origin is probably missing."
        )
    return problems



# Production-only resources. A process that is not production has no business writing to
# any of these: real customer conversations, the live FAQ index, and an email sender that
# reaches real inboxes.
_PRODUCTION_RESOURCES = (
    ("DATABASE_BACKEND", "supabase",
     "your test conversations would be written into the live customer database, where "
     "they corrupt the metrics and are indistinguishable from real traffic"),
    ("VECTOR_DB_BACKEND", "qdrant",
     "an ingest from this process would overwrite the live FAQ index that production "
     "serves from"),
)


def _local_pointing_at_production_problems() -> List[str]:
    """Problems for a NON-production process configured against production resources.

    The mirror of _production_problems(). That one stops production starting with
    throwaway backends; this stops a laptop starting with the real ones.

    Opt out with ALLOW_PRODUCTION_BACKENDS=1 when it is genuinely wanted - the point is
    that it becomes a decision someone made, not something they did not notice.
    """
    import os

    if os.getenv("ALLOW_PRODUCTION_BACKENDS", "").lower() in ("1", "true", "yes"):
        return []

    problems = []
    for setting, production_value, why in _PRODUCTION_RESOURCES:
        if getattr(config, setting) == production_value:
            problems.append(
                f"APP_ENV={config.APP_ENV!r} (not production) but {setting}="
                f"{production_value!r} - {why}. Use the defaults in .env.example for "
                f"local work, or set ALLOW_PRODUCTION_BACKENDS=1 if you really mean it."
            )
    if config.RESEND_API_KEY:
        problems.append(
            f"APP_ENV={config.APP_ENV!r} (not production) but RESEND_API_KEY is set - a "
            "test ticket would email the real support inbox. Leave it empty locally, or "
            "set ALLOW_PRODUCTION_BACKENDS=1 if you really mean it."
        )
    return problems


def validate_configuration(app_env: str) -> None:
    """Raises ConfigurationError listing everything wrong, rather than the first thing.

    One message with the whole picture beats three deploys each revealing one more
    missing variable.
    """
    problems = _consistency_problems()
    if app_env == "production":
        problems += _production_problems()
    else:
        # The mirror check: a laptop must not quietly write to production. See
        # _local_pointing_at_production_problems() for the two incidents behind this.
        problems += _local_pointing_at_production_problems()

    if problems:
        lines = "\n".join(f"  - {p}" for p in problems)
        raise ConfigurationError(
            f"\n\nRefusing to start: {len(problems)} configuration problem(s).\n{lines}\n\n"
            f"These are checked at startup on purpose. Every one of them would otherwise "
            f"let the service start, pass its health check, and fail silently.\n"
        )
