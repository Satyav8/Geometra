"""Guards against a non-production process writing to production data.

WHY THIS EXISTS

Two incidents in one day, neither a code bug, both costing an afternoon:

  Replies carrying pre-fix behaviour appeared in the production database for 21 hours
  after the fix had deployed. The fix looked broken in the data while production itself
  was fine. The signature was a build that was not the deployed one, writing into the
  live Supabase.

  A teammate cloned the repo, ran it locally, and reported that on-topic questions were
  refused. Her traffic never reached production at all - her local vector store was
  simply empty.

What they share is that the misconfiguration is INVISIBLE. The service starts, the health
check passes, and the only symptom is a bug report about a system nobody is looking at.
No test catches that, because nothing is wrong with the code.

So the rule is the mirror of the production guard that already exists: production must not
start with throwaway backends, and a non-production process must not start with the real
ones. The opt-out is deliberate - ALLOW_PRODUCTION_BACKENDS=1 makes it a decision someone
took rather than something they did not notice.

The tests set attributes on the config module directly, because config reads its values
at import and load_dotenv(override=True) means a real .env would otherwise decide the
outcome. An earlier version of this test ran in the repo and was silently judged against
the real .env - three cases passed only because that file happened to hold production
values.
"""
import pytest

import startup_checks as sc
from startup_checks import ConfigurationError


@pytest.fixture
def local(monkeypatch):
    """A clean, safe local configuration - the state a fresh clone should be in."""
    monkeypatch.setattr(sc.config, "APP_ENV", "development")
    monkeypatch.setattr(sc.config, "LLM_PROVIDER", "groq")
    monkeypatch.setattr(sc.config, "GROQ_API_KEY", "x")
    monkeypatch.setattr(sc.config, "DATABASE_BACKEND", "sqlite")
    monkeypatch.setattr(sc.config, "VECTOR_DB_BACKEND", "chroma")
    monkeypatch.setattr(sc.config, "EMBEDDING_BACKEND", "local")
    monkeypatch.setattr(sc.config, "RESEND_API_KEY", "")
    monkeypatch.setenv("ALLOW_PRODUCTION_BACKENDS", "")
    return monkeypatch


def test_an_ordinary_local_run_starts(local):
    """The case that matters most for adoption. A guard that blocks normal work gets
    switched off by the first person it annoys, and then protects nothing."""
    sc.validate_configuration("development")


def test_local_pointed_at_the_production_database_is_refused(local):
    local.setattr(sc.config, "DATABASE_BACKEND", "supabase")
    local.setattr(sc.config, "SUPABASE_URL", "https://x.supabase.co")
    local.setattr(sc.config, "SUPABASE_SERVICE_ROLE_KEY", "k")
    with pytest.raises(ConfigurationError) as excinfo:
        sc.validate_configuration("development")
    assert "live customer database" in str(excinfo.value)


def test_local_pointed_at_the_production_vector_store_is_refused(local):
    """The most damaging one: an ingest from a laptop would replace the FAQ index that
    production serves from."""
    local.setattr(sc.config, "VECTOR_DB_BACKEND", "qdrant")
    local.setattr(sc.config, "EMBEDDING_BACKEND", "openai")
    local.setattr(sc.config, "OPENAI_API_KEY", "k")
    local.setattr(sc.config, "QDRANT_URL", "https://x.qdrant.io")
    local.setattr(sc.config, "QDRANT_API_KEY", "k")
    with pytest.raises(ConfigurationError) as excinfo:
        sc.validate_configuration("development")
    assert "overwrite the live FAQ" in str(excinfo.value)


def test_local_with_a_real_email_key_is_refused(local):
    local.setattr(sc.config, "RESEND_API_KEY", "re_live_key")
    with pytest.raises(ConfigurationError) as excinfo:
        sc.validate_configuration("development")
    assert "real support inbox" in str(excinfo.value)


def test_the_message_says_what_to_do_instead(local):
    """A refusal that does not tell you the fix is just an obstacle."""
    local.setattr(sc.config, "DATABASE_BACKEND", "supabase")
    local.setattr(sc.config, "SUPABASE_URL", "https://x.supabase.co")
    local.setattr(sc.config, "SUPABASE_SERVICE_ROLE_KEY", "k")
    with pytest.raises(ConfigurationError) as excinfo:
        sc.validate_configuration("development")
    message = str(excinfo.value)
    assert ".env.example" in message
    assert "ALLOW_PRODUCTION_BACKENDS" in message


def test_the_opt_out_works(local):
    """Pointing a local process at production is sometimes genuinely wanted. The point is
    that it becomes explicit, not that it becomes impossible."""
    local.setattr(sc.config, "DATABASE_BACKEND", "supabase")
    local.setattr(sc.config, "SUPABASE_URL", "https://x.supabase.co")
    local.setattr(sc.config, "SUPABASE_SERVICE_ROLE_KEY", "k")
    local.setenv("ALLOW_PRODUCTION_BACKENDS", "1")
    sc.validate_configuration("development")


def test_production_is_unaffected(monkeypatch):
    """The new check must never run for the deploy it would block entirely."""
    monkeypatch.setattr(sc.config, "APP_ENV", "production")
    monkeypatch.setattr(sc.config, "LLM_PROVIDER", "openai")
    monkeypatch.setattr(sc.config, "OPENAI_API_KEY", "k")
    monkeypatch.setattr(sc.config, "DATABASE_BACKEND", "supabase")
    monkeypatch.setattr(sc.config, "SUPABASE_URL", "https://x.supabase.co")
    monkeypatch.setattr(sc.config, "SUPABASE_SERVICE_ROLE_KEY", "k")
    monkeypatch.setattr(sc.config, "VECTOR_DB_BACKEND", "qdrant")
    monkeypatch.setattr(sc.config, "QDRANT_URL", "https://x.qdrant.io")
    monkeypatch.setattr(sc.config, "QDRANT_API_KEY", "k")
    monkeypatch.setattr(sc.config, "EMBEDDING_BACKEND", "openai")
    monkeypatch.setattr(sc.config, "RESEND_API_KEY", "re_live")
    monkeypatch.setattr(sc.config, "CORS_ORIGINS", ["https://geometra-frontend.onrender.com"])
    monkeypatch.setenv("ALLOW_PRODUCTION_BACKENDS", "")
    # _PRODUCTION_BACKENDS snapshots config at IMPORT time, so patching the config module
    # does not reach it - the snapshot has to be rebuilt for the patched values to apply.
    # Worth knowing if this module is ever refactored: the production check reads
    # import-time values, which is fine at runtime (config is read once anyway) and is
    # exactly what makes it awkward to test.
    monkeypatch.setattr(sc, "_PRODUCTION_BACKENDS", {
        "DATABASE_BACKEND": ("supabase", "supabase", "x"),
        "VECTOR_DB_BACKEND": ("qdrant", "qdrant", "x"),
        "EMBEDDING_BACKEND": ("openai", "openai", "x"),
    })
    sc.validate_configuration("production")


def test_all_problems_are_reported_at_once(local):
    """Same principle as the production guard: one message with the whole picture beats
    three restarts each revealing one more thing."""
    local.setattr(sc.config, "DATABASE_BACKEND", "supabase")
    local.setattr(sc.config, "SUPABASE_URL", "https://x.supabase.co")
    local.setattr(sc.config, "SUPABASE_SERVICE_ROLE_KEY", "k")
    local.setattr(sc.config, "RESEND_API_KEY", "re_live_key")
    with pytest.raises(ConfigurationError) as excinfo:
        sc.validate_configuration("development")
    message = str(excinfo.value)
    assert "live customer database" in message and "real support inbox" in message
