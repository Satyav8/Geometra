"""Guards the startup configuration check.

It exists because every *_BACKEND in config.py defaults to the local-development choice.
That is correct on a laptop and dangerous in a container: a deployment that loses its
environment variables starts cleanly, passes its health check, writes every conversation to
a SQLite file on the container's own disk, and loses all of it on the next restart.

Both halves matter. Too lax and the silent-data-loss deploy still happens; too strict and
local development and this very test suite stop working - which is why the production
assertions are gated on APP_ENV and the consistency ones are not.
"""
import importlib

import dotenv
import pytest

import config
import startup_checks

_MANAGED = ("APP_ENV", "DATABASE_BACKEND", "VECTOR_DB_BACKEND", "EMBEDDING_BACKEND",
            "LLM_PROVIDER", "SUPABASE_URL", "SUPABASE_SERVICE_ROLE_KEY",
            "QDRANT_URL", "QDRANT_API_KEY", "OPENAI_API_KEY", "GROQ_API_KEY",
            "RESEND_API_KEY", "CORS_ORIGINS")


def _reload_with(monkeypatch, **env):
    """Rebuild config and startup_checks under a specific environment.

    dotenv is neutralised at its source, not on config: config.py does
    `from dotenv import load_dotenv` then calls it with override=True, so reloading config
    re-imports the real function and the developer's own .env would win over everything
    set here - the tests would silently assert against this machine rather than the case
    they describe.
    """
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **k: False)
    for key in _MANAGED:
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    importlib.reload(config)
    importlib.reload(startup_checks)
    return startup_checks


@pytest.fixture(autouse=True, scope="module")
def _restore_real_config():
    """Puts config back the way the rest of the suite expects.

    These tests reload config under fake environments. config is a module - everything
    that imported a value from it keeps the old one, but anything reading it later would
    see the fake. Right now this file happens to run last alphabetically, which is luck,
    not a guarantee. Reloading from the real environment on the way out removes the
    dependency on test ordering.
    """
    yield
    importlib.reload(config)
    importlib.reload(startup_checks)


PROD = dict(
    APP_ENV="production",
    DATABASE_BACKEND="supabase", SUPABASE_URL="https://x.supabase.co",
    SUPABASE_SERVICE_ROLE_KEY="sb_secret_x",
    VECTOR_DB_BACKEND="qdrant", QDRANT_URL="https://x.qdrant.io", QDRANT_API_KEY="k",
    EMBEDDING_BACKEND="openai", OPENAI_API_KEY="sk-x",
    LLM_PROVIDER="openai", RESEND_API_KEY="re_x",
    CORS_ORIGINS="https://app.geometra.in",
)


def test_a_fully_configured_production_deploy_starts(monkeypatch):
    sc = _reload_with(monkeypatch, **PROD)
    sc.validate_configuration("production")          # must not raise


def test_container_with_no_environment_is_refused(monkeypatch):
    """The exact failure this guard exists for: defaults everywhere, looks healthy."""
    sc = _reload_with(monkeypatch, APP_ENV="production", LLM_PROVIDER="openai",
                      OPENAI_API_KEY="sk-x")
    with pytest.raises(sc.ConfigurationError) as e:
        sc.validate_configuration("production")
    message = str(e.value)
    assert "DATABASE_BACKEND" in message      # would have silently used SQLite
    assert "VECTOR_DB_BACKEND" in message     # would have found no FAQ at all
    assert "EMBEDDING_BACKEND" in message     # 384-dim vectors against 1536-dim collections


@pytest.mark.parametrize("dropped,expected", [
    ("SUPABASE_SERVICE_ROLE_KEY", "SUPABASE_SERVICE_ROLE_KEY"),
    ("SUPABASE_URL", "SUPABASE_URL"),
    ("QDRANT_API_KEY", "QDRANT_API_KEY"),
    ("OPENAI_API_KEY", "OPENAI_API_KEY"),
])
def test_selected_backend_without_its_credential_is_refused(monkeypatch, dropped, expected):
    env = {k: v for k, v in PROD.items() if k != dropped}
    sc = _reload_with(monkeypatch, **env)
    with pytest.raises(sc.ConfigurationError) as e:
        sc.validate_configuration("production")
    assert expected in str(e.value)


def test_localhost_cors_in_production_is_refused(monkeypatch):
    sc = _reload_with(monkeypatch, **{**PROD, "CORS_ORIGINS": "http://localhost:5173"})
    with pytest.raises(sc.ConfigurationError) as e:
        sc.validate_configuration("production")
    assert "CORS_ORIGINS" in str(e.value)


def test_missing_resend_key_in_production_is_refused(monkeypatch):
    """A confirmed ticket would be stored with nobody notified."""
    env = {k: v for k, v in PROD.items() if k != "RESEND_API_KEY"}
    sc = _reload_with(monkeypatch, **env)
    with pytest.raises(sc.ConfigurationError) as e:
        sc.validate_configuration("production")
    assert "RESEND_API_KEY" in str(e.value)


def test_local_development_defaults_are_left_alone(monkeypatch):
    """SQLite + Chroma + local embeddings is the RIGHT answer on a laptop. The production
    assertions must not fire here, or development and CI both break."""
    sc = _reload_with(monkeypatch, LLM_PROVIDER="groq", GROQ_API_KEY="gsk_x")
    sc.validate_configuration("development")         # must not raise


def test_consistency_is_enforced_even_outside_production(monkeypatch):
    """Selecting supabase with no key is wrong on a laptop too."""
    sc = _reload_with(monkeypatch, DATABASE_BACKEND="supabase",
                      LLM_PROVIDER="groq", GROQ_API_KEY="gsk_x")
    with pytest.raises(sc.ConfigurationError) as e:
        sc.validate_configuration("development")
    assert "SUPABASE_URL" in str(e.value)


def test_unknown_llm_provider_is_refused(monkeypatch):
    sc = _reload_with(monkeypatch, LLM_PROVIDER="llama-on-my-toaster")
    with pytest.raises(sc.ConfigurationError) as e:
        sc.validate_configuration("development")
    assert "LLM_PROVIDER" in str(e.value)


def test_every_problem_is_reported_at_once(monkeypatch):
    """One deploy should reveal the whole list, not the first item eight times."""
    sc = _reload_with(monkeypatch, APP_ENV="production", LLM_PROVIDER="openai")
    with pytest.raises(sc.ConfigurationError) as e:
        sc.validate_configuration("production")
    assert str(e.value).count("  - ") >= 5
