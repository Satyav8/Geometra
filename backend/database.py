"""Thin dispatcher: routes every DB call to either the local SQLite implementation
(default, used for local dev/tests) or the Supabase implementation (production,
set DATABASE_BACKEND=supabase). Every other module imports from here, never
directly from database_sqlite / database_supabase, so the rest of the codebase
doesn't need to know or care which backend is active."""

from config import DATABASE_BACKEND

if DATABASE_BACKEND == "supabase":
    from database_supabase import (
        check_health,
        count_session_messages,
        ensure_session,
        get_message_by_id,
        get_session_message_flags,
        get_session_messages,
        get_session_state,
        get_unreviewed_unknown_questions,
        increment_session_turns,
        init_db,
        set_session_state,
        write_evaluation_logs,
        write_message,
        write_unknown_question,
    )
else:
    from database_sqlite import (
        check_health,
        count_session_messages,
        ensure_session,
        get_message_by_id,
        get_session_message_flags,
        get_session_messages,
        get_session_state,
        get_unreviewed_unknown_questions,
        increment_session_turns,
        init_db,
        set_session_state,
        write_evaluation_logs,
        write_message,
        write_unknown_question,
    )

__all__ = [
    "check_health",
    "count_session_messages",
    "ensure_session",
    "get_message_by_id",
    "get_session_message_flags",
    "get_session_messages",
    "get_session_state",
    "get_unreviewed_unknown_questions",
    "increment_session_turns",
    "init_db",
    "set_session_state",
    "write_evaluation_logs",
    "write_message",
    "write_unknown_question",
]

if __name__ == "__main__":
    init_db()
    print(f"Database initialized (backend: {DATABASE_BACKEND})")
