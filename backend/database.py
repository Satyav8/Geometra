import json
import sqlite3
from contextlib import contextmanager
from typing import List, Optional

from config import SQLITE_DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id      TEXT PRIMARY KEY,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    total_turns     INTEGER DEFAULT 0,
    resolved        BOOLEAN DEFAULT FALSE
);

CREATE TABLE IF NOT EXISTS messages (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id          TEXT NOT NULL,
    turn_number         INTEGER NOT NULL,
    query               TEXT NOT NULL,
    response            TEXT NOT NULL,
    retrieved_chunks    TEXT,
    similarity_scores   TEXT,
    confidence_level    TEXT,
    is_unknown_question BOOLEAN DEFAULT FALSE,
    response_latency_ms INTEGER,
    input_tokens        INTEGER,
    output_tokens        INTEGER,
    created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
);

CREATE TABLE IF NOT EXISTS unknown_questions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      TEXT NOT NULL,
    query           TEXT NOT NULL,
    similarity_score REAL,
    reviewed        BOOLEAN DEFAULT FALSE,
    answer          TEXT,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS evaluation_logs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    message_id      INTEGER NOT NULL,
    metric_name     TEXT NOT NULL,
    passed          BOOLEAN NOT NULL,
    score           REAL,
    detail          TEXT,
    FOREIGN KEY (message_id) REFERENCES messages(id)
);
"""


def get_connection():
    conn = sqlite3.connect(SQLITE_DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


@contextmanager
def get_cursor(commit: bool = False):
    conn = get_connection()
    try:
        cur = conn.cursor()
        yield cur
        if commit:
            conn.commit()
    finally:
        conn.close()


def init_db():
    conn = get_connection()
    try:
        conn.executescript(SCHEMA)
        conn.commit()
    finally:
        conn.close()


def ensure_session(session_id: str) -> None:
    with get_cursor(commit=True) as cur:
        cur.execute(
            "INSERT OR IGNORE INTO sessions (session_id) VALUES (?)", (session_id,)
        )


def increment_session_turns(session_id: str) -> None:
    with get_cursor(commit=True) as cur:
        cur.execute(
            "UPDATE sessions SET total_turns = total_turns + 1 WHERE session_id = ?",
            (session_id,),
        )


def write_message(
    session_id: str,
    turn_number: int,
    query: str,
    response: str,
    retrieved_chunk_ids: List[str],
    similarity_scores: List[float],
    confidence_level: str,
    is_unknown_question: bool,
    response_latency_ms: int,
    input_tokens: int,
    output_tokens: int,
) -> int:
    with get_cursor(commit=True) as cur:
        cur.execute(
            """
            INSERT INTO messages (
                session_id, turn_number, query, response, retrieved_chunks,
                similarity_scores, confidence_level, is_unknown_question,
                response_latency_ms, input_tokens, output_tokens
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                turn_number,
                query,
                response,
                json.dumps(retrieved_chunk_ids),
                json.dumps(similarity_scores),
                confidence_level,
                is_unknown_question,
                response_latency_ms,
                input_tokens,
                output_tokens,
            ),
        )
        return cur.lastrowid


def write_unknown_question(session_id: str, query: str, similarity_score: float) -> None:
    with get_cursor(commit=True) as cur:
        cur.execute(
            """
            INSERT INTO unknown_questions (session_id, query, similarity_score)
            VALUES (?, ?, ?)
            """,
            (session_id, query, similarity_score),
        )


def write_evaluation_logs(message_id: int, results) -> None:
    with get_cursor(commit=True) as cur:
        for r in results:
            cur.execute(
                """
                INSERT INTO evaluation_logs (message_id, metric_name, passed, score, detail)
                VALUES (?, ?, ?, ?, ?)
                """,
                (message_id, r.metric_name, r.passed, r.score, r.detail),
            )


def get_session_messages(session_id: str) -> List[dict]:
    with get_cursor() as cur:
        cur.execute(
            "SELECT * FROM messages WHERE session_id = ? ORDER BY turn_number ASC",
            (session_id,),
        )
        rows = cur.fetchall()
        return [dict(row) for row in rows]


def get_unreviewed_unknown_questions() -> List[dict]:
    with get_cursor() as cur:
        cur.execute(
            "SELECT * FROM unknown_questions WHERE reviewed = FALSE ORDER BY created_at DESC"
        )
        rows = cur.fetchall()
        return [dict(row) for row in rows]


if __name__ == "__main__":
    init_db()
    print(f"SQLite database initialized at {SQLITE_DB_PATH}")
