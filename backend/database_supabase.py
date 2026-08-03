import json
from typing import List, Optional

import requests

from config import SUPABASE_SERVICE_ROLE_KEY, SUPABASE_URL

BASE_HEADERS = {
    "apikey": SUPABASE_SERVICE_ROLE_KEY,
    "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
    "Content-Type": "application/json",
}


def _url(table: str) -> str:
    return f"{SUPABASE_URL}/rest/v1/{table}"


def init_db() -> None:
    # Tables are created once via backend/data/supabase_main_schema.sql in the
    # Supabase SQL Editor, not from application code.
    pass


def ensure_session(session_id: str) -> None:
    headers = {**BASE_HEADERS, "Prefer": "resolution=ignore-duplicates,return=minimal"}
    requests.post(_url("sessions"), headers=headers, json={"session_id": session_id}, timeout=10)


def increment_session_turns(session_id: str) -> None:
    resp = requests.get(
        _url("sessions"), headers=BASE_HEADERS,
        params={"session_id": f"eq.{session_id}", "select": "total_turns"}, timeout=10,
    )
    rows = resp.json() if resp.ok else []
    current = rows[0]["total_turns"] if rows else 0
    headers = {**BASE_HEADERS, "Prefer": "return=minimal"}
    requests.patch(
        _url("sessions"), headers=headers,
        params={"session_id": f"eq.{session_id}"},
        json={"total_turns": current + 1}, timeout=10,
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
    headers = {**BASE_HEADERS, "Prefer": "return=representation"}
    payload = {
        "session_id": session_id,
        "turn_number": turn_number,
        "query": query,
        "response": response,
        "retrieved_chunks": json.dumps(retrieved_chunk_ids),
        "similarity_scores": json.dumps(similarity_scores),
        "confidence_level": confidence_level,
        "is_unknown_question": is_unknown_question,
        "response_latency_ms": response_latency_ms,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }
    resp = requests.post(_url("messages"), headers=headers, json=payload, timeout=10)
    resp.raise_for_status()
    return resp.json()[0]["id"]


def write_unknown_question(session_id: str, query: str, similarity_score: float) -> int:
    headers = {**BASE_HEADERS, "Prefer": "return=representation"}
    payload = {"session_id": session_id, "query": query, "similarity_score": similarity_score}
    resp = requests.post(_url("unknown_questions"), headers=headers, json=payload, timeout=10)
    resp.raise_for_status()
    return resp.json()[0]["id"]


def write_evaluation_logs(message_id: int, results) -> None:
    headers = {**BASE_HEADERS, "Prefer": "return=minimal"}
    payload = [
        {
            "message_id": message_id,
            "metric_name": r.metric_name,
            "passed": r.passed,
            "score": r.score,
            "detail": r.detail,
        }
        for r in results
    ]
    resp = requests.post(_url("evaluation_logs"), headers=headers, json=payload, timeout=15)
    resp.raise_for_status()


def get_session_messages(session_id: str) -> List[dict]:
    params = {"session_id": f"eq.{session_id}", "order": "turn_number.asc"}
    resp = requests.get(_url("messages"), headers=BASE_HEADERS, params=params, timeout=10)
    resp.raise_for_status()
    return resp.json()


def get_unreviewed_unknown_questions() -> List[dict]:
    params = {"reviewed": "eq.false", "order": "created_at.desc"}
    resp = requests.get(_url("unknown_questions"), headers=BASE_HEADERS, params=params, timeout=10)
    resp.raise_for_status()
    return resp.json()


def get_session_message_flags(session_id: str) -> List[bool]:
    params = {"session_id": f"eq.{session_id}", "select": "is_unknown_question"}
    resp = requests.get(_url("messages"), headers=BASE_HEADERS, params=params, timeout=10)
    resp.raise_for_status()
    return [bool(r["is_unknown_question"]) for r in resp.json()]


def count_session_messages(session_id: str) -> int:
    headers = {**BASE_HEADERS, "Prefer": "count=exact"}
    params = {"session_id": f"eq.{session_id}", "select": "id"}
    resp = requests.get(_url("messages"), headers=headers, params=params, timeout=10)
    resp.raise_for_status()
    total = resp.headers.get("content-range", "*/0").split("/")[-1]
    return int(total) if total.isdigit() else len(resp.json())


def get_message_by_id(message_id: int) -> Optional[dict]:
    resp = requests.get(_url("messages"), headers=BASE_HEADERS, params={"id": f"eq.{message_id}"}, timeout=10)
    resp.raise_for_status()
    rows = resp.json()
    return rows[0] if rows else None


def check_health() -> bool:
    try:
        resp = requests.get(_url("sessions"), headers=BASE_HEADERS, params={"limit": 1}, timeout=10)
        return resp.ok
    except Exception:
        return False
