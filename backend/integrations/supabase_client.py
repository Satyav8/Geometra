import requests

from config import SUPABASE_SERVICE_ROLE_KEY, SUPABASE_URL

TABLE = "escalated_questions"


def write_escalated_question(
    question: str,
    criticality: str,
    similarity_score: float,
    session_id: str,
    turn_number: int,
) -> None:
    """Logs a relevant-but-uncovered question for the team to review.
    No-ops if Supabase isn't configured yet, and never raises — a Supabase
    hiccup must not break the customer-facing chat response."""
    if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
        return

    url = f"{SUPABASE_URL}/rest/v1/{TABLE}"
    headers = {
        "apikey": SUPABASE_SERVICE_ROLE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=minimal",
    }
    payload = {
        "question": question,
        "criticality": criticality,
        "similarity_score": similarity_score,
        "session_id": session_id,
        "turn_number": turn_number,
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=10)
        if response.status_code >= 400:
            print(f"[supabase] failed to log escalated question: {response.status_code} {response.text}")
    except requests.RequestException as e:
        print(f"[supabase] failed to log escalated question: {e}")
