from typing import List

import requests

from config import RESEND_API_KEY, RESEND_FROM_EMAIL, SUPPORT_EMAIL

API_URL = "https://api.resend.com/emails"


def _render_transcript_html(messages: List[dict]) -> str:
    turns = []
    for m in messages:
        turns.append(
            f"<p><strong>Turn {m['turn_number']}</strong><br>"
            f"<strong>Customer:</strong> {m['query']}<br>"
            f"<strong>S.A.M:</strong> {m['response']}</p>"
        )
    return "\n".join(turns)


def send_ticket_email(ticket_number: str, session_id: str, messages: List[dict]) -> None:
    """Emails the full conversation transcript to SUPPORT_EMAIL as a support ticket.
    No-ops if Resend isn't configured, and never raises — an email hiccup must not
    break the customer-facing chat response."""
    if not RESEND_API_KEY:
        return

    payload = {
        "from": RESEND_FROM_EMAIL,
        "to": [SUPPORT_EMAIL],
        "subject": f"[S.A.M] Support Ticket {ticket_number} — session {session_id}",
        "html": (
            f"<p>A customer question needs a human answer.</p>"
            f"<p><strong>Ticket:</strong> {ticket_number}<br>"
            f"<strong>Session:</strong> {session_id}</p>"
            f"<hr>{_render_transcript_html(messages)}"
        ),
    }
    headers = {
        "Authorization": f"Bearer {RESEND_API_KEY}",
        "Content-Type": "application/json",
    }

    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=10)
        if response.status_code >= 400:
            print(f"[resend] failed to send ticket email: {response.status_code} {response.text}")
    except requests.RequestException as e:
        print(f"[resend] failed to send ticket email: {e}")
