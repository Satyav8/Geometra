"""Checks whether a Supabase project can actually run this app, before you point at it.

Run it against whatever project you plan to use:

    SUPABASE_URL=https://xxx.supabase.co SUPABASE_SERVICE_ROLE_KEY=sb_secret_... \
        python preflight_supabase.py

Why this exists. Switching Supabase URL is usually harmless - the app speaks plain REST
and does not care which project answers. What is NOT harmless is a project whose schema
differs, and that is not hypothetical: a second project prepared for this app had
`sessions.user_id` declared NOT NULL with a foreign key to auth.users. Pointing the app at
it produced no startup error at all. Instead:

    ensure_session()      400, silently swallowed - it does not check its status
    write_message()       409, foreign key violation, three steps later
    set_session_state()   PATCH matched no rows, silently did nothing
    check_health()        reported healthy throughout

One missing column turned into a confusing 409 and a green health check. This script finds
that in ten seconds instead.

Read-only apart from one round-trip write, which it removes afterwards.
"""
import os
import sys
import uuid

import requests

URL = os.getenv("SUPABASE_URL", "").rstrip("/")
KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")

if not URL or not KEY:
    sys.exit("Set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY first.")

H = {"apikey": KEY, "Authorization": f"Bearer {KEY}", "Content-Type": "application/json"}

# Every table the app writes to, and the columns it relies on existing.
REQUIRED = {
    "sessions": ["session_id", "created_at", "total_turns", "resolved",
                 "awaiting", "pending_ticket_query", "pending_ticket_similarity"],
    "messages": ["session_id", "turn_number", "query", "response", "retrieved_chunks",
                 "similarity_scores", "confidence_level", "is_unknown_question",
                 "response_latency_ms", "input_tokens", "output_tokens"],
    "unknown_questions": ["session_id", "query", "similarity_score", "reviewed", "answer"],
    "escalated_questions": ["question", "criticality", "similarity_score", "session_id",
                            "turn_number", "reviewed"],
    "evaluation_logs": ["message_id", "metric_name", "passed", "score", "detail"],
}

problems = []
print(f"project: {URL}\n")

# ---- 1. do the tables exist, and can this key see them
print("tables")
for table in REQUIRED:
    r = requests.get(f"{URL}/rest/v1/{table}", headers=H,
                     params={"select": "*", "limit": 1}, timeout=20)
    if r.status_code != 200:
        problems.append(f"{table}: HTTP {r.status_code} {r.text[:90]}")
        print(f"  {table:<22} HTTP {r.status_code}")
    else:
        print(f"  {table:<22} reachable")

# ---- 2. the write path, which is where a schema difference actually shows up.
# A real insert is the only way to learn about a NOT NULL column the app never sets.
print("\nwrite path (this is what catches an extra required column)")
sid = f"PREFLIGHT-{uuid.uuid4()}"
r = requests.post(f"{URL}/rest/v1/sessions", headers={**H, "Prefer": "return=representation"},
                  json={"session_id": sid}, timeout=20)
if r.status_code in (200, 201):
    row = r.json()[0] if r.json() else {}
    print(f"  sessions insert        OK")
    missing = [c for c in REQUIRED["sessions"] if c not in row]
    if missing:
        problems.append(f"sessions is missing columns the app uses: {missing}")
        print(f"  sessions columns       MISSING {missing}")
    else:
        print(f"  sessions columns       all {len(REQUIRED['sessions'])} present")

    # messages has a foreign key to sessions, so this also proves the FK is satisfiable
    m = requests.post(f"{URL}/rest/v1/messages", headers={**H, "Prefer": "return=representation"},
                      json={"session_id": sid, "turn_number": 1, "query": "preflight",
                            "response": "preflight"}, timeout=20)
    if m.status_code in (200, 201):
        print("  messages insert        OK")
        mrow = m.json()[0] if m.json() else {}
        miss = [c for c in REQUIRED["messages"] if c not in mrow]
        if miss:
            problems.append(f"messages is missing columns the app uses: {miss}")
            print(f"  messages columns       MISSING {miss}")
        else:
            print(f"  messages columns       all {len(REQUIRED['messages'])} present")
        requests.delete(f"{URL}/rest/v1/messages", headers=H,
                        params={"session_id": f"eq.{sid}"}, timeout=20)
    else:
        problems.append(f"messages insert failed: HTTP {m.status_code} {m.text[:140]}")
        print(f"  messages insert        HTTP {m.status_code} {m.text[:110]}")

    # state round-trip: the ticket and clarification flows depend entirely on this
    requests.patch(f"{URL}/rest/v1/sessions", headers=H, params={"session_id": f"eq.{sid}"},
                   json={"awaiting": "ticket_confirmation"}, timeout=20)
    g = requests.get(f"{URL}/rest/v1/sessions", headers=H,
                     params={"session_id": f"eq.{sid}", "select": "awaiting"}, timeout=20)
    got = (g.json() or [{}])[0].get("awaiting")
    if got == "ticket_confirmation":
        print("  session state          round-trips OK")
    else:
        problems.append(f"session state did not persist (awaiting came back {got!r}) - "
                        f"the ticket and clarification flows will not work")
        print(f"  session state          BROKEN, got {got!r}")

    requests.delete(f"{URL}/rest/v1/sessions", headers=H,
                    params={"session_id": f"eq.{sid}"}, timeout=20)
    print("  cleanup                done")
else:
    body = r.text[:200]
    problems.append(f"cannot insert into sessions: HTTP {r.status_code} {body}")
    print(f"  sessions insert        HTTP {r.status_code}")
    print(f"    {body}")
    if "row-level security" in body:
        print("    -> this key is blocked by RLS. Use the service_role (secret) key, "
              "not the publishable/anon one.")
    if "null value in column" in body:
        print("    -> this project's schema requires a column the app never sets. "
              "Compare it against data/supabase_deploy_schema.sql.")

print()
if problems:
    print(f"NOT READY - {len(problems)} problem(s):")
    for p in problems:
        print(f"  - {p}")
    sys.exit(1)
print("READY - this project can run the app.")
