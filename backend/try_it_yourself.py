"""Interactive local test of the two-pass LLM flow (Understand -> fast-path scope check
-> Retrieve -> Answer/Refine) per dev-handoff-two-pass-flow.md. NOT part of the real app -
throwaway test tool, nothing here is wired into routers/chat.py or committed behavior.
Run: ./venv/Scripts/python.exe try_it_yourself.py
Type 'exit' to quit.
"""
import sys

sys.path.insert(0, ".")
from rag.retriever import retrieve
from rag.relevance import is_gratitude, is_greeting, is_query_relevant
from llm.client import call_llm
from llm.prompts import SYSTEM_PROMPT
from config import GRATITUDE_MESSAGE, GREETING_MESSAGE, OUT_OF_SCOPE_MESSAGE

# Loosened per the spec: fast-path check is keyword hit OR similarity >= 0.15 (was 0.30
# as the sole gate before). Below this AND no keyword hit -> out of scope, no LLM spent.
FAST_PATH_SIMILARITY = 0.15

UNDERSTAND_PROMPT = """You are the "Understand" stage of a two-pass customer support
pipeline for S.A.M, Geometra's chatbot. Geometra is an image-to-CAD tool that measures
wall elevations from phone photos.

Given the customer's latest message and (if any) the recent conversation, produce
EXACTLY two lines and nothing else:
QUERY: <a self-contained search query capturing what the customer actually wants to
        know, resolving any pronouns/references using the conversation history>
INTENT: <one short line summarizing their intent>

Do not answer the question. If the message is already clear and self-contained, QUERY
can just restate it cleanly."""

# Rule 2 (and its "use the fallback in Rule 2" references in Rule 3 / LOW CONFIDENCE
# MODE) is replaced, not just supplemented - appending new rules on top of the old ones
# left Rule 2's original fallback text as a competing instruction, and the model
# sometimes followed the old one instead of the new [CLARIFY]/[CANNOT_ANSWER] tags.
_ANSWER_BASE = SYSTEM_PROMPT.replace(
    """1. ANSWER ONLY FROM CONTEXT: Answer exclusively from the retrieved knowledge base
   chunks provided in the CONTEXT section below. Never use outside knowledge.""",
    """1. ANSWER ONLY FROM CONTEXT: Answer exclusively from the retrieved knowledge base
   chunks provided in the CONTEXT section below. Never use outside knowledge.

1B. CHECK ALL CHUNKS FIRST: The CONTEXT section contains multiple chunks, ranked by
   relevance, not just one. Scan all of them before deciding whether you can answer. If
   two or more chunks together establish a direct answer (e.g. one states a general rule,
   another confirms it applies to this specific case), combine them into one direct,
   confident answer rather than treating the question as unclear or uncovered. This check
   happens before Rule 2 or 2B, not instead of them.""",
).replace(
    """2. NO HALLUCINATION: If the context does not contain enough information to answer
   the question accurately, respond with EXACTLY this fallback message:
   "I don't have enough information about the question that you have asked.
   You can contact our support team through email."
   Do not add anything else to this fallback. Do not guess.""",
    """2. TOO VAGUE TO ANSWER: If you genuinely cannot tell what the customer is asking
   (not because the FAQ lacks the answer, but because their message doesn't say
   enough), do not guess. Ask TWO distinct clarifying questions in this one turn
   (not one) so you gather more context in a single round-trip instead of going
   back and forth. Respond with EXACTLY:
   [CLARIFY] 1) <first specific question> 2) <second specific question>
   The two questions must be genuinely different angles on the ambiguity, not
   two phrasings of the same question.

2B. CONTEXT DOESN'T COVER THIS: Before using this rule, check EVERY chunk in the
   CONTEXT section below, not just the first or most-similar one — the right answer
   is often sitting in a lower-ranked chunk. If, after checking all of them, the
   question is clear but none of them address it, respond with EXACTLY:
   [CANNOT_ANSWER]
   Do not add anything else after this tag.""",
).replace(
    "If you are not certain\n   from the context, use the fallback message in Rule 2.",
    "If you are not certain from the context, use Rule 2B ([CANNOT_ANSWER])."
).replace(
    "When in doubt, use the fallback from Rule 2.",
    "When in doubt, use Rule 2 ([CLARIFY]) or 2B ([CANNOT_ANSWER])."
)

ANSWER_PROMPT = _ANSWER_BASE

HEDGE_WORDS = ["i think", "i believe", "probably", "i'm not sure", "it seems", "perhaps", "i suppose"]

# Exact-match on the whole reply missed common natural phrasings like "yes please" or
# "yeah sure" (round-2 testing, turn 12). Checking just the first word instead covers those
# without needing a full affirmative-intent classifier.
AFFIRMATIVE_WORDS = ("yes", "y", "yeah", "yep", "yup", "sure", "ok", "okay")


def has_hedge(text: str) -> bool:
    lowered = text.lower()
    return any(w in lowered for w in HEDGE_WORDS)


def is_affirmative(text: str) -> bool:
    words = text.strip().lower().split()
    if not words:
        return False
    first_word = words[0].strip(".!,")
    return first_word in AFFIRMATIVE_WORDS


# The "yes please" fix only fires while awaiting == "ticket_confirmation". Round-3 testing
# found a gap one layer up: if a clarification exchange happens *after* a ticket was
# offered, awaiting moves to "clarification" and the original offer is forgotten, so an
# explicit "yeah okay raise the ticket" a couple turns later falls through to a brand new
# Pass 1/2 cycle instead of just raising it. Catching an explicit ticket request by keyword
# regardless of current awaiting state closes that gap without needing to track every past
# offer. Deliberately simple - "ticket" + an affirmative/action word - not a full intent
# classifier, so it can still misfire on something like "is my ticket raised yet".
TICKET_ACTION_WORDS = ("raise", "open", "create", "file", "submit", "log")


def wants_ticket(text: str) -> bool:
    lowered = text.strip().lower()
    if "ticket" not in lowered:
        return False
    if lowered.split()[0].strip(".!,") in ("no", "not"):
        return False
    return is_affirmative(text) or any(w in lowered for w in TICKET_ACTION_WORDS)


def history_block(history, label="RECENT CONVERSATION"):
    if not history:
        return ""
    lines = "\n".join(f"{'Customer' if role == 'customer' else 'S.A.M'}: {text}" for role, text in history[-4:])
    return f"{label}:\n{lines}\n\n"


def understand(query, history):
    user_message = f"{history_block(history)}LATEST CUSTOMER MESSAGE: {query}"
    response, _, _ = call_llm(UNDERSTAND_PROMPT, user_message)
    reformulated, intent = query, ""
    for line in response.splitlines():
        stripped = line.strip()
        if stripped.upper().startswith("QUERY:"):
            reformulated = stripped.split(":", 1)[1].strip()
        elif stripped.upper().startswith("INTENT:"):
            intent = stripped.split(":", 1)[1].strip()
    return reformulated, intent


def answer_pass(original_query, intent, chunks, confidence, hedge_retry=False, already_clarified=False):
    # No raw conversation history here, by design - the diagram only feeds history into
    # Pass 1. Pass 2 relies on Pass 1's distilled intent summary instead, so this
    # actually tests whether Pass 1's reformulation carries enough context on its own.
    context = "\n\n".join(f"[Source: {c.section}]\n{c.text}" for c in chunks)
    prefix = "[LOW CONFIDENCE]\n" if confidence == "low" else ""
    retry_note = (
        "\nNOTE: your previous attempt used hedging language (e.g. 'perhaps', 'it seems'). "
        "Answer plainly and directly this time, with no hedge words.\n" if hedge_retry else ""
    )
    # Caps clarification at one round. Without this, a genuinely uncovered question (e.g.
    # refund policy, where the FAQ itself just says "refer our policy") could chain
    # clarifying question after clarifying question forever instead of ever reaching
    # [CANNOT_ANSWER] and offering a ticket - found via round-4 testing.
    clarify_cap_note = (
        "\nNOTE: the customer was already asked a clarifying question last turn. Do not "
        "ask another one. If this reply still isn't enough to answer from the context, "
        "use [CANNOT_ANSWER] instead of asking again.\n" if already_clarified else ""
    )
    user_message = (
        f"{prefix}Customer's likely intent: {intent}\n{retry_note}{clarify_cap_note}\n"
        f"CONTEXT:\n{context}\n\nCUSTOMER QUESTION: {original_query}"
    )
    response, _, _ = call_llm(ANSWER_PROMPT, user_message)
    return response


def process_turn(query, history, awaiting):
    """Returns (response_text, new_awaiting_state)."""
    if is_gratitude(query):
        return GRATITUDE_MESSAGE, None
    if is_greeting(query):
        return GREETING_MESSAGE, None

    TICKET_RAISED_MESSAGE = "[TEST] Ticket would be raised here — last 3 turns emailed via Resend."

    # These two checks must be independent, not if/elif - "alright raise a ticket for
    # this then" while awaiting == "ticket_confirmation" doesn't match is_affirmative
    # (first word "alright" isn't in AFFIRMATIVE_WORDS), but it clearly asks for a
    # ticket, so wants_ticket() must still get a chance to catch it (found via testing:
    # an elif here let that exact phrasing fall through to a fresh Pass 1/2 cycle).
    if awaiting == "ticket_confirmation" and is_affirmative(query):
        return TICKET_RAISED_MESSAGE, None
    if wants_ticket(query):
        return TICKET_RAISED_MESSAGE, None
    # anything else: clear awaiting, fall through and treat this message as a new question

    # Pass 1 — Understand. Always gets recent history (not just when awaiting ==
    # "clarification" as the original diagram showed) - stress-testing found that a
    # short follow-up referencing the previous NORMAL answer (not just a clarifying
    # question) also needs history to resolve correctly, e.g. "can it" right after an
    # answer about washbasins got no context and produced an unrelated guess. Pass 1 is
    # a cheap, short-output call, so always including the last couple of turns costs
    # very little and closes that gap.
    reformulated_query, intent = understand(query, history)

    # Fast-path scope check: ONE retrieve() call, reused for both the gate and Pass 2
    chunks, confidence = retrieve(reformulated_query)
    top1 = chunks[0].similarity_score if chunks else 0.0
    keyword_hit = is_query_relevant(query)
    if not keyword_hit and top1 < FAST_PATH_SIMILARITY:
        return OUT_OF_SCOPE_MESSAGE, None

    # Pass 2 — Answer / Refine
    already_clarified = awaiting == "clarification"
    response = answer_pass(query, intent, chunks, confidence, already_clarified=already_clarified)
    if has_hedge(response):
        response = answer_pass(query, intent, chunks, confidence, hedge_retry=True, already_clarified=already_clarified)
        # accepted as-is even if the retry still hedges (one retry only, per spec)

    stripped = response.strip()
    if stripped.startswith("[CLARIFY]"):
        return stripped[len("[CLARIFY]"):].strip(), "clarification"
    if stripped.startswith("[CANNOT_ANSWER]"):
        return (
            "I don't have enough information to answer that. Would you like me to raise "
            "a support ticket for you? (yes/no)"
        ), "ticket_confirmation"
    return response, None


def main():
    print("S.A.M two-pass test mode — type a question, 'exit' to quit.\n")
    history = []
    awaiting = None
    while True:
        query = input("You: ").strip()
        if not query:
            continue
        if query.lower() in ("exit", "quit"):
            break
        response, awaiting = process_turn(query, history, awaiting)
        tag = f" [awaiting={awaiting}]" if awaiting else ""
        print(f"S.A.M{tag}: {response}\n")
        history.append(("customer", query))
        history.append(("sam", response))


if __name__ == "__main__":
    main()
