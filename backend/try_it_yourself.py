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
    """2. NO HALLUCINATION: If the context does not contain enough information to answer
   the question accurately, respond with EXACTLY this fallback message:
   "I don't have enough information about the question that you have asked.
   You can contact our support team through email."
   Do not add anything else to this fallback. Do not guess.""",
    """2. TOO VAGUE TO ANSWER: If you genuinely cannot tell what the customer is asking
   (not because the FAQ lacks the answer, but because their message doesn't say
   enough), do not guess. Respond with EXACTLY: [CLARIFY] <one specific question>

2B. CONTEXT DOESN'T COVER THIS: If the question is clear but the retrieved context
   genuinely doesn't address it, do not guess. Respond with EXACTLY: [CANNOT_ANSWER]
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


def has_hedge(text: str) -> bool:
    lowered = text.lower()
    return any(w in lowered for w in HEDGE_WORDS)


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


def answer_pass(original_query, intent, chunks, confidence, hedge_retry=False):
    # No raw conversation history here, by design - the diagram only feeds history into
    # Pass 1. Pass 2 relies on Pass 1's distilled intent summary instead, so this
    # actually tests whether Pass 1's reformulation carries enough context on its own.
    context = "\n\n".join(f"[Source: {c.section}]\n{c.text}" for c in chunks)
    prefix = "[LOW CONFIDENCE]\n" if confidence == "low" else ""
    retry_note = (
        "\nNOTE: your previous attempt used hedging language (e.g. 'perhaps', 'it seems'). "
        "Answer plainly and directly this time, with no hedge words.\n" if hedge_retry else ""
    )
    user_message = (
        f"{prefix}Customer's likely intent: {intent}\n{retry_note}\n"
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

    if awaiting == "ticket_confirmation":
        if query.strip().lower() in ("yes", "y", "yeah", "sure", "ok", "okay"):
            return "[TEST] Ticket would be raised here — last 3 turns emailed via Resend.", None
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
    response = answer_pass(query, intent, chunks, confidence)
    if has_hedge(response):
        response = answer_pass(query, intent, chunks, confidence, hedge_retry=True)
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
