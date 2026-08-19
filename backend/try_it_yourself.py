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
from rag.embedder import embed_text
from llm.client import call_llm
from llm.prompts import SYSTEM_PROMPT
from models import SourceChunk
from config import (
    GRATITUDE_MESSAGE,
    GREETING_MESSAGE,
    OUT_OF_SCOPE_MESSAGE,
    LOW_CONFIDENCE_THRESHOLD,
    MIN_SIMILARITY_SCORE,
)
import website_kb

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
    "STRICT RULES — you must follow all of these without exception:",
    """TONE: Talk like an attentive human support agent, not a script. Acknowledge what the
customer actually said, use natural phrasing, and sound genuinely interested in solving
their problem rather than reciting a manual. Never end a turn on a flat dead end — if you
can't fully answer, still leave the customer with a clear, warm next step. This applies
throughout every rule below, especially Rules 2 and 5.

STRICT RULES — you must follow all of these without exception:""",
).replace(
    """1. ANSWER ONLY FROM CONTEXT: Answer exclusively from the retrieved knowledge base
   chunks provided in the CONTEXT section below. Never use outside knowledge.""",
    """1. ANSWER ONLY FROM CONTEXT: Answer exclusively from the retrieved knowledge base
   chunks provided in the CONTEXT section below. Never use outside knowledge.

1B. CHECK ALL CHUNKS FIRST: The CONTEXT section contains multiple chunks, ranked by
   relevance, not just one. Scan all of them before deciding whether you can answer. If
   two or more chunks together establish a direct answer (e.g. one states a general rule,
   another confirms it applies to this specific case), combine them into one direct,
   confident answer rather than treating the question as unclear or uncovered. This check
   happens before Rule 2 or 2B, not instead of them.

1C. GENERALIZE ESTABLISHED RULES: If the context establishes a general rule (e.g. "each
   distinct depth on a wall needs its own marker"), apply it confidently to any
   structurally similar feature even if that exact feature isn't named in the context -
   a bay window, alcove, pillar, or recessed shelf all create a depth change just like a
   fireplace or windowsill do. A new example of an already-established general rule is
   not unclear or uncovered just because its specific name doesn't appear verbatim in the
   retrieved chunks.

1D. DON'T INFER UNSTATED CLAIMS: Only state what the context actually asserts - do not
   draw further conclusions that merely sound like a natural extension of it. Example:
   "measurements are calculated with math, not AI" does NOT mean "no human ever reviews
   the output" - those are separate, unrelated claims, and the second one isn't stated
   anywhere, so asserting it would be a guess dressed up as fact. This is different from
   Rule 1C: 1C applies an established RULE to a new, structurally similar CASE; this rule
   stops you from inventing a brand-new, unstated FACT that isn't actually a case of any
   established rule. If a question reaches for something adjacent to but not actually
   covered by the context, treat it as uncovered (Rule 2B) rather than inventing a
   plausible-sounding answer.""",
).replace(
    """2. NO HALLUCINATION: If the context does not contain enough information to answer
   the question accurately, respond with EXACTLY this fallback message:
   "I don't have enough information about the question that you have asked.
   You can contact our support team through email."
   Do not add anything else to this fallback. Do not guess.""",
    """2. TOO VAGUE TO ANSWER: If, after applying Rule 1B, you genuinely cannot tell what
   the customer is asking (not because the FAQ lacks the answer, but because their
   message doesn't say enough), do not guess. Ask TWO clarifying questions in this one
   turn, grounded in what the CONTEXT below actually contains - e.g. if it covers both
   a wardrobe and a washbasin scenario, ask which one they mean, not a generic "what do
   you want to measure?" Open with a brief, warm acknowledgment of what they did say.
   Respond with EXACTLY:
   [CLARIFY] <short warm acknowledgment>. 1) <first diagnostic question, grounded in
   the context> 2) <second diagnostic question, a genuinely different angle - not a
   reworded copy of the first>
   Do not use this rule if the customer's message already gives you enough to answer
   directly (see Rule 1B) - clarifying a question you could already answer is worse
   than just answering it.

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
).replace(
    """8. MEASUREMENT SCOPE: Geometra can measure any physical surface or object — walls,
   wardrobes, washbasins, ceilings, floors, and other physical objects — not just walls.
   This is a confirmed product fact, true regardless of whether a specific object is
   named in the retrieved context. The only requirement is that the Geometra marker is
   properly placed and the photo clearly shows at least 3 visible corners of the
   surface/object being measured. State this confidently and directly when asked whether
   something can be measured — do not decline or hedge just because that specific object
   isn't named in the context, as long as it is a real physical surface or object capable
   of being photographed with 3 visible corners.""",
    """8. MEASUREMENT SCOPE: Geometra can measure any physical surface or object — walls,
   wardrobes, washbasins, ceilings, floors, and other physical objects — not just walls.
   This is a confirmed product fact, true regardless of whether a specific object is
   named in the retrieved context. The only requirement is that the Geometra marker is
   properly placed and the photo clearly shows at least 3 visible corners of the
   surface/object being measured. State this confidently and directly when asked whether
   something can be measured — do not decline or hedge just because that specific object
   isn't named in the context, as long as it is a real physical surface or object capable
   of being photographed with 3 visible corners.

8B. MULTI-SIDED / N-CORNER SURFACES: "At least 3 visible corners" is the specific case
   for a standard 4-sided wall (N=4 corners, N-1=3 must be visible). This generalizes:
   for ANY closed surface with N sides/corners, N-1 of those corners must be visible in
   the photo — a 5-sided room needs 4 visible corners, a 6-sided room needs 5, and so
   on. The surface must be a closed shape. State this confidently when asked about
   rooms or surfaces with more than 4 sides — Geometra is not limited to simple
   rectangular walls, as long as the N-1 visibility rule and the closed-surface
   requirement are met. Do not decline or say "not possible" for a multi-sided room
   just because it has more than 4 corners.""",
)

ANSWER_PROMPT = _ANSWER_BASE

HEDGE_WORDS = ["i think", "i believe", "probably", "i'm not sure", "it seems", "perhaps", "i suppose"]

# Exact-match on the whole reply missed common natural phrasings like "yes please" or
# "yeah sure" (round-2 testing, turn 12). Checking just the first word instead covers those
# without needing a full affirmative-intent classifier. Expanded again after "ya sure do
# it" wasn't recognized (round-5 testing) - "ya" is a very common informal "yes" that
# wasn't in the original list. Deliberately excludes "please" and "fine" as standalone
# triggers - both have plausible non-affirmative first-word uses ("please don't", "fine,
# whatever") that would misfire.
AFFIRMATIVE_WORDS = (
    "yes", "y", "yeah", "yea", "yeh", "ya", "yah", "yep", "yup", "mhm", "mhmm",
    "sure", "ok", "okay", "alright", "aight", "definitely", "absolutely", "certainly",
)


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


# Found via manual testing: a bare backchannel utterance like "mhm" (not a real question,
# not confirming anything) was going all the way through Pass 1/2 and coming back as "I
# don't have enough information to answer that, would you like a ticket?" - a question
# deserves that kind of response, a verbal filler doesn't. This is checked AFTER the
# ticket-confirmation checks above, so "mhm" while a ticket offer is actually pending still
# counts as a "yes" via is_affirmative - this only catches fillers with nothing pending.
FILLER_PHRASES = ("mhm", "mhmm", "hmm", "hm", "mm", "uh huh", "uhhuh", "huh", "meh")


def is_filler(text: str) -> bool:
    stripped = text.strip().lower().strip(".!,")
    return stripped in FILLER_PHRASES


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
        "\nNOTE: the customer was already asked a clarifying question last turn. Rule 2 "
        "([CLARIFY]) is NOT available on this turn - do not produce another [CLARIFY] "
        "response no matter how tempting. Either answer directly (use Rule 1B to combine "
        "whatever the customer's last two messages together now tell you), or if it's "
        "still genuinely not covered, use [CANNOT_ANSWER].\n" if already_clarified else ""
    )
    user_message = (
        f"{prefix}Customer's likely intent: {intent}\n{retry_note}{clarify_cap_note}\n"
        f"CONTEXT:\n{context}\n\nCUSTOMER QUESTION: {original_query}"
    )
    response, _, _ = call_llm(ANSWER_PROMPT, user_message)
    return response


def retrieve_combined(query_text):
    """Same as rag.retriever.retrieve(), but also queries the isolated geometra_website
    collection (see website_kb.py) and merges results in, re-sorted by similarity and
    re-scored for confidence. Kept as a wrapper here rather than editing rag/retriever.py
    directly, so the production retrieve() path used by routers/chat.py is untouched."""
    faq_chunks, _ = retrieve(query_text)
    query_embedding = embed_text(query_text)
    website_results = website_kb.query(query_embedding, top_k=5)
    website_chunks = [
        SourceChunk(
            chunk_id=r["chunk_id"], section=r["section"], text=r["text"],
            similarity_score=r["similarity_score"],
        )
        for r in website_results
    ]
    combined = sorted(faq_chunks + website_chunks, key=lambda c: c.similarity_score, reverse=True)[:15]

    top1 = combined[0].similarity_score if combined else 0.0
    if top1 >= LOW_CONFIDENCE_THRESHOLD:
        confidence = "high"
    elif top1 >= MIN_SIMILARITY_SCORE:
        confidence = "low"
    else:
        confidence = "unknown"
    return combined, confidence


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

    if is_filler(query):
        return "No worries! Let me know whenever you have a question about Geometra.", None

    # Pass 1 — Understand. Always gets recent history (not just when awaiting ==
    # "clarification" as the original diagram showed) - stress-testing found that a
    # short follow-up referencing the previous NORMAL answer (not just a clarifying
    # question) also needs history to resolve correctly, e.g. "can it" right after an
    # answer about washbasins got no context and produced an unrelated guess. Pass 1 is
    # a cheap, short-output call, so always including the last couple of turns costs
    # very little and closes that gap.
    reformulated_query, intent = understand(query, history)

    # Fast-path scope check: ONE retrieve() call, reused for both the gate and Pass 2.
    # retrieve_combined() also pulls in the isolated website knowledge (see website_kb.py).
    chunks, confidence = retrieve_combined(reformulated_query)
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
        # Warmer than a flat "I don't have enough information" - the customer's question
        # was clear, the FAQ just genuinely doesn't cover it, so this should read as "I
        # won't guess and get it wrong for you," not as a dead end.
        return (
            "That's a fair question, and I'd rather not guess and risk giving you the "
            "wrong answer. I don't have that specific detail available to me right now, "
            "but I can raise a support ticket so our team follows up with you directly "
            "with an accurate answer. Would you like me to do that? (yes/no)"
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
