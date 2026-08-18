"""Interactive local test of the clarifying-question idea. NOT part of the real app -
throwaway test tool only, safe to delete anytime, nothing here is wired into
routers/chat.py or committed behavior. Run: ./venv/Scripts/python.exe try_it_yourself.py
Type 'exit' to quit.
"""
import sys

sys.path.insert(0, ".")
from rag.retriever import retrieve
from rag.relevance import is_gratitude, is_greeting
from llm.client import call_llm
from llm.prompts import SYSTEM_PROMPT
from config import GRATITUDE_MESSAGE, GREETING_MESSAGE

# Edit this block freely to test different "what Geometra can do" facts.
BULLETIN_CONTEXT = """
BACKGROUND CONTEXT ABOUT GEOMETRA (always true, treat as known fact even if not
explicitly repeated in the retrieved chunks below):
Geometra can measure ONLY the following: wall surfaces, doors, and windows. It cannot
measure anything else - not floors, ceilings, furniture, washbasins, staircases, or any
other object or surface. This is a strict, exhaustive list.
"""

CLARIFY_RULE = """

9. ASK FOR CLARIFICATION WHEN THE QUESTION ITSELF IS TOO VAGUE: If you genuinely cannot
   tell what the customer is asking about - not because the FAQ lacks the answer, but
   because their message doesn't say enough to know what they mean - do not guess, and
   do not use the Rule 2 fallback. Instead ask ONE short, specific clarifying question
   that would let you answer correctly. You may do this at most twice in a row for the
   same underlying request. If, after two clarifying questions, you still cannot tell
   what they're asking, say so plainly and suggest contacting support.
"""

TEST_SYSTEM_PROMPT = SYSTEM_PROMPT.replace(
    "LOW CONFIDENCE MODE:", BULLETIN_CONTEXT + CLARIFY_RULE + "\nLOW CONFIDENCE MODE:"
)


def build_message(query, chunks, confidence_level, history):
    prefix = "[LOW CONFIDENCE]\n" if confidence_level == "low" else ""
    context = "\n\n".join(f"[Source: {c.section}]\n{c.text}" for c in chunks)
    history_block = ""
    if history:
        history_block = "CONVERSATION SO FAR:\n" + "\n".join(
            f"{'Customer' if role == 'customer' else 'S.A.M'}: {text}" for role, text in history
        ) + "\n\n"
    return f"{prefix}{history_block}CONTEXT:\n{context}\n\nCUSTOMER QUESTION: {query}"


def ask(query, history):
    if is_gratitude(query):
        return GRATITUDE_MESSAGE, "n/a"
    if is_greeting(query):
        return GREETING_MESSAGE, "n/a"
    chunks, confidence = retrieve(query)
    user_message = build_message(query, chunks, confidence, history)
    response, _, _ = call_llm(TEST_SYSTEM_PROMPT, user_message)
    return response, confidence


def main():
    print("S.A.M test mode — type a question, 'exit' to quit.\n")
    history = []
    while True:
        query = input("You: ").strip()
        if not query:
            continue
        if query.lower() in ("exit", "quit"):
            break
        response, confidence = ask(query, history)
        print(f"S.A.M ({confidence}): {response}\n")
        history.append(("customer", query))
        history.append(("sam", response))


if __name__ == "__main__":
    main()
