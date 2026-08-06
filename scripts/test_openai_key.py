"""Minimal standalone check: does OPENAI_API_KEY work at all?
Sends exactly one question to GPT-4o mini. No RAG, no embeddings, no project code.
Run: python scripts/test_openai_key.py
"""
import os
import sys

from dotenv import load_dotenv

# override=True: .env must win over any stray OPENAI_API_KEY already set at the
# OS/shell level (dotenv otherwise silently keeps the shell's value instead).
load_dotenv(override=True)

API_KEY = os.getenv("OPENAI_API_KEY", "")
QUESTION = "In one short sentence, what is the capital of France?"


def main():
    if not API_KEY:
        print("FAILED: OPENAI_API_KEY is empty or not found in .env")
        sys.exit(1)

    from openai import OpenAI

    client = OpenAI(api_key=API_KEY)

    print(f"Sending question: {QUESTION!r}")
    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": QUESTION}],
        )
    except Exception as e:
        print("FAILED: key/account is not working.")
        print(f"Error type: {type(e).__name__}")
        print(f"Error detail: {e}")
        sys.exit(1)

    answer = completion.choices[0].message.content
    print("SUCCESS: key is working.")
    print(f"Answer: {answer}")
    print(f"Tokens used: {completion.usage.prompt_tokens} in / {completion.usage.completion_tokens} out")


if __name__ == "__main__":
    main()
