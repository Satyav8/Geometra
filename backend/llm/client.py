from typing import Tuple

from config import (
    LLM_PROVIDER,
    LLM_MODEL,
    OPENAI_API_KEY,
    GEMINI_API_KEY,
    ANTHROPIC_API_KEY,
    GROQ_API_KEY,
)


def call_llm(system_prompt: str, user_message: str) -> Tuple[str, int, int]:
    """Swappable LLM adapter. Returns (response_text, input_tokens, output_tokens)."""
    if LLM_PROVIDER == "groq":
        return _call_groq(system_prompt, user_message)
    if LLM_PROVIDER == "openai":
        return _call_openai(system_prompt, user_message)
    if LLM_PROVIDER == "gemini":
        return _call_gemini(system_prompt, user_message)
    if LLM_PROVIDER == "anthropic":
        return _call_anthropic(system_prompt, user_message)
    raise ValueError(f"Unsupported LLM_PROVIDER: {LLM_PROVIDER}")


# Found via production debugging: the openai SDK's own defaults (600s timeout, 2
# automatic retries) meant a genuine connectivity blip to the provider (an SSL handshake
# timeout, observed directly) could hang a single call for minutes rather than seconds -
# and this function runs 2-3 times per customer turn (Understand, Answer, an occasional
# hedge-retry), so an unbounded hang here is far more customer-visible than the same gap
# in the redundant moderation layer. 20s is generous next to the 1-10s normal successful
# calls actually take (per this project's own logged latencies), while still bounding a
# real outage to a a few tens of seconds instead of minutes. One retry, not zero - unlike
# the moderation check, there's no other layer that produces the customer's actual answer
# if this fails, so a single retry for a transient blip is worth the modest extra latency
# it costs on the rare case it's needed.
_LLM_TIMEOUT_SECONDS = 20.0
_LLM_MAX_RETRIES = 1


def _call_groq(system_prompt: str, user_message: str) -> Tuple[str, int, int]:
    from openai import OpenAI

    client = OpenAI(
        api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1",
        timeout=_LLM_TIMEOUT_SECONDS, max_retries=_LLM_MAX_RETRIES,
    )
    completion = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        temperature=0.5,
        top_p=0.9,
    )
    text = completion.choices[0].message.content
    usage = completion.usage
    return text, usage.prompt_tokens, usage.completion_tokens


def _call_openai(system_prompt: str, user_message: str) -> Tuple[str, int, int]:
    from openai import OpenAI

    client = OpenAI(api_key=OPENAI_API_KEY, timeout=_LLM_TIMEOUT_SECONDS, max_retries=_LLM_MAX_RETRIES)
    completion = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        temperature=0.5,
        top_p=0.9,
    )
    text = completion.choices[0].message.content
    usage = completion.usage
    cached = getattr(getattr(usage, "prompt_tokens_details", None), "cached_tokens", 0) or 0
    if cached:
        print(f"[cache] {cached}/{usage.prompt_tokens} prompt tokens served from cache")
    return text, usage.prompt_tokens, usage.completion_tokens


def _call_gemini(system_prompt: str, user_message: str) -> Tuple[str, int, int]:
    import google.generativeai as genai

    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel(model_name=LLM_MODEL, system_instruction=system_prompt)
    response = model.generate_content(user_message)
    text = response.text
    input_tokens = response.usage_metadata.prompt_token_count
    output_tokens = response.usage_metadata.candidates_token_count
    return text, input_tokens, output_tokens


def _call_anthropic(system_prompt: str, user_message: str) -> Tuple[str, int, int]:
    import anthropic

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    message = client.messages.create(
        model=LLM_MODEL,
        max_tokens=10000,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}],
    )
    text = message.content[0].text
    return text, message.usage.input_tokens, message.usage.output_tokens
