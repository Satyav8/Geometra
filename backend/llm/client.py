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


def _call_groq(system_prompt: str, user_message: str) -> Tuple[str, int, int]:
    from openai import OpenAI

    client = OpenAI(api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")
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

    client = OpenAI(api_key=OPENAI_API_KEY)
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
