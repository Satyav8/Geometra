"""OpenAI Moderation API check - a free, ~20ms classifier call that sits alongside the
wordlist-based is_severe_slur()/is_injection_attempt() checks in two_pass.py, catching
sexual/hate/violence/self-harm/harassment/illicit content the English-only wordlist was
never going to cover (broader semantic matching, 50+ languages per OpenAI). Kept as its
own module, not llm/client.py, since it's a fixed classifier endpoint, not a swappable
chat-completion call - it always calls OpenAI directly, regardless of LLM_PROVIDER.

Not a complete solution on its own - independent benchmarking (RealHarm, arXiv:2504.10277)
found commercial moderation APIs including this one catch as little as 10-50% of unsafe
content in adversarial test sets, and it has no dedicated "extremism" category (folded
into hate/illicit). It's one layer of three - the wordlist, this, and Pass 2's own SAFETY
rule - not a replacement for any of them.
"""
from config import OPENAI_API_KEY

MODERATION_MODEL = "omni-moderation-latest"


# The openai SDK's own defaults are a 600s timeout and up to 2 automatic retries with
# exponential backoff on transient errors (429/5xx) - found via production testing to
# turn a single rate-limited or slow moderation call into a 90+ second hang on the
# customer's entire chat response, since the "fail open" except block below only helps
# once the call actually raises. A short, explicit timeout with no retries makes this
# check fail FAST when OpenAI is slow, not just fail SAFE eventually - the wordlist check
# and Pass 2's own SAFETY rule are unaffected either way, so there's nothing gained by
# waiting longer for this one layer.
_TIMEOUT_SECONDS = 4.0
_client = None


def _get_client():
    global _client
    if _client is None:
        from openai import OpenAI

        _client = OpenAI(api_key=OPENAI_API_KEY, timeout=_TIMEOUT_SECONDS, max_retries=0)
    return _client


def is_flagged_by_moderation(text: str) -> bool:
    """Fails open: a missing key, network error, timeout, or API error skips this layer
    rather than blocking the customer's message - the same "never let an optional layer
    break the customer-facing response" convention as send_ticket_email/
    write_escalated_question elsewhere in this codebase. The wordlist check and Pass 2's
    own SAFETY rule still apply even if this call fails, so a moderation outage degrades
    to two layers, not zero.
    """
    if not OPENAI_API_KEY:
        return False
    try:
        response = _get_client().moderations.create(model=MODERATION_MODEL, input=text)
        return bool(response.results[0].flagged)
    except Exception as e:
        print(f"[moderation] check failed, continuing without it: {e}")
        return False
