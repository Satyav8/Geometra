"""Shared slowapi Limiter instance - kept in its own module (not main.py or routers/chat.py
directly) so both can import the same instance without a circular import: main.py needs it
to register the exception handler/middleware, routers/chat.py needs it to decorate the
/chat route.

Keyed by client IP (get_remote_address) since the concern here is cost/abuse control on a
public endpoint, not per-customer fairness. In-memory storage (slowapi's default) - correct
for the current single-instance Render deployment; if this ever scales to multiple
instances, each instance keeps its own counters and the effective limit multiplies per
instance, since there's no shared store (e.g. Redis) behind it.
"""
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
