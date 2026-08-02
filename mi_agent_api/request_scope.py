"""mi_agent_api/request_scope.py — per-request memo for governed resolution.

Some governed resolution steps are *revalidations*: they ask storage "has this
changed?" before reusing a local mirror. Doing that once per request is correct;
doing it once per call site is waste, because several helpers on one request
each re-ask about the same prefix.

This module provides a request-scoped dictionary so a resolution result can be
reused **within a single request** and revalidated on the next one. That is
deliberately weaker than a cross-request cache: freshness is unchanged at the
request boundary, which is the granularity the dashboard actually observes.

Inactive outside a request (the ingestion path, a CLI, a test that does not open
a scope), in which case every helper behaves exactly as it did before.
"""

from __future__ import annotations

import contextvars
from contextlib import contextmanager
from typing import Any, Callable, Dict, Iterator, Optional, TypeVar

T = TypeVar("T")

_CURRENT: contextvars.ContextVar[Optional[Dict[str, Any]]] = contextvars.ContextVar(
    "mi_agent_request_scope", default=None)

_MISSING = object()


@contextmanager
def scope() -> Iterator[Dict[str, Any]]:
    """Open a request scope for the duration of the block."""
    store: Dict[str, Any] = {}
    token = _CURRENT.set(store)
    try:
        yield store
    finally:
        try:
            _CURRENT.reset(token)
        except Exception:  # noqa: BLE001 - a reset across tasks must not raise
            _CURRENT.set(None)


def active() -> Optional[Dict[str, Any]]:
    try:
        return _CURRENT.get()
    except Exception:  # noqa: BLE001
        return None


def memo(key: str, build: Callable[[], T]) -> T:
    """Return ``build()``, reusing the value for the rest of this request.

    With no active scope this is a plain call, so behaviour outside an HTTP
    request is unchanged. A builder that raises stores nothing.
    """
    store = active()
    if store is None:
        return build()
    cached = store.get(key, _MISSING)
    if cached is not _MISSING:
        return cached  # type: ignore[return-value]
    value = build()
    store[key] = value
    return value
