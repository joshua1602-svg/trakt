"""trakt_core/perf.py — request-scoped performance instrumentation.

A dependency-free collector for "where did the time go in this request?". It is
deliberately in ``trakt_core`` (not ``mi_agent_api``) so the storage layer in
``apps.blob_trigger_app`` can report blob operation counts without importing the
web tier — the dependency direction the governed-capability architecture already
establishes.

Three properties hold by construction:

* **Inactive by default.** Nothing collects until a caller opens
  :func:`collect`. The MI API middleware opens one per HTTP request; ingestion
  never does, so ingestion behaviour is byte-for-byte unchanged and pays only a
  ``ContextVar.get()`` per instrumented call.
* **Fail-open.** Every public function swallows its own exceptions. An
  instrumentation fault must never turn a working request into a failed one.
* **No payload capture.** Stages and counters record names, durations and
  counts. Loan-level data, file contents and principal identifiers are never
  recorded here; the caller chooses which coarse, non-sensitive fields to stamp
  on the collector (tenant, route, run id).

Usage::

    with perf.collect(route="/mi/snapshot", tenant_id="ERE") as c:
        with perf.stage("resolve_run"):
            df = resolve(...)
        perf.count("blob.etag", ms=12.3)
        c.snapshot()   # -> structured dict for one log line
"""

from __future__ import annotations

import contextvars
import os
import time
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Optional

__all__ = [
    "Collector", "collect", "active", "stage", "stage_fn", "count", "counted",
    "cache_event", "stamp", "enabled",
    "CACHE_HIT", "CACHE_MISS", "CACHE_STORE", "CACHE_BYPASS",
]

#: Cache outcome vocabulary. Closed on purpose — an unknown outcome is a caller
#: bug, matching the discipline used elsewhere in the repository.
CACHE_HIT = "hit"
CACHE_MISS = "miss"
CACHE_STORE = "store"
CACHE_BYPASS = "bypass"

#: Master switch. Instrumentation is on by default (it is cheap and it is the
#: only way the serving layer can be tuned), and can be turned off entirely with
#: ``TRAKT_PERF_INSTRUMENTATION=off`` without touching code.
_DISABLED_VALUES = {"0", "off", "false", "no"}


def enabled() -> bool:
    """Is instrumentation permitted in this process?"""
    return str(os.environ.get("TRAKT_PERF_INSTRUMENTATION", "on")).strip().lower() \
        not in _DISABLED_VALUES


class Collector:
    """Accumulates stage timings, counters and cache outcomes for one request.

    Not thread-safe by design: one collector belongs to one request, and a
    request is served by one task. Concurrent mutation would only skew a metric,
    never corrupt a response — nothing reads these values to make a decision.
    """

    __slots__ = ("fields", "stages", "counters", "caches", "started_at", "_t0")

    def __init__(self, **fields: Any) -> None:
        self.fields: Dict[str, Any] = {k: v for k, v in fields.items() if v is not None}
        #: stage name -> {"ms": float, "n": int}
        self.stages: Dict[str, Dict[str, float]] = {}
        #: counter name -> {"n": int, "ms": float}
        self.counters: Dict[str, Dict[str, float]] = {}
        #: cache name -> {outcome: count}
        self.caches: Dict[str, Dict[str, int]] = {}
        self.started_at = time.time()
        self._t0 = time.perf_counter()

    # -- recording ---------------------------------------------------------- #
    def add_stage(self, name: str, ms: float) -> None:
        slot = self.stages.get(name)
        if slot is None:
            self.stages[name] = {"ms": round(ms, 2), "n": 1}
        else:
            slot["ms"] = round(slot["ms"] + ms, 2)
            slot["n"] += 1

    def add_count(self, name: str, n: int = 1, ms: Optional[float] = None) -> None:
        slot = self.counters.setdefault(name, {"n": 0, "ms": 0.0})
        slot["n"] += n
        if ms:
            slot["ms"] = round(slot["ms"] + ms, 2)

    def add_cache(self, cache: str, outcome: str) -> None:
        self.caches.setdefault(cache, {})[outcome] = \
            self.caches.setdefault(cache, {}).get(outcome, 0) + 1

    def stamp(self, **fields: Any) -> None:
        """Attach (or overwrite) coarse context fields discovered mid-request."""
        for k, v in fields.items():
            if v is not None:
                self.fields[k] = v

    # -- reading ------------------------------------------------------------ #
    def elapsed_ms(self) -> float:
        return round((time.perf_counter() - self._t0) * 1000.0, 2)

    def cache_totals(self) -> Dict[str, int]:
        """Flat ``{"<cache>.<outcome>": n}`` view, convenient for assertions."""
        out: Dict[str, int] = {}
        for cache, outcomes in self.caches.items():
            for outcome, n in outcomes.items():
                out[f"{cache}.{outcome}"] = n
        return out

    def snapshot(self) -> Dict[str, Any]:
        """One structured record for a single log line."""
        return {
            "event": "perf.request",
            **self.fields,
            "total_ms": self.elapsed_ms(),
            "stages": {k: v["ms"] for k, v in sorted(self.stages.items())},
            "stage_calls": {k: int(v["n"]) for k, v in sorted(self.stages.items())},
            "counters": {k: int(v["n"]) for k, v in sorted(self.counters.items())},
            "counter_ms": {k: v["ms"] for k, v in sorted(self.counters.items())
                           if v["ms"]},
            "caches": {k: dict(v) for k, v in sorted(self.caches.items())},
        }

    def server_timing(self, limit: int = 12) -> str:
        """A ``Server-Timing`` header value for the heaviest stages.

        Stage names are internal identifiers (``resolve_run``, ``funded_prep``),
        never data — safe to expose to the browser's performance panel.
        """
        top = sorted(self.stages.items(), key=lambda kv: -kv[1]["ms"])[:limit]
        parts = [f"{_token(k)};dur={v['ms']:.1f}" for k, v in top]
        parts.append(f"total;dur={self.elapsed_ms():.1f}")
        return ", ".join(parts)


def _token(name: str) -> str:
    """Sanitise a stage name into a Server-Timing token."""
    return "".join(ch if (ch.isalnum() or ch in "_-") else "_" for ch in name)[:40]


_CURRENT: contextvars.ContextVar[Optional[Collector]] = contextvars.ContextVar(
    "trakt_perf_collector", default=None)


def active() -> Optional[Collector]:
    """The collector for the current request, or ``None`` when inactive."""
    try:
        return _CURRENT.get()
    except Exception:  # noqa: BLE001 - instrumentation must never raise
        return None


@contextmanager
def collect(**fields: Any) -> Iterator[Optional[Collector]]:
    """Activate a collector for the duration of the block.

    Yields ``None`` when instrumentation is disabled, so callers can branch on
    it without a second env lookup.
    """
    if not enabled():
        yield None
        return
    collector = Collector(**fields)
    token = _CURRENT.set(collector)
    try:
        yield collector
    finally:
        try:
            _CURRENT.reset(token)
        except Exception:  # noqa: BLE001 - a reset across tasks must not raise
            _CURRENT.set(None)


@contextmanager
def stage(name: str) -> Iterator[None]:
    """Time a named stage into the active collector (no-op when inactive)."""
    collector = active()
    if collector is None:
        yield
        return
    t0 = time.perf_counter()
    try:
        yield
    finally:
        try:
            collector.add_stage(name, (time.perf_counter() - t0) * 1000.0)
        except Exception:  # noqa: BLE001
            pass


def stage_fn(name: str):
    """Decorator form of :func:`stage`, for timing a whole function.

    Preserves the wrapped function's signature and return value exactly; when no
    collector is active the wrapper is a single ``ContextVar.get()`` and a call.
    """
    def decorate(fn):
        import functools

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            collector = active()
            if collector is None:
                return fn(*args, **kwargs)
            t0 = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                try:
                    collector.add_stage(name, (time.perf_counter() - t0) * 1000.0)
                except Exception:  # noqa: BLE001
                    pass
        return wrapper
    return decorate


def count(name: str, n: int = 1, ms: Optional[float] = None) -> None:
    """Increment a named counter (no-op when inactive)."""
    collector = active()
    if collector is None:
        return
    try:
        collector.add_count(name, n=n, ms=ms)
    except Exception:  # noqa: BLE001
        pass


@contextmanager
def counted(name: str) -> Iterator[None]:
    """Count one occurrence of ``name`` and attribute its duration to it."""
    collector = active()
    if collector is None:
        yield
        return
    t0 = time.perf_counter()
    try:
        yield
    finally:
        try:
            collector.add_count(name, 1, (time.perf_counter() - t0) * 1000.0)
        except Exception:  # noqa: BLE001
            pass


def cache_event(cache: str, outcome: str) -> None:
    """Record a cache hit/miss/store/bypass (no-op when inactive)."""
    collector = active()
    if collector is None:
        return
    try:
        collector.add_cache(cache, outcome)
    except Exception:  # noqa: BLE001
        pass


def stamp(**fields: Any) -> None:
    """Attach coarse context fields to the active collector (no-op when inactive).

    Only non-sensitive identifiers belong here: tenant, portfolio context, run
    id, route. Never loan-level values.
    """
    collector = active()
    if collector is None:
        return
    try:
        collector.stamp(**fields)
    except Exception:  # noqa: BLE001
        pass
