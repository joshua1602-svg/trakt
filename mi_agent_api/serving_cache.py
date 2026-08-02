"""mi_agent_api/serving_cache.py — bounded, identity-keyed serving caches.

Phase 1A reuses the *results* of expensive preparation inside a process. It does
NOT introduce a materialised read model, a shared cache service, or any new
source of truth: every entry is a memo of a calculation the request-time path
still performs on a miss.

Design rules, all enforced here rather than at each call site:

* **Immutable source identity, never time.** A key is built from the identity of
  the bytes that produced the value — URI plus ETag (blob) or ``mtime_ns:size``
  (filesystem) — plus the tenant, the portfolio scope and a methodology version.
  A re-published or corrected extract therefore produces a NEW key and can never
  be served from an old entry. There is deliberately no TTL: a TTL would be both
  unsafe (stale within the window) and pointless (a changed file already misses).
* **Tenant and scope are always in the key.** ``key_for`` refuses to build a key
  without them, so a cross-tenant or cross-portfolio hit is not something a call
  site can forget to prevent.
* **Bounded.** Every cache is an LRU with an explicit entry ceiling.
* **Never publishes a partial object.** The value is inserted only after the
  builder returns. A builder that raises stores nothing and the exception
  propagates to the existing error handling.
* **Degrades to the calculation.** Any fault inside the cache itself falls
  through to the builder — correctness never depends on the cache working.
* **Per-process, and correct without sharing.** With two or more gunicorn
  workers each holds its own copy; that costs a duplicated first build per
  worker and nothing else. No entry is authoritative for anything.

Kill switches (acceptance criterion: every optimisation is revertible without
touching the canonical pipeline)::

    TRAKT_SERVING_CACHE=off                  disable all serving caches
    TRAKT_SERVING_CACHE_<NAME>=off           disable one (e.g. PIPELINE_PREP)
"""

from __future__ import annotations

import hashlib
import logging
import os
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from trakt_core import perf

logger = logging.getLogger("mi_agent_api.serving_cache")

_OFF = {"0", "off", "false", "no"}

#: Bump when a change alters what a cached builder PRODUCES. Every key embeds
#: it, so an upgraded deployment cannot serve values built by the old code.
METHODOLOGY_VERSION = "1"

#: Distinguishes "cached a legitimate None" from "absent".
_MISSING = object()

#: All live caches, so tests (and a future admin route) can reset every one.
_REGISTRY: "List[BoundedCache]" = []


def caches_enabled() -> bool:
    return str(os.environ.get("TRAKT_SERVING_CACHE", "on")).strip().lower() not in _OFF


def _cache_enabled(name: str) -> bool:
    if not caches_enabled():
        return False
    var = f"TRAKT_SERVING_CACHE_{name.upper()}"
    return str(os.environ.get(var, "on")).strip().lower() not in _OFF


class BoundedCache:
    """A bounded LRU memo with hit/miss instrumentation.

    Thread-safety: the mapping is guarded by a lock, but the BUILDER runs
    outside it. Two concurrent cold requests may therefore both build — which
    wastes one computation and cannot corrupt anything — rather than one
    blocking the other for the length of a multi-second preparation.
    """

    def __init__(self, name: str, max_entries: int) -> None:
        self.name = name
        self.max_entries = max_entries
        self._data: "OrderedDict[str, Any]" = OrderedDict()
        self._lock = threading.RLock()
        self.hits = 0
        self.misses = 0
        _REGISTRY.append(self)

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()
            self.hits = 0
            self.misses = 0

    def peek(self, key: str) -> Any:
        with self._lock:
            return self._data.get(key, _MISSING)

    def get_or_build(self, key: Optional[str], build: Callable[[], Any]) -> Any:
        """Return the cached value for ``key``, building it on a miss.

        ``key is None`` means the caller could not establish an immutable
        identity (an unreadable source, a missing etag). That is a BYPASS, not a
        miss: rather than key on something mutable, the value is computed and
        not stored.
        """
        if key is None or not _cache_enabled(self.name):
            perf.cache_event(self.name, perf.CACHE_BYPASS)
            return build()
        try:
            with self._lock:
                value = self._data.get(key, _MISSING)
                if value is not _MISSING:
                    self._data.move_to_end(key)
                    self.hits += 1
                    perf.cache_event(self.name, perf.CACHE_HIT)
                    return value
        except Exception:  # noqa: BLE001 - a cache fault must never fail a read
            logger.warning("serving cache %s lookup failed; computing", self.name)
            perf.cache_event(self.name, perf.CACHE_BYPASS)
            return build()

        self.misses += 1
        perf.cache_event(self.name, perf.CACHE_MISS)
        # Built OUTSIDE the lock, and stored only on success — a builder that
        # raises leaves the cache untouched and the error path unchanged.
        value = build()
        try:
            with self._lock:
                self._data[key] = value
                self._data.move_to_end(key)
                while len(self._data) > self.max_entries:
                    self._data.popitem(last=False)
            perf.cache_event(self.name, perf.CACHE_STORE)
        except Exception:  # noqa: BLE001 - failing to STORE is not failing
            logger.warning("serving cache %s store failed", self.name)
        return value


def clear_all() -> None:
    """Reset every serving cache (tests, and the existing manual refresh)."""
    for cache in _REGISTRY:
        cache.clear()


def stats() -> Dict[str, Dict[str, int]]:
    return {c.name: {"entries": len(c), "hits": c.hits, "misses": c.misses,
                     "max_entries": c.max_entries}
            for c in _REGISTRY}


# --------------------------------------------------------------------------- #
# Identity
# --------------------------------------------------------------------------- #
def file_identity(path: str | os.PathLike) -> Optional[str]:
    """``mtime_ns:size`` for a local file, or ``None`` when it cannot be stat'd.

    This is the same change-token the filesystem storage backend reports as an
    ETag, so a mirrored blob whose ETag changed (and was therefore re-downloaded)
    naturally produces a new identity.
    """
    try:
        st = Path(path).stat()
    except OSError:
        return None
    return f"{st.st_mtime_ns}:{st.st_size}"


def fingerprint(*parts: Any) -> str:
    """A short, stable digest of the given parts.

    Used to compress a long identity (an ordered extract set, a stage-rate
    table) into a key component without losing its discriminating power.
    """
    hasher = hashlib.sha256()
    for part in parts:
        hasher.update(repr(part).encode("utf-8", "replace"))
        hasher.update(b"\x1f")
    return hasher.hexdigest()[:32]


def key_for(*, tenant: Optional[str], scope: Optional[str],
            identity: Iterable[Any], version: str = METHODOLOGY_VERSION
            ) -> Optional[str]:
    """Build a cache key, or ``None`` when identity cannot be established.

    ``tenant`` and ``scope`` are mandatory components — an entry can only ever
    be reused for the same tenant and the same governed portfolio scope.
    ``identity`` must contain the immutable source markers (URI + ETag/mtime per
    contributing file). Any ``None`` inside ``identity`` collapses the whole key
    to ``None``: an unidentifiable source is never cached under a guessed key.
    """
    parts = list(identity)
    if not parts or any(p is None for p in parts):
        return None
    return "|".join((
        f"v{version}",
        f"t={tenant or '-'}",
        f"s={scope or '-'}",
        fingerprint(*parts),
    ))


def model_fingerprint(model: Optional[Dict[str, Any]]) -> str:
    """Identity of a historical completion model, for keys that depend on it.

    The prepared pipeline dataset is weighted by this model's stage rates, so a
    different model must produce a different prepared frame — and therefore a
    different key. Only the fields that affect the OUTPUT are included.
    """
    if not model:
        return "none"
    return fingerprint(
        model.get("historicalCompletionRateByStage"),
        model.get("historicalCompletionTimingByStage"),
        model.get("stage_rates"),
        model.get("completion_probability_basis"),
        model.get("uniqueWeeklyExtractsUsed"),
        model.get("minObservations"),
    )


def resolved_tenant() -> Optional[str]:
    """The deployment's trusted tenant id, for key isolation. Never request data."""
    try:
        from .dependencies import default_tenant_id
        return default_tenant_id()
    except Exception:  # noqa: BLE001 - an unresolvable tenant disables caching
        return None


def scope_token(scope: Any) -> str:
    """A stable token for a governed portfolio scope.

    ``None``/Total is ``total``; anything else contributes its context id AND its
    resolved portfolio id list, so two different scopes that happen to share a
    label can never collide.
    """
    if scope is None:
        return "total"
    if isinstance(scope, str):
        return scope
    context_id = getattr(scope, "context_id", None)
    ids = tuple(getattr(scope, "portfolio_ids", ()) or ())
    filters = getattr(scope, "filters", None)
    if context_id is None and not ids and not filters:
        return "total"
    return fingerprint(context_id, ids, filters)
