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
* **One build per key at a time.** Concurrent requests for the same key wait
  for the one build in flight (a request that arrives while the start-up warm
  is building an extract's summary takes that result) rather than repeating a
  multi-second preparation beside it; different keys build in parallel.
* **A persistent tier for small results, per build.** A cache constructed with
  ``persist=True`` also keeps each value as JSON on persistent storage, under
  the deployed commit, so a restart or a second worker reads it back instead of
  re-preparing it. Only values that survive a JSON round trip unchanged are
  written; a different build never reads another build's entries.

Kill switches (acceptance criterion: every optimisation is revertible without
touching the canonical pipeline)::

    TRAKT_SERVING_CACHE=off                  disable all serving caches
    TRAKT_SERVING_CACHE_<NAME>=off           disable one (e.g. PIPELINE_PREP)
    TRAKT_SERVING_CACHE_PERSIST=off          keep every cache in memory only
    TRAKT_SERVING_CACHE_DIR=<path>           where the persistent tier lives
                                             (default on App Service:
                                             /home/data/trakt/serving_cache)
"""

from __future__ import annotations

import hashlib
import json
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
METHODOLOGY_VERSION = "2"  # 2: stage run-off pipeline forecast

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


#: How long a request waits for another thread's build of the same key before
#: building it itself (a build that hangs must not hang every waiter with it).
SINGLE_FLIGHT_WAIT_S = 240.0


class BoundedCache:
    """A bounded LRU memo with hit/miss instrumentation.

    Thread-safety: the mapping is guarded by a lock, but the BUILDER runs
    outside it, so a multi-second preparation never blocks requests for other
    keys. Requests for the SAME key are single-flight: the first builds, the
    rest wait for its result (up to ``SINGLE_FLIGHT_WAIT_S``) instead of
    repeating the build beside it. If the build fails, each waiter builds for
    itself, so an error path is never shared or hidden.

    ``persist=True`` adds the persistent tier (see the module docstring): a
    memory miss reads the value back from disk before building.
    """

    def __init__(self, name: str, max_entries: int, *, persist: bool = False) -> None:
        self.name = name
        self.max_entries = max_entries
        self.persist = persist
        self._data: "OrderedDict[str, Any]" = OrderedDict()
        self._lock = threading.RLock()
        #: key -> (done event, building thread id), for single-flight builds.
        self._building: Dict[str, Tuple[threading.Event, int]] = {}
        self.hits = 0
        self.misses = 0
        self.disk_hits = 0
        _REGISTRY.append(self)

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()
            self.hits = 0
            self.misses = 0
            self.disk_hits = 0

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

        # SINGLE-FLIGHT: one build per key. A thread that finds the key being
        # built elsewhere waits for that build; the building thread itself
        # (a builder that asks for its own key) never waits on itself.
        me = threading.get_ident()
        with self._lock:
            value = self._data.get(key, _MISSING)
            if value is not _MISSING:     # built by another thread meanwhile
                self.hits += 1
                perf.cache_event(self.name, perf.CACHE_HIT)
                return value
            in_flight = self._building.get(key)
            if in_flight is None:
                done = threading.Event()
                self._building[key] = (done, me)
        if in_flight is not None and in_flight[1] != me:
            in_flight[0].wait(SINGLE_FLIGHT_WAIT_S)
            with self._lock:
                value = self._data.get(key, _MISSING)
            if value is not _MISSING:
                self.hits += 1
                perf.cache_event(self.name, perf.CACHE_HIT)
                return value
            # The other build failed or is still running: build here, as the
            # request would have without the cache.
            perf.cache_event(self.name, perf.CACHE_BYPASS)
            return build()
        if in_flight is not None:
            return build()          # re-entrant: the builder asked for its own key

        try:
            value = self._disk_read(key)
            if value is not _MISSING:
                self.disk_hits += 1
                perf.cache_event(self.name, perf.CACHE_HIT)
            else:
                self.misses += 1
                perf.cache_event(self.name, perf.CACHE_MISS)
                # Built OUTSIDE the lock, and stored only on success — a builder
                # that raises leaves the cache untouched and the error path
                # unchanged.
                value = build()
                self._disk_write(key, value)
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
        finally:
            with self._lock:
                self._building.pop(key, None)
            done.set()

    # -- the persistent tier ------------------------------------------------ #
    def _disk_path(self, key: str) -> Optional[Path]:
        if not self.persist:
            return None
        base = persist_dir()
        if base is None:
            return None
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return base / self.name / f"{digest}.json"

    def _disk_read(self, key: str) -> Any:
        try:
            path = self._disk_path(key)
            if path is None or not path.exists():
                return _MISSING
            stored = json.loads(path.read_text(encoding="utf-8"))
            if stored.get("key") != key:
                return _MISSING
            return stored["value"]
        except Exception:  # noqa: BLE001 - an unreadable entry is a miss
            return _MISSING

    def _disk_write(self, key: str, value: Any) -> None:
        try:
            path = self._disk_path(key)
            if path is None:
                return
            text = json.dumps({"key": key, "value": value}, sort_keys=True)
            # Only a value that reads back EQUAL is written: a tuple, a numpy
            # integer or any type JSON would change is kept in memory only.
            if json.loads(text)["value"] != value:
                return
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}")
            tmp.write_text(text, encoding="utf-8")
            os.replace(tmp, path)
        except Exception:  # noqa: BLE001 - failing to persist is not failing
            logger.info("serving cache %s could not persist an entry", self.name)


def clear_all() -> None:
    """Reset every serving cache (tests, and the existing manual refresh)."""
    for cache in _REGISTRY:
        cache.clear()


def stats() -> Dict[str, Dict[str, int]]:
    return {c.name: {"entries": len(c), "hits": c.hits, "misses": c.misses,
                     "disk_hits": c.disk_hits, "max_entries": c.max_entries}
            for c in _REGISTRY}


#: App Service's persistent storage (survives restarts; shared by workers).
_APP_SERVICE_PERSIST_DIR = "/home/data/trakt/serving_cache"


def persist_dir() -> Optional[Path]:
    """The persistent tier's directory for THIS build, or None (memory only).

    Under the deployed commit, so entries built by one build are never read by
    another: a code change that alters a result needs no version bump to be
    safe. Off when caching is off, when ``TRAKT_SERVING_CACHE_PERSIST=off``,
    when the build has no commit stamp (a local run), and outside App Service
    unless ``TRAKT_SERVING_CACHE_DIR`` names a directory.
    """
    if not caches_enabled():
        return None
    if str(os.environ.get("TRAKT_SERVING_CACHE_PERSIST", "on")).strip().lower() in _OFF:
        return None
    configured = (os.environ.get("TRAKT_SERVING_CACHE_DIR") or "").strip()
    if configured:
        base = Path(configured)
    elif os.environ.get("WEBSITE_SITE_NAME"):
        base = Path(_APP_SERVICE_PERSIST_DIR)
    else:
        return None
    try:
        from .build_info import build_info
        commit = build_info().get("commit")
    except Exception:  # noqa: BLE001 - no stamp, no persistent tier
        commit = None
    if not commit:
        return None
    return base / str(commit)[:40]


def prune_other_builds() -> int:
    """Remove other builds' persistent entries; returns how many builds went."""
    current = persist_dir()
    if current is None or not current.parent.exists():
        return 0
    import shutil
    removed = 0
    for sibling in current.parent.iterdir():
        if sibling.is_dir() and sibling != current:
            shutil.rmtree(sibling, ignore_errors=True)
            removed += 1
    return removed


# --------------------------------------------------------------------------- #
# Identity
# --------------------------------------------------------------------------- #
#: Written beside a file mirrored from object storage: the ETag of the bytes the
#: store published (see ``datasets._materialise_pipeline_root_uncached``).
ETAG_SIDECAR_SUFFIX = ".etag"


def file_identity(path: str | os.PathLike) -> Optional[str]:
    """The identity of a local file's bytes, or ``None`` when it cannot be stat'd.

    A file mirrored from object storage is identified by the ETag the store
    published for it (its sidecar) and its size: the same bytes keep the same
    identity when a second worker or a new container downloads them again, and
    a republished file changes it. Any other file is ``mtime_ns:size``, the
    change-token the filesystem storage backend reports as an ETag.
    """
    p = Path(path)
    try:
        st = p.stat()
    except OSError:
        return None
    try:
        etag = p.with_name(p.name + ETAG_SIDECAR_SUFFIX).read_text(
            encoding="utf-8").strip()
    except OSError:
        etag = ""
    if etag:
        return f"etag:{etag}:{st.st_size}"
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
