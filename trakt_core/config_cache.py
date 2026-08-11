"""trakt_core.config_cache — parse each governance registry once per content version.

The five governance registries — organisations, resources, entitlements,
principals, tenancy — are server-side configuration read on the authorisation
path of every channel. Before this module each was re-read and re-validated from
YAML on **every request**, at a cost linear in the number of organisations,
resources and grants:

    scale                orgs / resources / grants   agent call   MI query   Copilot
    A  pilot                2 /      5 /      4          7 ms       1 ms       5 ms
    B  early platform      30 /    251 /    150        232 ms      14 ms     147 ms
    C  established        150 /  2 101 /    750      1 768 ms     102 ms   1 078 ms
    D  A2A                400 /  6 001 /  2 000      5 253 ms     272 ms   3 116 ms

paid before any data is touched. This module makes it once per worker per
content version instead.

Why not ``mi_agent_api.serving_cache``
--------------------------------------
That class has the right design and this module deliberately copies its rules,
but it cannot be reused here: ``serving_cache`` imports ``trakt_core.perf``, so
depending on it from ``trakt_core`` would invert the dependency direction that
``tests/test_governance_dependency_direction.py`` exists to protect. This module
is therefore standard library only — which is also what lets a CLI, a pipeline
job and a test benefit from it without importing a web stack.

The rules, which are the safety argument
----------------------------------------
* **Content identity, never time.** A key is ``(path, mtime_ns, size)``. There is
  no TTL: a TTL would be both unsafe (stale inside the window) and pointless (a
  changed file already misses). An edited registry is picked up on the next call.
* **Absence is a cached state too.** "No file" is a real, meaningful answer for
  every one of these registries — it selects compatibility mode — so it is
  cached under a distinct key rather than re-stat'ed into a re-parse.
* **Configuration only. Never a decision.** What is memoised is the *parsed and
  validated registry*, not "may this caller reach that resource". Authorisation
  still runs in full on every request, against the cached configuration, and
  ``ExecutionContext.entitlements`` is still resolved per request and frozen. A
  revoked grant therefore stops working on the next request, not on a TTL.
* **Injected collaborators bypass the cache.** ``load_entitlement_store`` accepts
  explicit registries; when a caller supplies them the result depends on objects
  this module cannot fingerprint, so it is not cached at all.
* **Bounded, and thread-safe.** Small LRU under a lock — several worker threads
  will race on the first request and one of them wins harmlessly.
* **Degrades to the parse.** Any fault inside the cache falls through to the
  loader. Correctness never depends on the cache working.

Kill switch: ``TRAKT_CONFIG_CACHE=off`` restores the previous behaviour exactly.
"""

from __future__ import annotations

import logging
import os
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

logger = logging.getLogger("trakt_core.config_cache")

_OFF = {"0", "off", "false", "no"}

#: Bump when a change alters what a loader PRODUCES from unchanged bytes, so an
#: upgraded worker cannot serve a registry built by the old code.
CACHE_VERSION = "1"

#: Five registries × a handful of path/argument variants. Deliberately small:
#: this is a memo of a parse, not a store.
MAX_ENTRIES = 32

_LOCK = threading.Lock()
_CACHE: "OrderedDict[Tuple[Any, ...], Any]" = OrderedDict()
_STATS = {"hits": 0, "misses": 0}


def enabled() -> bool:
    return str(os.environ.get("TRAKT_CONFIG_CACHE", "on")).strip().lower() not in _OFF


def file_identity(path: Path) -> Tuple[Any, ...]:
    """``(resolved_path, mtime_ns, size)``, or ``(resolved_path, None, None)``.

    The second form is how an absent file is represented — a real state that must
    be cacheable, because "no organisations.yaml" means compatibility mode and
    re-deciding it per request is exactly the waste this module removes.

    A file replaced with identical bytes *and* an identical mtime is
    indistinguishable from an unchanged one. That is the same assumption
    ``serving_cache`` makes for filesystem sources and the same one ``make``
    has always made; where it matters, touch the file.
    """
    try:
        stat = path.stat()
        return (str(path), stat.st_mtime_ns, stat.st_size)
    except OSError:
        return (str(path), None, None)


def get_or_load(name: str, path: Path, loader: Callable[[], Any], *,
                extra_key: Tuple[Any, ...] = ()) -> Any:
    """Return the cached registry for ``path``'s current content, else load it.

    ``extra_key`` carries any argument that changes the RESULT for identical file
    content — ``default_tenant_id`` is the only one today. Omitting it would let
    two deployments sharing a process serve each other's default tenant.
    """
    if not enabled():
        return loader()

    try:
        key = (CACHE_VERSION, name) + file_identity(path) + tuple(extra_key)
    except Exception:  # noqa: BLE001 - a key we cannot build is a cache we skip
        logger.warning("config cache key failed for %s; loading directly", name)
        return loader()

    with _LOCK:
        if key in _CACHE:
            _CACHE.move_to_end(key)
            _STATS["hits"] += 1
            return _CACHE[key]

    # Loaded OUTSIDE the lock: these loaders parse and validate whole files, and
    # holding a global lock across that would serialise every worker thread on
    # the first request after a config change — replacing a slow path with a
    # contended one. A concurrent duplicate load is wasteful once and correct.
    value = loader()

    with _LOCK:
        _STATS["misses"] += 1
        _CACHE[key] = value
        _CACHE.move_to_end(key)
        while len(_CACHE) > MAX_ENTRIES:
            _CACHE.popitem(last=False)
    return value


def reset() -> None:
    """Drop every entry. For tests, and for an operator forcing a reload."""
    with _LOCK:
        _CACHE.clear()
        _STATS["hits"] = 0
        _STATS["misses"] = 0


def stats() -> Dict[str, int]:
    """Hit/miss counters, for tests and diagnostics."""
    with _LOCK:
        return {**_STATS, "entries": len(_CACHE)}
