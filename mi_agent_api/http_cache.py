"""mi_agent_api/http_cache.py — conditional responses for stable MI reads.

The dashboard's read routes answer from published, immutable artefacts: a given
(tenant, portfolio scope, run, source content) always produces the same body.
That makes them safe to serve conditionally, so a page reload, a second browser
tab or a returning user pays a revalidation instead of a recomputation.

Design decisions, and why:

* **The validator is derived from source identity, not from the body.** Hashing
  the body would require doing all the work first, which is the cost being
  avoided. :func:`begin` builds the ETag from the tenant, the governed portfolio
  scope, the run, the identity of the published bytes and a per-route version,
  then short-circuits BEFORE the route computes anything.
* **``no-cache``, not ``max-age``.** The browser must revalidate every time, so a
  newly published (or newly approved) run is never masked by a stale window.
  The saving is the 304 and the skipped recomputation, not a staleness window.
* **``private``.** Tenant data must never enter a shared or CDN cache.
* **Authentication and authorisation are unaffected.** These helpers are called
  from INSIDE route bodies, so FastAPI has already run the global
  ``Depends(auth_guard)``. A conditional request is authenticated exactly like an
  unconditional one; an unauthenticated caller is rejected before reaching here
  and never learns whether a validator would have matched.
* **Errors are never given a validator.** :func:`finish` withholds the ETag from
  any payload that reports failure or unavailability, so a transient error can
  never be revalidated into a "nothing changed" 304.

Disable with ``TRAKT_HTTP_CACHE=off`` — routes then behave exactly as before.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Iterable, Mapping, Optional

from trakt_core import perf

from . import request_scope, serving_cache

_OFF = {"0", "off", "false", "no"}

#: Bumped when a route's response SHAPE changes, so a client holding an old
#: validator revalidates into a fresh body rather than a 304.
ROUTE_VERSION = "1"

CACHE_CONTROL = "private, no-cache"


class NotModified(Exception):
    """Raised by :func:`begin` when the caller's validator still matches.

    Handled by the app's exception handler, which returns a bare 304.
    """

    def __init__(self, etag: str) -> None:
        super().__init__("not modified")
        self.etag = etag


def enabled() -> bool:
    return str(os.environ.get("TRAKT_HTTP_CACHE", "on")).strip().lower() not in _OFF


def _parse_if_none_match(raw: Optional[str]) -> set:
    """The client's validator set. Handles weak tags and the ``*`` wildcard."""
    if not raw:
        return set()
    out = set()
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        if token.startswith("W/"):
            token = token[2:]
        out.add(token.strip('"'))
    return out


def build_etag(*, route: str, tenant: Optional[str], scope: Optional[str],
               identity: Iterable[Any]) -> Optional[str]:
    """A deterministic weak validator, or ``None`` when identity is unknown.

    Returning ``None`` (rather than a guessed tag) is what stops an
    unidentifiable source being served conditionally.
    """
    parts = list(identity)
    if not parts or any(p is None for p in parts):
        return None
    digest = hashlib.sha256()
    for part in (ROUTE_VERSION, serving_cache.METHODOLOGY_VERSION, route,
                 tenant or "-", scope or "-", *parts):
        digest.update(repr(part).encode("utf-8", "replace"))
        digest.update(b"\x1f")
    return f'W/"{digest.hexdigest()[:32]}"'


def begin(request: Any, *, route: str, identity: Iterable[Any],
          scope: Optional[str] = None) -> Optional[str]:
    """Compute this response's validator and short-circuit when it matches.

    Raises :class:`NotModified` when the caller already holds the current
    representation. Returns the validator to hand to :func:`finish`, or ``None``
    when the response must not be cached at all.
    """
    if request is None or not enabled():
        return None
    try:
        etag = build_etag(route=route, tenant=serving_cache.resolved_tenant(),
                          scope=scope, identity=identity)
    except Exception:  # noqa: BLE001 - a validator fault must not fail the read
        return None
    if etag is None:
        return None
    try:
        presented = _parse_if_none_match(request.headers.get("if-none-match"))
    except Exception:  # noqa: BLE001
        return etag
    if presented and (etag.strip('W/"') in presented or "*" in presented):
        perf.cache_event("http_conditional", perf.CACHE_HIT)
        raise NotModified(etag)
    perf.cache_event("http_conditional", perf.CACHE_MISS)
    return etag


#: Route results, keyed by the SAME validator the response carries. A hit is by
#: construction a response the server would otherwise have re-sent as a 304, so
#: the cache cannot answer for a different client, portfolio, scope, as-of date,
#: data version or configuration version — each of those already changes the
#: validator, and an unestablishable identity yields no validator and therefore
#: no caching at all.
_RESULT_CACHE = serving_cache.BoundedCache("http_result", max_entries=96)


def cached(etag: Optional[str], build):
    """Return the result for ``etag``, computing it once per validator.

    ``etag is None`` means identity could not be established, so the result is
    computed and NOT stored — the same rule that governs whether a validator is
    issued at all. A payload that reports failure is never stored, so a
    transient error cannot be replayed to later callers.
    """
    if etag is None or not enabled():
        return build()

    def _build_and_screen():
        payload = build()
        if not _is_successful(payload):
            raise _Unsuccessful(payload)
        return payload

    try:
        return _RESULT_CACHE.get_or_build(etag, _build_and_screen)
    except _Unsuccessful as exc:
        return exc.payload


class _Unsuccessful(Exception):
    """Carries a failed payload out of the cache builder so it is not stored."""

    def __init__(self, payload: Any) -> None:
        super().__init__("payload reported failure")
        self.payload = payload


#: Keys whose falsity marks a payload as an error / unavailable state. A payload
#: reporting any of these is served WITHOUT a validator.
_FAILURE_FLAGS = ("ok", "available")
_ERROR_KEYS = ("error", "reason")


def _is_successful(payload: Any) -> bool:
    if not isinstance(payload, Mapping):
        return True
    for flag in _FAILURE_FLAGS:
        if flag in payload and not payload.get(flag):
            return False
    # ``reason`` alone is not a failure (routes use it for disclosure); an
    # explicit ``error`` always is.
    if payload.get("error"):
        return False
    return True


def finish(response: Any, etag: Optional[str], payload: Any) -> Any:
    """Attach the validator to a SUCCESSFUL response and return the payload.

    A failed or unavailable payload is returned unchanged and uncached, so a
    transient failure can never be revalidated into a 304.
    """
    if response is None or etag is None or not enabled():
        return payload
    try:
        if _is_successful(payload):
            response.headers["ETag"] = etag
            response.headers["Cache-Control"] = CACHE_CONTROL
        else:
            response.headers["Cache-Control"] = "no-store"
    except Exception:  # noqa: BLE001 - header assembly must not fail the read
        pass
    return payload


# --------------------------------------------------------------------------- #
# Source identity
# --------------------------------------------------------------------------- #
def config_fingerprint(client_id: Optional[str]) -> str:
    """A short content hash of the governed client configuration in force.

    Used as part of a route's identity so that re-approving a client's
    configuration invalidates every cached result derived from it. Never raises:
    an absent or unreadable configuration fingerprints as ``"-"``, which is
    stable and therefore still cacheable.
    """
    try:
        from . import currency as currency_mod
        location = currency_mod.client_config_path(client_id)
        if not location:
            return "-"
        doc = currency_mod._load_client_config(location)
        if not doc:
            return "-"
        blob = json.dumps(doc, sort_keys=True, default=str)
        return hashlib.sha256(blob.encode("utf-8", "replace")).hexdigest()[:16]
    except Exception:  # noqa: BLE001 - identity must never fail a read
        return "-"


def dataset_identity(client_id: Optional[str], run_id: Optional[str], *,
                     include_pipeline: bool = False) -> Optional[list]:
    """The immutable identity of the data a run-scoped route answers from.

    Funded side: the ETag of the dated platform canonical for the run. Pipeline
    side (only where the route depends on it): the identity of the discovered
    weekly extract set. Both are already resolved and memoised by the serving
    caches, so this adds no storage round-trip beyond the first per request.

    ``None`` when identity cannot be established — the route is then served
    unconditionally, exactly as before.
    """
    return request_scope.memo(
        f"dataset_identity|{client_id}|{run_id}|{include_pipeline}",
        lambda: _dataset_identity_uncached(client_id, run_id, include_pipeline))


def _dataset_identity_uncached(client_id: Optional[str], run_id: Optional[str],
                               include_pipeline: bool) -> Optional[list]:
    from . import datasets as datasets_mod
    from . import platform_snapshots_blob as blob_mod

    identity: list = [client_id or "-", run_id or "-"]
    # The governed client CONFIGURATION is part of what a response means (it
    # decides, among other things, the reporting currency), so a configuration
    # change must produce a different validator. Absent config contributes a
    # stable marker rather than nothing, so identity stays establishable.
    identity.append(("config", config_fingerprint(client_id)))
    try:
        root = datasets_mod._onboarding_output_root()
    except Exception:  # noqa: BLE001
        return None
    if not root:
        return None
    identity.append(str(root))

    if blob_mod.is_blob_root(root) and run_id:
        try:
            from apps.blob_trigger_app.storage import open_storage
            etag = blob_mod.canonical_etag(root, open_storage(), run_id)
        except Exception:  # noqa: BLE001
            etag = None
        if etag is None:
            return None
        identity.append(("funded_etag", etag))
    else:
        # A non-blob root: fall back to the active dataset's own content
        # fingerprint, which the governance layer already computes.
        try:
            descriptor = datasets_mod.describe_active_dataset()
        except Exception:  # noqa: BLE001
            return None
        if not getattr(descriptor, "content_hash", None):
            return None
        identity.append(("funded_hash", descriptor.content_hash))

    if include_pipeline:
        try:
            from . import pipeline_contract as pipeline_mod
            proot = datasets_mod._pipeline_discovery_root()
            if not proot:
                identity.append(("pipeline", "none"))
            else:
                sources = pipeline_mod.discover_pipeline_sources(
                    proot, client_id=client_id)
                identity.append(("pipeline", tuple(
                    (s.get("source_file"), s.get("pipeline_as_of_date"),
                     s.get("row_count")) for s in sources)))
        except Exception:  # noqa: BLE001 - an unknown pipeline means no validator
            return None
    return identity
