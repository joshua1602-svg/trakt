"""mi_agent_api/perf_http.py — HTTP instrumentation for the MI Agent API.

A pure-ASGI middleware (deliberately not ``BaseHTTPMiddleware``) so the
``ContextVar`` holding the request's :class:`~trakt_core.perf.Collector` is set
in the same task that then invokes the application. Starlette copies the
context into the threadpool when it runs a synchronous route, so the ``def``
routes in :mod:`mi_agent_api.app` see the collector without any per-route
plumbing.

What it produces per request:

* one structured ``perf.request`` log record — route, tenant, portfolio
  context, run id, total ms, per-stage ms, blob/read counters, cache outcomes,
  response bytes and status;
* a ``Server-Timing`` response header so the same breakdown is visible in the
  browser's performance panel without a log search.

It records names, durations and counts only. No loan-level data, no principal
identifiers and no storage paths are logged.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

from trakt_core import perf

from . import request_scope

logger = logging.getLogger("mi_agent_api.perf")

#: Paths that are pure liveness/probe noise — instrumented but not logged, so a
#: health-check loop cannot drown the performance record.
_QUIET_PATHS = {"/health", "/", "/openapi.json", "/docs", "/redoc", "/favicon.ico"}

#: Query parameters worth stamping onto the record. These are governed selectors
#: (which portfolio / which run), never data values.
_CONTEXT_PARAMS = ("portfolioId", "portfolioContext", "client_id", "runId",
                   "run_id", "toRunId", "to_run_id", "testId", "dataset")


def _decode_qs(raw: bytes) -> Dict[str, str]:
    """Parse a query string into a flat dict, keeping the first value per key."""
    out: Dict[str, str] = {}
    try:
        from urllib.parse import parse_qsl
        for key, value in parse_qsl(raw.decode("latin-1"), keep_blank_values=False):
            out.setdefault(key, value)
    except Exception:  # noqa: BLE001 - instrumentation must never raise
        return {}
    return out


def _strip_prefix(path: str, prefix: str) -> str:
    """The route as the application sees it, with the gateway prefix removed."""
    if prefix and path.startswith(prefix):
        return path[len(prefix):] or "/"
    return path


class PerfMiddleware:
    """Pure-ASGI performance collector. Never alters the response body."""

    def __init__(self, app: Any, *, api_prefix: str = "") -> None:
        self.app = app
        self.api_prefix = api_prefix or ""

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope.get("type") != "http" or not perf.enabled():
            await self.app(scope, receive, send)
            return

        raw_path = scope.get("path") or "/"
        route = _strip_prefix(raw_path, self.api_prefix)
        params = _decode_qs(scope.get("query_string") or b"")
        headers = {k.decode("latin-1").lower(): v.decode("latin-1")
                   for k, v in scope.get("headers") or []}

        fields: Dict[str, Any] = {
            "route": route,
            "method": scope.get("method"),
            "request_id": headers.get("x-request-id") or headers.get("x-correlation-id"),
        }
        for key in _CONTEXT_PARAMS:
            if key in params:
                fields[key] = params[key]
        try:
            from .dependencies import default_tenant_id
            fields["tenant_id"] = default_tenant_id()
        except Exception:  # noqa: BLE001 - tenant labelling is best-effort
            pass

        state: Dict[str, Any] = {"status": None, "bytes": 0}

        with perf.collect(**fields) as collector:
            async def _send(message: Any) -> None:
                try:
                    kind = message.get("type")
                    if kind == "http.response.start":
                        state["status"] = message.get("status")
                        if collector is not None:
                            raw_headers: List[Tuple[bytes, bytes]] = list(
                                message.get("headers") or [])
                            raw_headers.append(
                                (b"server-timing",
                                 collector.server_timing().encode("latin-1")))
                            message["headers"] = raw_headers
                    elif kind == "http.response.body":
                        state["bytes"] += len(message.get("body") or b"")
                except Exception:  # noqa: BLE001 - never break the response
                    pass
                await send(message)

            # The request scope rides with the collector: one governed
            # revalidation per request instead of one per call site. See
            # mi_agent_api/request_scope.py.
            try:
                with request_scope.scope():
                    await self.app(scope, receive, _send)
            finally:
                if collector is not None and route not in _QUIET_PATHS:
                    _emit(collector, status=state["status"], size=state["bytes"])


def _emit(collector: "perf.Collector", *, status: Optional[int], size: int) -> None:
    """Write the one structured record for this request. Never raises."""
    try:
        record = collector.snapshot()
        record["status"] = status
        record["response_bytes"] = size
        logger.info("perf %s", json.dumps(record, default=str, sort_keys=True))
    except Exception:  # noqa: BLE001
        pass
