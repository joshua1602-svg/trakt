"""trakt_core.audit — the governed-execution audit event.

One compact structured line per governed execution, on the existing ``logging``
infrastructure. This is not an observability platform; it is the minimum needed
to answer "who asked what, of which snapshot, and what happened".

What is deliberately **not** logged: the answer body, loan-level rows, tokens,
signed URLs, storage paths, connection strings, or borrower data. The event
carries identifiers and an outcome — nothing that would turn the log into a
second copy of the data.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

from .envelope import AuditMetadata, GovernedResult

AUDIT_LOGGER = "trakt.audit"

logger = logging.getLogger(AUDIT_LOGGER)

#: Keys that must never appear in an audit event, whatever a caller passes.
_FORBIDDEN_KEYS = frozenset({
    "answer", "result", "rows", "artifacts", "token", "download_url",
    "downloadUrl", "signed_url", "path", "connection_string", "authorization",
})


def audit_event_from_result(result: GovernedResult) -> Dict[str, Any]:
    """Project a :class:`GovernedResult` into a flat audit event."""
    audit: Optional[AuditMetadata] = result.audit
    if audit is not None:
        event = audit.to_dict()
    else:  # a result built without audit metadata still audits its identifiers
        event = {
            "capability": result.capability,
            "request_id": result.request_id,
            "correlation_id": result.correlation_id,
            "tenant_id": result.tenant_id,
            "portfolio_id": result.portfolio_id,
            "outcome": result.status,
            "error_code": result.error_code,
        }
    event.setdefault("outcome", result.status)
    event.setdefault("error_code", result.error_code)
    if result.snapshot is not None:
        event.setdefault("snapshot_id", result.snapshot.snapshot_id)
        event["source_kind"] = result.snapshot.source_kind
    event["schema_version"] = result.schema_version
    return {k: v for k, v in event.items() if k not in _FORBIDDEN_KEYS}


def emit_audit_event(result: GovernedResult, *, log: Optional[logging.Logger] = None) -> Dict[str, Any]:
    """Emit and return the audit event for ``result``.

    Returned so callers and tests can assert on it without parsing log output.
    Never raises: an audit failure must not fail a governed execution that has
    already succeeded.
    """
    event = audit_event_from_result(result)
    try:
        (log or logger).info("trakt.audit %s", json.dumps(event, default=str, sort_keys=True))
    except Exception:  # noqa: BLE001 - auditing must never break the request
        (log or logger).warning("audit emit failed for request_id=%s", result.request_id)
    return event
