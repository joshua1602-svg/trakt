#!/usr/bin/env python3
"""Structured operational telemetry for the notification pathway.

One event per meaningful step, so an operator can answer "what happened to
Tuesday's update" from logs alone rather than by reading storage. Events are
emitted through the standard logger — no new observability dependency — with a
stable ``event`` field so they can be queried in Application Insights.

What is never logged, enforced by :func:`_scrub` rather than by care:

* portfolio VALUES — a headline or a metric would put commercially sensitive
  figures into a log store with a different retention and access model from the
  data itself;
* credentials, bearer tokens and signed URLs;
* full Teams conversation references — the conversation id is recorded because
  an operator needs it to correlate a delivery, the rest is not.

Recipient ids appear, and they are already hashed by
:func:`trakt_notifications.recipients.recipient_id`, so a log carries no
directory identifier.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger("trakt_notifications.telemetry")

# Event names — one per step of the pathway described in the plan.
EVT_UPDATE_RECEIVED = "notification.update_received"
EVT_SUPPRESSED = "notification.suppressed"
EVT_BATCH_GENERATED = "notification.batch_generated"
EVT_INSIGHTS_SELECTED = "notification.insights_selected"
EVT_RECIPIENTS_RESOLVED = "notification.recipients_resolved"
EVT_OUTBOX_CREATED = "notification.outbox_created"
EVT_SEND_ATTEMPT = "notification.send_attempt"
EVT_SEND_SUCCESS = "notification.send_success"
EVT_SEND_FAILURE = "notification.send_failure"
EVT_RETRY_SCHEDULED = "notification.retry_scheduled"
EVT_CORRECTION = "notification.correction"
EVT_DEDUPLICATED = "notification.deduplicated"
EVT_SUPERSEDED = "notification.superseded"
EVT_DEEP_LINK = "notification.deep_link_target"

#: Field names that must never carry a value into a log line.
_FORBIDDEN_KEYS = frozenset({
    "headline", "summary", "text", "items", "value", "amount", "balance",
    "metrics", "conversation_reference", "token", "access_token", "password",
    "app_password", "client_secret", "service_url", "authorization",
})

#: Keys whose value is truncated rather than dropped — long enough to correlate,
#: short enough not to become a payload.
_TRUNCATE = 200


def _scrub(fields: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in (fields or {}).items():
        if key.lower() in _FORBIDDEN_KEYS:
            continue
        if isinstance(value, (dict, list)):
            # Structures are summarised by size, not serialised: a nested
            # payload is exactly how a portfolio value ends up in a log.
            out[key] = f"<{type(value).__name__}:{len(value)}>"
        elif isinstance(value, str):
            out[key] = value[:_TRUNCATE]
        else:
            out[key] = value
    return out


def emit(event: str, *, level: int = logging.INFO, **fields: Any) -> None:
    """Record one telemetry event.

    Never raises: telemetry failing must not fail a delivery, and an operator
    losing one log line is a far better outcome than a notification pathway
    that stops because a field would not serialise.
    """
    try:
        payload = _scrub(fields)
        logger.log(level, "%s %s", event,
                   json.dumps(payload, sort_keys=True, default=str))
    except Exception:  # noqa: BLE001 - telemetry must never break delivery
        logger.log(level, "%s <telemetry payload unavailable>", event)


def failure(event: str, *, error: Optional[str] = None, **fields: Any) -> None:
    """A failure event, with the error message trimmed and de-identified."""
    from .outbox import _redact
    emit(event, level=logging.WARNING,
         error=_redact(error or ""), **fields)
