#!/usr/bin/env python3
"""Durable storage for generated batches and the reporting-position index.

A generated batch is **immutable**. It is written once, at approval, and the
delivery worker renders whatever it finds there — it never regenerates. That is
what makes a retry cheap (a card render, not an MI run) and what makes the
audit trail meaningful: the message an operator reads three weeks later is the
message that was sent, not a fresh interpretation of the same week.

The reporting index is the small piece of state that makes corrections
deterministic. It maps a reporting position — tenant, portfolio, update type,
source dates — to the batch that last described it. A corrected upload for the
same reporting date finds the earlier batch there and labels itself a correction
of it; a duplicate upload finds its own batch id and stops.
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
from typing import Any, Dict, List, Optional

from apps.blob_trigger_app.storage import Storage

from .contract import (
    MessageItem, NotificationBatch, NotificationMessage,
)
from .layout import NotificationLayout

logger = logging.getLogger("trakt_notifications.store")


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat()


def _message_from_dict(data: Dict[str, Any]) -> NotificationMessage:
    return NotificationMessage(
        message_type=data.get("message_type", ""),
        headline=data.get("headline", ""),
        summary=data.get("summary"),
        severity=data.get("severity", "info"),
        items=[MessageItem(
            text=i.get("text", ""), metric=i.get("metric"), raw=i.get("raw"),
            insight_id=i.get("insight_id"),
            unavailable=bool(i.get("unavailable")))
            for i in (data.get("items") or [])],
        deep_link=data.get("deep_link"),
        deep_link_label=data.get("deep_link_label") or "Open Trakt",
        insight_ids=list(data.get("insight_ids") or []),
        unavailable_checks=list(data.get("unavailable_checks") or []),
        recommendation_level=data.get("recommendation_level"),
        provenance=dict(data.get("provenance") or {}),
    )


def batch_from_dict(data: Dict[str, Any]) -> NotificationBatch:
    """Rehydrate a stored batch.

    ``notification_batch_id`` and ``reporting_key`` are recomputed from the
    identity fields rather than read back, so a tampered or hand-edited stored
    document cannot present itself under a different identity than its contents
    imply.
    """
    return NotificationBatch(
        tenant_id=data.get("tenant_id", ""),
        portfolio_id=data.get("portfolio_id", ""),
        portfolio_context=data.get("portfolio_context", "total"),
        update_type=data.get("update_type", ""),
        approved_run_ids=list(data.get("approved_run_ids") or []),
        source_dates=dict(data.get("source_dates") or {}),
        messages=[_message_from_dict(m) for m in (data.get("messages") or [])],
        generated_at=data.get("generated_at"),
        methodology_versions=dict(data.get("methodology_versions") or {}),
        provenance=dict(data.get("provenance") or {}),
        delivery_eligibility=dict(data.get("delivery_eligibility") or {}),
        corrects_batch_id=data.get("corrects_batch_id"),
        version=data.get("version", "1"),
    )


class BatchStore:
    """Immutable generated batches, plus the reporting-position index."""

    def __init__(self, storage: Storage,
                 layout: Optional[NotificationLayout] = None):
        self.storage = storage
        self.layout = layout or NotificationLayout.from_env()

    # -- batches ----------------------------------------------------------- #
    def save(self, batch: NotificationBatch) -> str:
        """Persist a batch, refusing to alter one that already exists.

        Content is immutable after generation, except through an explicit
        correction — which is a NEW batch with its own id, not an edit of this
        one. Overwriting here would break the audit trail's central promise.
        """
        uri = self.layout.batch_uri(batch.tenant_id, batch.notification_batch_id)
        if self.storage.exists(uri):
            logger.info("notifications: batch %s already stored; leaving the "
                        "existing payload untouched",
                        batch.notification_batch_id)
            return uri
        self.storage.write_text(uri, batch.to_json())
        return uri

    def load(self, tenant_id: str,
             batch_id: str) -> Optional[NotificationBatch]:
        uri = self.layout.batch_uri(tenant_id, batch_id)
        if not self.storage.exists(uri):
            return None
        try:
            return batch_from_dict(json.loads(self.storage.read_text(uri)))
        except Exception as exc:  # noqa: BLE001
            logger.warning("notifications: unreadable batch %s (%s)",
                           batch_id, exc)
            return None

    def exists(self, tenant_id: str, batch_id: str) -> bool:
        return self.storage.exists(self.layout.batch_uri(tenant_id, batch_id))

    # -- reporting-position index ------------------------------------------ #
    def reporting_pointer(self, tenant_id: str,
                          reporting_key: str) -> Optional[Dict[str, Any]]:
        uri = self.layout.reporting_uri(tenant_id, reporting_key)
        if not self.storage.exists(uri):
            return None
        try:
            return json.loads(self.storage.read_text(uri))
        except Exception:  # noqa: BLE001
            return None

    def record_reporting(self, batch: NotificationBatch) -> Dict[str, Any]:
        """Point a reporting position at the batch that now describes it.

        ``history`` keeps every batch id that has described this position, in
        order, so a correction of a correction is still traceable back to the
        original.
        """
        existing = self.reporting_pointer(batch.tenant_id, batch.reporting_key) or {}
        history: List[str] = list(existing.get("history") or [])
        if existing.get("notification_batch_id") \
                and existing["notification_batch_id"] not in history:
            history.append(str(existing["notification_batch_id"]))
        document = {
            "reporting_key": batch.reporting_key,
            "tenant_id": batch.tenant_id,
            "portfolio_context": batch.portfolio_context,
            "update_type": batch.update_type,
            "source_dates": dict(batch.source_dates),
            "notification_batch_id": batch.notification_batch_id,
            "approved_run_ids": list(batch.approved_run_ids),
            "history": history,
            "updated_at": _now(),
        }
        self.storage.write_text(
            self.layout.reporting_uri(batch.tenant_id, batch.reporting_key),
            json.dumps(document, indent=2, default=str))
        return document

    def prior_batch_id(self, tenant_id: str,
                       reporting_key: str) -> Optional[str]:
        """The batch that last described this reporting position, if any."""
        pointer = self.reporting_pointer(tenant_id, reporting_key)
        return (pointer or {}).get("notification_batch_id")
