#!/usr/bin/env python3
"""The delivery worker — turns durable intent into Teams messages.

Runs independently of ingestion and of MI. A restart between approval and
delivery loses nothing; a Teams outage delays messages rather than dropping
them; retrying costs a card render, never an MI recalculation, because the
payload was rendered once at generation time and stored.

Two ordering rules the worker enforces:

* **Within a batch, the Portfolio Update precedes the Risk Review.** The reader
  needs to know what changed before being told what to consider about it.
  If the update has not yet been delivered, the risk message waits.

* **A permanently failed update releases its risk message.** Blocking the Risk
  Review forever because the Portfolio Update could not be delivered would
  suppress the more important of the two messages for the least important
  reason.
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from apps.blob_trigger_app.storage import Storage, open_storage

from . import cards, telemetry
from .config import max_attempts, retry_delay_seconds
from .contract import (
    MESSAGE_PORTFOLIO_UPDATE, NotificationBatch, NotificationMessage,
)
from .layout import NotificationLayout
from .outbox import (
    DELIVERED, Outbox, OutboxItem, STATE_PERMANENT, STATE_SENT,
)
from .store import BatchStore
from .teams_client import TeamsClient, TeamsSendError, from_config

logger = logging.getLogger("trakt_notifications.delivery")


def _now() -> _dt.datetime:
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0)


@dataclass
class DeliveryReport:
    """What one worker pass did. Returned so a caller can log or assert on it."""

    tenant_id: str
    attempted: int = 0
    sent: int = 0
    retried: int = 0
    failed: int = 0
    skipped: int = 0
    blocked: int = 0
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tenant_id": self.tenant_id, "attempted": self.attempted,
            "sent": self.sent, "retried": self.retried, "failed": self.failed,
            "skipped": self.skipped, "blocked": self.blocked,
            "errors": list(self.errors),
        }


class DeliveryWorker:
    """One pass over one tenant's due outbox items."""

    def __init__(self, *, storage: Optional[Storage] = None,
                 layout: Optional[NotificationLayout] = None,
                 client: Optional[TeamsClient] = None):
        self.storage = storage or open_storage()
        self.layout = layout or NotificationLayout.from_env()
        self.outbox = Outbox(self.storage, self.layout)
        self.batches = BatchStore(self.storage, self.layout)
        self.client = client or from_config()

    def run(self, tenant_id: str, *, limit: int = 50) -> DeliveryReport:
        """Attempt every due item, in delivery order."""
        report = DeliveryReport(tenant_id=tenant_id)
        due = self.outbox.due(tenant_id)[:limit]
        if not due:
            return report

        if not self.client.configured:
            # Fail closed and loudly, without consuming retry attempts on a
            # condition no amount of retrying will fix.
            telemetry.failure(telemetry.EVT_SEND_FAILURE, tenant_id=tenant_id,
                              error="bot identity not configured",
                              due_items=len(due))
            report.errors.append("the Teams bot identity is not configured")
            report.skipped = len(due)
            return report

        # Which batches already have their Portfolio Update settled, so the
        # ordering rule can be applied without re-reading storage per item.
        settled = self._settled_updates(tenant_id)

        for item in due:
            if self._blocked_by_ordering(item, settled):
                report.blocked += 1
                continue
            outcome = self._deliver(item)
            report.attempted += 1
            if outcome == STATE_SENT:
                report.sent += 1
                if item.message_type == MESSAGE_PORTFOLIO_UPDATE:
                    settled.add(item.notification_batch_id)
            elif outcome == STATE_PERMANENT:
                report.failed += 1
                if item.message_type == MESSAGE_PORTFOLIO_UPDATE:
                    # Release the Risk Review rather than suppressing it.
                    settled.add(item.notification_batch_id)
            elif outcome is None:
                report.skipped += 1
            else:
                report.retried += 1
        return report

    # -- ordering ---------------------------------------------------------- #
    def _settled_updates(self, tenant_id: str) -> set:
        """Batches whose Portfolio Update has been delivered or given up on."""
        settled = set()
        for item in self.outbox.list(tenant_id):
            if item.message_type != MESSAGE_PORTFOLIO_UPDATE:
                continue
            if item.state in DELIVERED or item.state == STATE_PERMANENT:
                settled.add(item.notification_batch_id)
        return settled

    def _blocked_by_ordering(self, item: OutboxItem, settled: set) -> bool:
        if item.message_type == MESSAGE_PORTFOLIO_UPDATE:
            return False
        # A batch with no Portfolio Update at all (the message type is disabled
        # by configuration) must not block its Risk Review forever.
        has_update = any(
            i.message_type == MESSAGE_PORTFOLIO_UPDATE
            and i.notification_batch_id == item.notification_batch_id
            for i in self.outbox.list(item.tenant_id))
        return has_update and item.notification_batch_id not in settled

    # -- one item ---------------------------------------------------------- #
    def _deliver(self, item: OutboxItem) -> Optional[str]:
        """Claim, render, send, record. Returns the resulting state."""
        claimed = self.outbox.claim(item)
        if claimed is None:
            # Superseded or already delivered since the listing.
            telemetry.emit(telemetry.EVT_DEDUPLICATED,
                           tenant_id=item.tenant_id, item_id=item.item_id,
                           message_type=item.message_type)
            return None

        batch = self.batches.load(item.tenant_id, item.notification_batch_id)
        if batch is None:
            self.outbox.mark_permanent(
                claimed, error="the generated notification payload is missing")
            telemetry.failure(telemetry.EVT_SEND_FAILURE,
                              tenant_id=item.tenant_id, item_id=item.item_id,
                              error="payload missing")
            return STATE_PERMANENT

        message = batch.message(item.message_type)
        if message is None:
            self.outbox.mark_permanent(
                claimed, error="the payload carries no message of this type")
            return STATE_PERMANENT

        telemetry.emit(telemetry.EVT_SEND_ATTEMPT, tenant_id=item.tenant_id,
                       item_id=item.item_id, message_type=item.message_type,
                       attempt=claimed.attempt_count + 1,
                       recipient_id=item.recipient_id,
                       batch_id=item.notification_batch_id)
        try:
            result = self.client.send_card(
                service_url=claimed.service_url or "",
                conversation_id=claimed.conversation_id or "",
                attachment=cards.attachment(batch, message),
                summary=cards.summary_text(batch, message))
        except TeamsSendError as exc:
            return self._record_failure(claimed, exc)
        except Exception as exc:  # noqa: BLE001 - an unexpected render/transport
            return self._record_failure(               # fault is retryable once
                claimed, TeamsSendError(str(exc), retryable=True))

        self.outbox.mark_sent(claimed, teams_message_id=result.teams_message_id)
        telemetry.emit(telemetry.EVT_SEND_SUCCESS, tenant_id=item.tenant_id,
                       item_id=item.item_id, message_type=item.message_type,
                       recipient_id=item.recipient_id,
                       teams_message_id=result.teams_message_id)
        if message.deep_link:
            telemetry.emit(telemetry.EVT_DEEP_LINK, tenant_id=item.tenant_id,
                           item_id=item.item_id,
                           # The path only — query parameters carry the
                           # reporting date and context, which are data.
                           target=message.deep_link.split("?")[0])
        return STATE_SENT

    def _record_failure(self, item: OutboxItem,
                        exc: TeamsSendError) -> str:
        if not exc.retryable:
            self.outbox.mark_permanent(item, error=str(exc))
            telemetry.failure(telemetry.EVT_SEND_FAILURE,
                              tenant_id=item.tenant_id, item_id=item.item_id,
                              message_type=item.message_type,
                              status=exc.status, error=str(exc),
                              retryable=False)
            return STATE_PERMANENT

        delay = retry_delay_seconds(item.attempt_count + 1)
        next_at = (_now() + _dt.timedelta(seconds=delay)).isoformat()
        updated = self.outbox.mark_retryable(
            item, error=str(exc), next_attempt_at=next_at,
            max_attempts=max_attempts())
        if updated.state == STATE_PERMANENT:
            telemetry.failure(telemetry.EVT_SEND_FAILURE,
                              tenant_id=item.tenant_id, item_id=item.item_id,
                              message_type=item.message_type,
                              attempts=updated.attempt_count,
                              error=str(exc), retryable=False)
            return STATE_PERMANENT
        telemetry.emit(telemetry.EVT_RETRY_SCHEDULED, tenant_id=item.tenant_id,
                       item_id=item.item_id, message_type=item.message_type,
                       attempt=updated.attempt_count, next_attempt_at=next_at,
                       status=exc.status)
        return updated.state


def run_once(tenant_id: str, *, storage: Optional[Storage] = None,
             client: Optional[TeamsClient] = None,
             limit: int = 50) -> DeliveryReport:
    """One delivery pass. The entry point for a timer trigger or the CLI."""
    return DeliveryWorker(storage=storage, client=client).run(
        tenant_id, limit=limit)
