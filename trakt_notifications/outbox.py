#!/usr/bin/env python3
"""The durable outbox — at-least-once processing, effectively-once experience.

Nothing is sent from the ingestion request path. Approval writes durable intent
and returns; a worker turns intent into a Teams message later, and retries
without the approving operator ever knowing there was a retry. An API restart
between the two loses nothing, because the intent is on storage rather than in
a queue in memory.

The gap between "at least once" and "the user sees it once" is closed by
identity rather than by locking. An outbox item's id is derived from the
message id — which is derived from the deterministic batch id — and the
recipient. Re-running the generator over the same approved run produces the
same item ids, and :meth:`Outbox.enqueue` will not overwrite an item that has
already been sent. A duplicate upload therefore cannot produce a second
message, and neither can a duplicate worker pass.

States::

    pending      → claimed → sent
                          ↘ retryable_failure → (claimed …)
                          ↘ permanent_failure
    pending      → superseded          (a corrected upload overtook it)
    sent         → corrected           (a later batch corrected it)

``claimed`` uses a claim token rather than a lease lock. Blob writes are
last-writer-wins, so two workers racing for one item would both believe they
hold it; the token makes that detectable and the send-time idempotency check
makes it harmless. v1 runs a single worker, and this is documented as the
reason it must stay that way rather than left as an assumption.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import logging
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from apps.blob_trigger_app.storage import Storage

from .contract import MESSAGE_SEQUENCE, NotificationBatch
from .layout import NotificationLayout
from .recipients import Recipient

logger = logging.getLogger("trakt_notifications.outbox")

STATE_PENDING = "pending"
STATE_CLAIMED = "claimed"
STATE_SENT = "sent"
STATE_RETRYABLE = "retryable_failure"
STATE_PERMANENT = "permanent_failure"
STATE_SUPERSEDED = "superseded"
STATE_CORRECTED = "corrected"

STATES = (STATE_PENDING, STATE_CLAIMED, STATE_SENT, STATE_RETRYABLE,
          STATE_PERMANENT, STATE_SUPERSEDED, STATE_CORRECTED)

#: States a worker may pick up.
CLAIMABLE = (STATE_PENDING, STATE_RETRYABLE)
#: States that mean the user has already seen this message. Never re-sent.
DELIVERED = (STATE_SENT, STATE_CORRECTED)
#: States nothing further will happen to.
TERMINAL = (STATE_SENT, STATE_PERMANENT, STATE_SUPERSEDED, STATE_CORRECTED)


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat()


def item_id(message_id: str, recipient_id: str) -> str:
    """Deterministic per (message, recipient).

    This is the whole idempotency mechanism: regenerating a batch for the same
    approved run yields the same message id, so the same item id, so an
    already-sent item is recognised rather than duplicated.
    """
    raw = f"{message_id}|{recipient_id}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


@dataclass
class OutboxItem:
    """One message, for one recipient, and everything needed to audit it."""

    item_id: str
    tenant_id: str
    portfolio_context: str
    notification_batch_id: str
    reporting_key: str
    message_type: str
    message_id: str
    recipient_id: str
    #: Denormalised so the worker can address the chat without re-reading the
    #: recipient record — and so the audit trail records where it actually went.
    conversation_id: Optional[str] = None
    service_url: Optional[str] = None
    sequence: int = 99
    update_type: Optional[str] = None
    approved_run_ids: List[str] = field(default_factory=list)
    source_dates: Dict[str, Any] = field(default_factory=dict)
    state: str = STATE_PENDING
    attempt_count: int = 0
    last_error: Optional[str] = None
    #: The Teams activity id of the delivered message.
    teams_message_id: Optional[str] = None
    claim_token: Optional[str] = None
    claimed_at: Optional[str] = None
    sent_at: Optional[str] = None
    next_attempt_at: Optional[str] = None
    superseded_by: Optional[str] = None
    corrected_by: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OutboxItem":
        known = {f for f in cls.__dataclass_fields__}  # noqa: SLF001
        return cls(**{k: v for k, v in (data or {}).items() if k in known})

    @property
    def terminal(self) -> bool:
        return self.state in TERMINAL


class Outbox:
    """Durable outbox on the existing storage abstraction."""

    def __init__(self, storage: Storage,
                 layout: Optional[NotificationLayout] = None):
        self.storage = storage
        self.layout = layout or NotificationLayout.from_env()

    # -- persistence ------------------------------------------------------- #
    def _write(self, item: OutboxItem) -> OutboxItem:
        item.updated_at = _now()
        item.created_at = item.created_at or item.updated_at
        self.storage.write_text(
            self.layout.outbox_uri(item.tenant_id, item.item_id),
            json.dumps(item.to_dict(), indent=2, default=str))
        return item

    def load(self, tenant_id: str, iid: str) -> Optional[OutboxItem]:
        uri = self.layout.outbox_uri(tenant_id, iid)
        if not self.storage.exists(uri):
            return None
        try:
            return OutboxItem.from_dict(json.loads(self.storage.read_text(uri)))
        except Exception as exc:  # noqa: BLE001
            logger.warning("notifications: unreadable outbox item %s (%s)",
                           iid, exc)
            return None

    def list(self, tenant_id: str, *,
             states: Optional[List[str]] = None) -> List[OutboxItem]:
        """One tenant's outbox, ordered for delivery.

        Ordered by batch and then by message sequence so the Portfolio Update
        is always offered to the worker before the Risk Review for the same
        batch. Delivery order is a product requirement, not an accident of
        listing order.
        """
        out: List[OutboxItem] = []
        for uri in self.storage.list(self.layout.outbox_prefix(tenant_id)):
            if not uri.endswith(".json"):
                continue
            try:
                item = OutboxItem.from_dict(
                    json.loads(self.storage.read_text(uri)))
            except Exception:  # noqa: BLE001 - one bad record, not the pass
                continue
            if states is None or item.state in states:
                out.append(item)
        return sorted(out, key=lambda i: (i.created_at or "",
                                          i.notification_batch_id,
                                          i.sequence, i.item_id))

    # -- enqueue ----------------------------------------------------------- #
    def enqueue(self, batch: NotificationBatch,
                recipients: List[Recipient]) -> List[OutboxItem]:
        """Write one durable item per (message, recipient).

        Existing items are inspected rather than blindly overwritten: an item
        already delivered stays delivered, so re-running an approved update
        cannot resend it. This is where "duplicate upload → do not send again"
        is actually enforced.
        """
        written: List[OutboxItem] = []
        for message in sorted(batch.messages, key=lambda m: m.sequence):
            mid = batch.message_id(message.message_type)
            for recipient in recipients:
                iid = item_id(mid, recipient.recipient_id)
                existing = self.load(batch.tenant_id, iid)
                if existing is not None and existing.state in DELIVERED:
                    logger.info("notifications: %s already delivered to %s, "
                                "not re-enqueued", message.message_type,
                                recipient.recipient_id)
                    continue
                if existing is not None and existing.state == STATE_CLAIMED:
                    # A worker holds it. Re-enqueueing would reset the attempt
                    # count and could produce a second send of the same message.
                    logger.info("notifications: %s is claimed by a worker, "
                                "not re-enqueued", iid)
                    continue
                item = OutboxItem(
                    item_id=iid,
                    tenant_id=batch.tenant_id,
                    portfolio_context=batch.portfolio_context,
                    notification_batch_id=batch.notification_batch_id,
                    reporting_key=batch.reporting_key,
                    message_type=message.message_type,
                    message_id=mid,
                    recipient_id=recipient.recipient_id,
                    conversation_id=recipient.conversation_id,
                    service_url=recipient.service_url,
                    sequence=MESSAGE_SEQUENCE.get(message.message_type, 99),
                    update_type=batch.update_type,
                    approved_run_ids=list(batch.approved_run_ids),
                    source_dates=dict(batch.source_dates),
                    created_at=(existing.created_at if existing else None),
                )
                written.append(self._write(item))
        return written

    # -- claim / complete -------------------------------------------------- #
    def claim(self, item: OutboxItem) -> Optional[OutboxItem]:
        """Take ownership of an item, or ``None`` if it is no longer claimable.

        Re-reads before writing so an item that reached a terminal state since
        the listing — superseded by a correction, or delivered by an earlier
        pass — is not picked up.
        """
        current = self.load(item.tenant_id, item.item_id)
        if current is None or current.state not in CLAIMABLE:
            return None
        current.state = STATE_CLAIMED
        current.claim_token = uuid.uuid4().hex
        current.claimed_at = _now()
        return self._write(current)

    def mark_sent(self, item: OutboxItem, *,
                  teams_message_id: Optional[str]) -> OutboxItem:
        item.state = STATE_SENT
        item.teams_message_id = teams_message_id
        item.sent_at = _now()
        item.last_error = None
        item.next_attempt_at = None
        item.claim_token = None
        return self._write(item)

    def mark_retryable(self, item: OutboxItem, *, error: str,
                       next_attempt_at: Optional[str],
                       max_attempts: int) -> OutboxItem:
        """Record a transient failure, escalating to permanent at the cap.

        Retrying forever would keep a broken destination in the operator's
        actionable list indefinitely; escalating makes it something they can
        see and close.
        """
        item.attempt_count += 1
        item.last_error = _redact(error)
        item.claim_token = None
        if item.attempt_count >= max_attempts:
            item.state = STATE_PERMANENT
            item.next_attempt_at = None
            item.last_error = (f"{item.last_error} (gave up after "
                               f"{item.attempt_count} attempts)")
        else:
            item.state = STATE_RETRYABLE
            item.next_attempt_at = next_attempt_at
        return self._write(item)

    def mark_permanent(self, item: OutboxItem, *, error: str) -> OutboxItem:
        item.attempt_count += 1
        item.state = STATE_PERMANENT
        item.last_error = _redact(error)
        item.next_attempt_at = None
        item.claim_token = None
        return self._write(item)

    # -- corrections and supersession -------------------------------------- #
    def supersede_undelivered(self, tenant_id: str, reporting_key: str, *,
                              superseded_by: str,
                              keep_batch_id: Optional[str] = None
                              ) -> List[OutboxItem]:
        """Cancel not-yet-sent items for a reporting position that was re-stated.

        A corrected upload that arrives before the original was delivered must
        not produce two messages — the reader would see a stale figure followed
        by a correction of something they never saw. The undelivered original
        is cancelled instead.
        """
        changed: List[OutboxItem] = []
        for item in self.list(tenant_id, states=list(CLAIMABLE)):
            if item.reporting_key != reporting_key:
                continue
            if keep_batch_id and item.notification_batch_id == keep_batch_id:
                continue
            item.state = STATE_SUPERSEDED
            item.superseded_by = superseded_by
            changed.append(self._write(item))
        return changed

    def mark_corrected(self, tenant_id: str, reporting_key: str, *,
                       corrected_by: str,
                       keep_batch_id: Optional[str] = None) -> List[OutboxItem]:
        """Record that DELIVERED items for a reporting position were corrected.

        The delivered message is not edited — v1 sends a clearly labelled
        correction instead — but the audit trail records which message the
        correction replaced.
        """
        changed: List[OutboxItem] = []
        for item in self.list(tenant_id, states=[STATE_SENT]):
            if item.reporting_key != reporting_key:
                continue
            if keep_batch_id and item.notification_batch_id == keep_batch_id:
                continue
            item.state = STATE_CORRECTED
            item.corrected_by = corrected_by
            changed.append(self._write(item))
        return changed

    # -- operator view ----------------------------------------------------- #
    def due(self, tenant_id: str, *, now: Optional[str] = None
            ) -> List[OutboxItem]:
        """Claimable items whose backoff has elapsed, in delivery order."""
        moment = now or _now()
        out = [i for i in self.list(tenant_id, states=list(CLAIMABLE))
               if not i.next_attempt_at or i.next_attempt_at <= moment]
        return out

    def failures(self, tenant_id: str) -> List[OutboxItem]:
        """What an operator still has to act on."""
        return self.list(tenant_id, states=[STATE_RETRYABLE, STATE_PERMANENT])


def _redact(error: str) -> str:
    """Trim and de-identify an error before it is persisted.

    Errors from a transport layer routinely contain a bearer token or a signed
    service URL. Truncating is not enough on its own, so anything that looks
    like a credential is dropped before the string is stored.
    """
    text = str(error or "").strip()
    lowered = text.lower()
    for marker in ("bearer ", "authorization", "access_token", "password",
                   "client_secret", "sig="):
        idx = lowered.find(marker)
        if idx >= 0:
            text = text[:idx] + "[redacted]"
            break
    return text[:500]
