#!/usr/bin/env python3
"""Who may receive a proactive message, and where it goes.

A proactive Teams message has no request to authorise against: nobody asked for
it, so there is no caller whose entitlement can be checked at send time. The
authorisation has to have happened earlier and be recorded durably, which is
what this module holds.

The record is built from facts the platform asserts, never from facts a caller
supplies:

* the **Microsoft tenant id** and **Entra object id** come from the validated
  Bot Framework activity, not from a body field;
* the **Teams conversation reference** and **service URL** come from the same
  activity — they are the only way to address a personal chat, and they cannot
  be constructed;
* the **Trakt tenant** and the **authorised portfolio contexts** come from an
  operator mapping, never from the user and never inferred from an email
  domain.

Display names and email addresses are stored for the operator's benefit and are
never used to make a decision. A display name is user-controlled in most
directories; treating one as an identity would be an authorisation bug.

Delivery requires ALL of: an enabled record, a captured conversation reference,
an installed app, a matching Microsoft tenant, and the portfolio context in the
authorised list. Any one missing fails closed.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from apps.blob_trigger_app.storage import Storage

from .layout import NotificationLayout

logger = logging.getLogger("trakt_notifications.recipients")

# Installation state of the Trakt app for this user.
INSTALL_INSTALLED = "installed"
INSTALL_REMOVED = "removed"
INSTALL_UNKNOWN = "unknown"

DELIVERY_PERSONAL = "personal"


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat()


def recipient_id(microsoft_tenant_id: str, entra_object_id: str) -> str:
    """Stable id for one person in one Microsoft tenant.

    Hashed rather than concatenated so the storage key carries no directory
    identifier, and derived from the tenant AND the object id so the same
    object id in two tenants can never collide onto one record.
    """
    raw = f"{(microsoft_tenant_id or '').lower()}|{(entra_object_id or '').lower()}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


@dataclass
class Recipient:
    """One durable notification destination."""

    tenant_id: str                       # Trakt tenant — governs the data
    microsoft_tenant_id: str             # Entra directory the user signs in to
    entra_object_id: str                 # the user's stable object id
    conversation_id: Optional[str] = None
    service_url: Optional[str] = None
    #: The Bot Framework conversation reference, stored verbatim so a proactive
    #: send can restore the conversation without reconstructing it.
    conversation_reference: Dict[str, Any] = field(default_factory=dict)
    #: Operator-approved portfolio contexts. Empty means nothing is authorised.
    portfolio_contexts: List[str] = field(default_factory=list)
    installation_status: str = INSTALL_UNKNOWN
    notifications_enabled: bool = False
    delivery_preference: str = DELIVERY_PERSONAL
    #: Operator-facing only. Never used in an authorisation decision.
    display_name: Optional[str] = None
    user_principal_name: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    #: Set by an operator to record who approved the portfolio mapping.
    authorised_by: Optional[str] = None

    @property
    def recipient_id(self) -> str:
        return recipient_id(self.microsoft_tenant_id, self.entra_object_id)

    @property
    def addressable(self) -> bool:
        """Whether a message could physically be delivered to this record."""
        return bool(self.conversation_id and self.service_url)

    def authorised_for(self, portfolio_context: str) -> bool:
        return str(portfolio_context or "total") in set(self.portfolio_contexts)

    def to_dict(self) -> Dict[str, Any]:
        out = asdict(self)
        out["recipient_id"] = self.recipient_id
        out["addressable"] = self.addressable
        return out

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Recipient":
        known = {f for f in cls.__dataclass_fields__}  # noqa: SLF001
        return cls(**{k: v for k, v in (data or {}).items() if k in known})


@dataclass
class Refusal:
    """Why a candidate recipient was not delivered to.

    Returned rather than logged-and-dropped so the reason reaches the outbox
    audit trail and the operator diagnostic view. A recipient who quietly
    receives nothing is indistinguishable from a broken pipeline.
    """

    recipient_id: str
    reason: str
    code: str


# Refusal codes, so an operator view can group them.
REFUSE_DISABLED = "notifications_disabled"
REFUSE_NOT_ADDRESSABLE = "no_conversation_reference"
REFUSE_NOT_INSTALLED = "app_not_installed"
REFUSE_UNAUTHORISED_PORTFOLIO = "portfolio_not_authorised"
REFUSE_TENANT_MISMATCH = "tenant_mismatch"


class RecipientStore:
    """Durable recipient records on the existing storage abstraction."""

    def __init__(self, storage: Storage,
                 layout: Optional[NotificationLayout] = None):
        self.storage = storage
        self.layout = layout or NotificationLayout.from_env()

    def save(self, recipient: Recipient) -> Recipient:
        recipient.updated_at = _now()
        recipient.created_at = recipient.created_at or recipient.updated_at
        uri = self.layout.recipient_uri(recipient.tenant_id,
                                        recipient.recipient_id)
        self.storage.write_text(uri, json.dumps(recipient.to_dict(), indent=2,
                                                default=str))
        return recipient

    def load(self, tenant_id: str, rid: str) -> Optional[Recipient]:
        uri = self.layout.recipient_uri(tenant_id, rid)
        if not self.storage.exists(uri):
            return None
        try:
            return Recipient.from_dict(json.loads(self.storage.read_text(uri)))
        except Exception as exc:  # noqa: BLE001
            logger.warning("notifications: unreadable recipient %s (%s)", rid, exc)
            return None

    def list(self, tenant_id: str) -> List[Recipient]:
        """Every recipient record for ONE tenant.

        The tenant is a path segment, so this listing physically cannot
        enumerate another tenant's recipients.
        """
        out: List[Recipient] = []
        for uri in self.storage.list(self.layout.recipients_prefix(tenant_id)):
            if not uri.endswith(".json"):
                continue
            try:
                out.append(Recipient.from_dict(
                    json.loads(self.storage.read_text(uri))))
            except Exception:  # noqa: BLE001 - one bad record, not the send
                continue
        return sorted(out, key=lambda r: r.recipient_id)

    # -- registration ------------------------------------------------------ #
    def capture_conversation(self, *, tenant_id: str, microsoft_tenant_id: str,
                             entra_object_id: str, conversation_id: str,
                             service_url: str,
                             conversation_reference: Dict[str, Any],
                             display_name: Optional[str] = None,
                             user_principal_name: Optional[str] = None
                             ) -> Recipient:
        """Record the conversation reference captured from a bot activity.

        Capturing a conversation makes a user *addressable*; it does not make
        them *authorised*. A newly captured record has no portfolio contexts
        and notifications disabled, so installing the app can never, by itself,
        start a feed of portfolio data to whoever installed it.
        """
        rid = recipient_id(microsoft_tenant_id, entra_object_id)
        existing = self.load(tenant_id, rid)
        if existing is None:
            existing = Recipient(
                tenant_id=tenant_id,
                microsoft_tenant_id=microsoft_tenant_id,
                entra_object_id=entra_object_id)
        elif existing.microsoft_tenant_id.lower() != microsoft_tenant_id.lower():
            # The id is derived from the tenant, so this should be
            # unreachable; refuse rather than overwrite if it ever is.
            raise ValueError("recipient record does not match the asserted "
                             "Microsoft tenant")
        existing.conversation_id = conversation_id
        existing.service_url = service_url
        existing.conversation_reference = dict(conversation_reference or {})
        existing.installation_status = INSTALL_INSTALLED
        existing.display_name = display_name or existing.display_name
        existing.user_principal_name = (user_principal_name
                                        or existing.user_principal_name)
        return self.save(existing)

    def mark_uninstalled(self, tenant_id: str, rid: str) -> Optional[Recipient]:
        """Record an app removal. Disables delivery; keeps the audit record."""
        recipient = self.load(tenant_id, rid)
        if recipient is None:
            return None
        recipient.installation_status = INSTALL_REMOVED
        recipient.notifications_enabled = False
        return self.save(recipient)

    def authorise(self, tenant_id: str, rid: str, *,
                  portfolio_contexts: List[str], actor: str,
                  enable: bool = True) -> Recipient:
        """The operator step that turns an addressable user into a recipient."""
        recipient = self.load(tenant_id, rid)
        if recipient is None:
            raise KeyError(f"no such recipient: {rid}")
        recipient.portfolio_contexts = sorted({str(p) for p in portfolio_contexts
                                               if str(p).strip()})
        recipient.notifications_enabled = bool(enable and
                                               recipient.portfolio_contexts)
        recipient.authorised_by = actor
        return self.save(recipient)


def select(store: RecipientStore, *, tenant_id: str, portfolio_context: str,
           expected_microsoft_tenant: Optional[str] = None
           ) -> "tuple[List[Recipient], List[Refusal]]":
    """The recipients a batch may be delivered to, and why the others were not.

    Every gate is applied explicitly and its failure is reported. The order
    matters for the operator reading the result: the cheapest, most common
    reason (notifications simply not enabled) is checked first so a diagnostic
    view is not dominated by incidental failures.
    """
    eligible: List[Recipient] = []
    refused: List[Refusal] = []

    for recipient in store.list(tenant_id):
        rid = recipient.recipient_id
        if not recipient.notifications_enabled:
            refused.append(Refusal(rid, "notifications are not enabled for "
                                        "this recipient", REFUSE_DISABLED))
            continue
        if recipient.installation_status == INSTALL_REMOVED:
            refused.append(Refusal(rid, "the Trakt app has been removed for "
                                        "this recipient", REFUSE_NOT_INSTALLED))
            continue
        if not recipient.addressable:
            refused.append(Refusal(rid, "no Teams conversation reference has "
                                        "been captured", REFUSE_NOT_ADDRESSABLE))
            continue
        if not recipient.authorised_for(portfolio_context):
            refused.append(Refusal(
                rid, f"not authorised for portfolio context "
                     f"{portfolio_context!r}", REFUSE_UNAUTHORISED_PORTFOLIO))
            continue
        if (expected_microsoft_tenant
                and recipient.microsoft_tenant_id.lower()
                != str(expected_microsoft_tenant).lower()):
            # A record whose Microsoft tenant does not match the tenant this
            # deployment serves is never re-routed — it is refused.
            refused.append(Refusal(rid, "recipient is in a different Microsoft "
                                        "tenant", REFUSE_TENANT_MISMATCH))
            continue
        eligible.append(recipient)

    return eligible, refused
