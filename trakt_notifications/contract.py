#!/usr/bin/env python3
"""The versioned notification envelope — one batch, exactly two messages.

Designed so a rendered card, an audit record and a replayed batch can never
disagree with the Weekly Portfolio Brief they came from. Three properties carry
that:

* **Deterministic batch identity.** ``notification_batch_id`` hashes what the
  batch IS — contract version, tenant, portfolio context, update type, the
  approved run ids and the source dates. It deliberately excludes
  ``generated_at`` and every value, so regenerating the same approved update
  produces the same id and the outbox can de-duplicate on it. Re-running the
  same approved run therefore cannot send twice.

* **A separate reporting key.** ``reporting_key`` hashes the same identity
  WITHOUT the run ids, so a corrected upload for a reporting date that has
  already been notified is recognisable as a correction of a specific earlier
  batch rather than as an unrelated new one.

* **Self-contained payloads.** A message carries its own headline, its own
  items with their own pre-formatted display values, its own deep link, the
  insight ids it was selected from and the methodology versions behind it. A
  renderer never re-queries and never recalculates — that is the whole point of
  the object.

The severity vocabulary is the Insight Engine's (``info`` / ``attention`` /
``concern``) plus one state the Insight Engine has no need for: ``clear``, the
explicit "nothing material was found" outcome a Risk Review must be able to
state. Nothing here re-grades an insight's severity; a message's severity is
the maximum of the severities of the governed insights it selected.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

#: Bumped when the SHAPE of the envelope changes. Metric methodologies carry
#: their own versions inside ``methodology_versions``.
CONTRACT_VERSION = "1"

# --------------------------------------------------------------------------- #
# Update types — what was approved
# --------------------------------------------------------------------------- #
UPDATE_PIPELINE = "PIPELINE"
UPDATE_FUNDED = "FUNDED"
UPDATE_COMBINED = "COMBINED"

UPDATE_TYPES = (UPDATE_PIPELINE, UPDATE_FUNDED, UPDATE_COMBINED)

# --------------------------------------------------------------------------- #
# Message types — exactly two, and the list is closed on purpose
# --------------------------------------------------------------------------- #
MESSAGE_PORTFOLIO_UPDATE = "PORTFOLIO_UPDATE"
MESSAGE_RISK_REVIEW = "RISK_REVIEW"

MESSAGE_TYPES = (MESSAGE_PORTFOLIO_UPDATE, MESSAGE_RISK_REVIEW)

#: Delivery order within a batch. Operational information first, risk and
#: decision support second — the reader needs to know what changed before being
#: told what to consider about it.
MESSAGE_SEQUENCE = {MESSAGE_PORTFOLIO_UPDATE: 1, MESSAGE_RISK_REVIEW: 2}

# --------------------------------------------------------------------------- #
# Severity — the Insight Engine's vocabulary, plus an explicit clear state
# --------------------------------------------------------------------------- #
SEVERITY_CLEAR = "clear"
SEVERITY_INFO = "info"
SEVERITY_ATTENTION = "attention"
SEVERITY_CONCERN = "concern"

SEVERITY_RANK = {
    SEVERITY_CLEAR: 0,
    SEVERITY_INFO: 1,
    SEVERITY_ATTENTION: 2,
    SEVERITY_CONCERN: 3,
}

#: Adaptive Card container styles. Presentation only — a style never changes
#: what a message claims about itself.
SEVERITY_CARD_STYLE = {
    SEVERITY_CLEAR: "good",
    SEVERITY_INFO: "default",
    SEVERITY_ATTENTION: "warning",
    SEVERITY_CONCERN: "attention",
}


def max_severity(severities: List[Optional[str]]) -> str:
    """The highest severity present, defaulting to ``clear``.

    Used to derive a message's severity from the governed insights it selected.
    A message can therefore never be graded above the evidence behind it.
    """
    best = SEVERITY_CLEAR
    for s in severities:
        if s and SEVERITY_RANK.get(s, 0) > SEVERITY_RANK[best]:
            best = s
    return best


# --------------------------------------------------------------------------- #
# Message items
# --------------------------------------------------------------------------- #
@dataclass
class MessageItem:
    """One bullet on a card.

    ``label``/``value`` are ALREADY formatted for display by the generator that
    owns the metric's units. A renderer must not reformat them: doing so is how
    two channels start disagreeing about whether a number was in millions.

    ``metric`` and ``raw`` carry the machine-readable original so an audit or a
    later channel can read the value without parsing the display string.
    """

    text: str
    metric: Optional[str] = None
    raw: Optional[Any] = None
    #: Governed insight this item was taken from, where there is exactly one.
    insight_id: Optional[str] = None
    #: Set when the item exists to explain that something could NOT be shown.
    unavailable: bool = False

    def to_dict(self) -> Dict[str, Any]:
        out = {k: v for k, v in asdict(self).items()
               if v is not None or k == "text"}
        # ``unavailable`` is the exception a reader cares about, so it is
        # carried only when true rather than on every item as noise.
        if not self.unavailable:
            out.pop("unavailable", None)
        return out


@dataclass
class NotificationMessage:
    """One of the two messages in a batch."""

    message_type: str
    headline: str
    #: The one-sentence position the card leads with, above the bullets. Kept
    #: separate from ``headline`` because the headline is what a reader sees in
    #: a notification list and must stay short enough to survive truncation.
    summary: Optional[str] = None
    severity: str = SEVERITY_INFO
    items: List[MessageItem] = field(default_factory=list)
    deep_link: Optional[str] = None
    #: Stable ids of the governed insights this message was selected from.
    insight_ids: List[str] = field(default_factory=list)
    #: Human-readable label for the deep-link button.
    deep_link_label: str = "Open Trakt"
    #: Populated when a check the message would normally run was unavailable.
    #: A Risk Review must never say "no risks" on the back of a check that did
    #: not execute, so the two facts are carried separately.
    unavailable_checks: List[str] = field(default_factory=list)
    #: ``observation`` | ``consideration`` | ``approved_action`` — the highest
    #: action-language level used anywhere in this message.
    recommendation_level: Optional[str] = None
    #: Compact, non-identifying provenance shown as card footer text.
    provenance: Dict[str, Any] = field(default_factory=dict)

    @property
    def sequence(self) -> int:
        return MESSAGE_SEQUENCE.get(self.message_type, 99)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "message_type": self.message_type,
            "sequence": self.sequence,
            "severity": self.severity,
            "headline": self.headline,
            "summary": self.summary,
            "items": [i.to_dict() for i in self.items],
            "deep_link": self.deep_link,
            "deep_link_label": self.deep_link_label,
            "insight_ids": list(self.insight_ids),
            "unavailable_checks": list(self.unavailable_checks),
            "recommendation_level": self.recommendation_level,
            "provenance": dict(self.provenance),
        }


# --------------------------------------------------------------------------- #
# Batch
# --------------------------------------------------------------------------- #
def _hash(parts: List[str]) -> str:
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:24]


def _date_fingerprint(source_dates: Dict[str, Any]) -> str:
    """A stable rendering of the source dates, order-independent.

    ``json.dumps(sort_keys=True)`` rather than ``str(dict)`` so two batches
    built from dictionaries populated in a different order hash identically.
    """
    return json.dumps({k: v for k, v in sorted((source_dates or {}).items())},
                      sort_keys=True, default=str)


@dataclass
class NotificationBatch:
    """Everything produced by one approved update, for one portfolio context."""

    tenant_id: str
    portfolio_id: str
    portfolio_context: str
    update_type: str
    approved_run_ids: List[str] = field(default_factory=list)
    source_dates: Dict[str, Any] = field(default_factory=dict)
    messages: List[NotificationMessage] = field(default_factory=list)
    generated_at: Optional[str] = None
    methodology_versions: Dict[str, Any] = field(default_factory=dict)
    provenance: Dict[str, Any] = field(default_factory=dict)
    delivery_eligibility: Dict[str, Any] = field(default_factory=dict)
    #: Set when this batch supersedes an earlier one for the same reporting
    #: dates. Messages are then labelled as corrections.
    corrects_batch_id: Optional[str] = None
    version: str = CONTRACT_VERSION

    # -- identity ---------------------------------------------------------- #
    @property
    def notification_batch_id(self) -> str:
        """Deterministic across regenerations of the SAME approved update.

        Excludes ``generated_at`` and every value: two runs of the generator
        over one approved update must collide so the outbox de-duplicates.
        """
        return _hash([
            self.version, "batch",
            self.tenant_id or "-", self.portfolio_id or "-",
            self.portfolio_context or "total", self.update_type or "-",
            ",".join(sorted(str(r) for r in self.approved_run_ids)),
            _date_fingerprint(self.source_dates),
        ])

    @property
    def reporting_key(self) -> str:
        """Identity of the REPORTING POSITION, independent of which run produced it.

        Two batches sharing a reporting key describe the same reporting dates
        for the same portfolio: the second is a correction of the first, not a
        new event. The run ids are excluded for exactly that reason.
        """
        return _hash([
            self.version, "reporting",
            self.tenant_id or "-", self.portfolio_id or "-",
            self.portfolio_context or "total", self.update_type or "-",
            _date_fingerprint(self.source_dates),
        ])

    @property
    def is_correction(self) -> bool:
        return bool(self.corrects_batch_id)

    def message(self, message_type: str) -> Optional[NotificationMessage]:
        return next((m for m in self.messages
                     if m.message_type == message_type), None)

    def message_id(self, message_type: str) -> str:
        """Stable per-message identity, used as the outbox idempotency key."""
        return f"{self.notification_batch_id}.{message_type}"

    # -- serialisation ----------------------------------------------------- #
    def to_dict(self) -> Dict[str, Any]:
        return {
            "notification_batch_id": self.notification_batch_id,
            "reporting_key": self.reporting_key,
            "version": self.version,
            "tenant_id": self.tenant_id,
            "portfolio_id": self.portfolio_id,
            "portfolio_context": self.portfolio_context,
            "update_type": self.update_type,
            "approved_run_ids": list(self.approved_run_ids),
            "source_dates": dict(self.source_dates),
            "generated_at": self.generated_at,
            "corrects_batch_id": self.corrects_batch_id,
            "is_correction": self.is_correction,
            "messages": [m.to_dict() for m in
                         sorted(self.messages, key=lambda m: m.sequence)],
            "methodology_versions": dict(self.methodology_versions),
            "provenance": dict(self.provenance),
            "delivery_eligibility": dict(self.delivery_eligibility),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, default=str, sort_keys=False)


def validate(batch: NotificationBatch) -> List[str]:
    """Structural problems that must stop a batch reaching the outbox.

    Returned rather than raised: the trigger records them as a suppression
    reason with the rest of the eligibility evidence, so an operator can see
    why nothing was sent instead of finding a silent gap.
    """
    problems: List[str] = []
    if batch.update_type not in UPDATE_TYPES:
        problems.append(f"unknown update_type {batch.update_type!r}")
    if not batch.tenant_id:
        problems.append("no tenant_id — a batch may never be tenant-ambiguous")
    if not batch.portfolio_context:
        problems.append("no portfolio_context")
    present = sorted(m.message_type for m in batch.messages)
    if present != sorted(MESSAGE_TYPES):
        problems.append(
            f"a batch must carry exactly the two required messages, got {present}")
    for m in batch.messages:
        if m.severity not in SEVERITY_RANK:
            problems.append(f"{m.message_type}: unknown severity {m.severity!r}")
        if not m.headline:
            problems.append(f"{m.message_type}: no headline")
    return problems
