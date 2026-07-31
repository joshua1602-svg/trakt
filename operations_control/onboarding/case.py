"""operations_control.onboarding.case — the governed onboarding case.

A case is the unit of work: one attempt to bring a client into Trakt, or to
change what Trakt already holds about one. It carries a system-generated
reference, a status, the answers gathered so far, the information still
outstanding from the client, and its own event history.

Three kinds share the model, because they gather the same information and
generate the same artefacts. What differs is where the answers start:

  ``new_client``  blank. The primary product.
  ``migration``   pre-populated from a legacy client's existing configuration.
  ``amendment``   pre-populated from the client's active approved version.

Only the entry point differs — never the model, the validation or the
generation. That is the point: a migrated client and a newly onboarded one are
the same client afterwards.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from ..contracts import new_id, now_iso

# -- kinds ------------------------------------------------------------------ #
KIND_NEW_CLIENT = "new_client"
KIND_MIGRATION = "migration"
KIND_AMENDMENT = "amendment"
KINDS = (KIND_NEW_CLIENT, KIND_MIGRATION, KIND_AMENDMENT)

KIND_LABELS = {
    KIND_NEW_CLIENT: "New client",
    KIND_MIGRATION: "Legacy migration",
    KIND_AMENDMENT: "Amendment",
}

# -- statuses --------------------------------------------------------------- #
DRAFT = "draft"
INFORMATION_REQUESTED = "information_requested"
AWAITING_CLIENT = "awaiting_client"
IN_REVIEW = "in_review"
CHANGES_REQUIRED = "changes_required"
READY_FOR_APPROVAL = "ready_for_approval"
APPROVED = "approved"
ACTIVATED = "activated"
WITHDRAWN = "withdrawn"

STATUSES = (DRAFT, INFORMATION_REQUESTED, AWAITING_CLIENT, IN_REVIEW,
            CHANGES_REQUIRED, READY_FOR_APPROVAL, APPROVED, ACTIVATED,
            WITHDRAWN)

#: A case that has reached one of these is finished; nothing further is written.
TERMINAL = (ACTIVATED, WITHDRAWN)

#: Statuses in which the case still writes NO active configuration. Everything
#: except ACTIVATED — approval records the decision, activation performs it.
NO_ACTIVE_CONFIGURATION = tuple(s for s in STATUSES if s != ACTIVATED)

STATUS_LABELS = {
    DRAFT: "Draft",
    INFORMATION_REQUESTED: "Information requested",
    AWAITING_CLIENT: "Awaiting client",
    IN_REVIEW: "In review",
    CHANGES_REQUIRED: "Changes required",
    READY_FOR_APPROVAL: "Ready for approval",
    APPROVED: "Approved",
    ACTIVATED: "Activated",
    WITHDRAWN: "Withdrawn",
}

#: Legal transitions. Enforced by the service, not by the browser.
TRANSITIONS: Dict[str, tuple] = {
    DRAFT: (INFORMATION_REQUESTED, IN_REVIEW, READY_FOR_APPROVAL, WITHDRAWN),
    INFORMATION_REQUESTED: (AWAITING_CLIENT, DRAFT, IN_REVIEW, WITHDRAWN),
    AWAITING_CLIENT: (IN_REVIEW, DRAFT, CHANGES_REQUIRED, WITHDRAWN),
    IN_REVIEW: (CHANGES_REQUIRED, READY_FOR_APPROVAL, DRAFT, WITHDRAWN),
    CHANGES_REQUIRED: (DRAFT, IN_REVIEW, INFORMATION_REQUESTED, WITHDRAWN),
    READY_FOR_APPROVAL: (APPROVED, CHANGES_REQUIRED, DRAFT, WITHDRAWN),
    APPROVED: (ACTIVATED, CHANGES_REQUIRED, WITHDRAWN),
    ACTIVATED: (),
    WITHDRAWN: (),
}

# -- information requests --------------------------------------------------- #
REQUEST_OPEN = "open"
REQUEST_SENT = "sent"
REQUEST_ANSWERED = "answered"
REQUEST_ACCEPTED = "accepted"
REQUEST_REJECTED = "rejected"
REQUEST_STATUSES = (REQUEST_OPEN, REQUEST_SENT, REQUEST_ANSWERED,
                    REQUEST_ACCEPTED, REQUEST_REJECTED)

CASE_ID_RE = re.compile(r"^ONB-\d{4}-\d{4}$")


class CaseError(Exception):
    """An operator-safe refusal from the case model."""

    def __init__(self, code: str, message: str, http_status: int = 400):
        self.code, self.message, self.http_status = code, message, http_status
        super().__init__(f"{code}: {message}")


def mint_case_id(year: int, sequence: int) -> str:
    """``ONB-2026-0001``. Sortable, quotable on a phone call, and obviously not
    a client identifier."""
    return f"ONB-{year:04d}-{sequence:04d}"


@dataclass
class InformationRequest:
    """One request for information from a party outside the OCC.

    The OCC does not need a client portal to be useful here: what it needs is a
    governed record of what was asked, of whom, when, what came back, and who
    accepted it. A portal can post into the same record later.
    """

    request_id: str
    #: Catalogue field references: [{"section", "field", "index", "label"}]
    items: List[Dict[str, Any]] = field(default_factory=list)
    responsible_party: str = "client"        # client | operator | third_party
    status: str = REQUEST_OPEN
    requested_by: str = ""
    requested_at: str = ""
    sent_at: str = ""
    due_date: str = ""
    response_note: str = ""
    responded_at: str = ""
    responded_by: str = ""
    #: [{"name", "reference", "received_at", "received_by"}] — a reference to
    #: evidence, never the document itself.
    evidence: List[Dict[str, str]] = field(default_factory=list)
    reviewed_by: str = ""
    reviewed_at: str = ""
    review_note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, doc: Dict[str, Any]) -> "InformationRequest":
        known = {f for f in cls.__dataclass_fields__}   # type: ignore[attr-defined]
        return cls(**{k: v for k, v in (doc or {}).items() if k in known})

    @property
    def outstanding(self) -> bool:
        return self.status in (REQUEST_OPEN, REQUEST_SENT)


@dataclass
class OnboardingCase:
    """One governed onboarding, migration or amendment."""

    case_id: str
    kind: str = KIND_NEW_CLIENT
    status: str = DRAFT
    #: Empty until the client is identified. A case is opened before anyone
    #: knows what the client will be called.
    client_id: str = ""
    client_name: str = ""
    #: Section key -> answers. Repeatable sections hold a list.
    answers: Dict[str, Any] = field(default_factory=dict)
    information_requests: List[Dict[str, Any]] = field(default_factory=list)
    #: Free-text questions the operator has not turned into a formal request.
    open_questions: List[Dict[str, Any]] = field(default_factory=list)
    #: Where a migrated or amended case's answers came from, per field path.
    #: Human-readable, and shown as written ("the durable source
    #: registrations", "the currency used in GB").
    provenance: Dict[str, str] = field(default_factory=dict)
    #: The same question, answered as a CATEGORY rather than a sentence:
    #: ``human_supplied`` / ``client_supplied`` / ``artefact_derived`` /
    #: ``agent_proposed`` / ``human_approved`` / ``inherited_configuration``.
    #: Kept beside :attr:`provenance` rather than replacing it, because the two
    #: answer different needs — one is read by a person, the other by a control
    #: — and collapsing them would make the sentence unwritable or the category
    #: unusable. Sparse: only paths something has classified appear.
    provenance_class: Dict[str, str] = field(default_factory=dict)
    #: For amendments and migrations: the version this case started from.
    based_on_version: Optional[int] = None
    #: For migrations: the legacy documents, so unmanaged blocks survive.
    base_documents: Dict[str, Any] = field(default_factory=dict)
    #: Appended on every state change: who, when, what, before, after, why.
    events: List[Dict[str, Any]] = field(default_factory=list)
    created_by: str = ""
    created_at: str = ""
    updated_by: str = ""
    updated_at: str = ""
    approved_by: str = ""
    approved_at: str = ""
    approval_reason: str = ""
    activated_by: str = ""
    activated_at: str = ""
    withdrawn_by: str = ""
    withdrawn_at: str = ""
    withdrawal_reason: str = ""
    #: Set at activation: the version the approved answers became.
    activated_version: Optional[int] = None

    # -- serialisation ----------------------------------------------------- #
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, doc: Dict[str, Any]) -> "OnboardingCase":
        known = {f for f in cls.__dataclass_fields__}   # type: ignore[attr-defined]
        return cls(**{k: v for k, v in (doc or {}).items() if k in known})

    # -- answers ------------------------------------------------------------ #
    def block(self, section_key: str) -> Dict[str, Any]:
        value = self.answers.get(section_key)
        if not isinstance(value, dict):
            value = {}
            self.answers[section_key] = value
        return value

    def items(self, section_key: str) -> List[Dict[str, Any]]:
        value = self.answers.get(section_key)
        if not isinstance(value, list):
            value = []
            self.answers[section_key] = value
        return value

    @property
    def products(self) -> List[str]:
        return list((self.answers.get("reporting") or {}).get("products") or [])

    def entity(self, entity_id: str) -> Optional[Dict[str, Any]]:
        return next((e for e in self.items("entities")
                     if e.get("entity_id") == entity_id), None)

    def entities_with_role(self, role: str) -> List[Dict[str, Any]]:
        return [e for e in self.items("entities")
                if role in (e.get("roles") or [])]

    def portfolio(self, portfolio_id: str) -> Optional[Dict[str, Any]]:
        return next((p for p in self.items("portfolios")
                     if p.get("portfolio_id") == portfolio_id), None)

    # -- information requests ---------------------------------------------- #
    def requests(self) -> List[InformationRequest]:
        return [InformationRequest.from_dict(r)
                for r in self.information_requests]

    def request(self, request_id: str) -> Optional[InformationRequest]:
        return next((r for r in self.requests() if r.request_id == request_id),
                    None)

    def save_request(self, request: InformationRequest) -> None:
        doc = request.to_dict()
        for i, existing in enumerate(self.information_requests):
            if existing.get("request_id") == request.request_id:
                self.information_requests[i] = doc
                return
        self.information_requests.append(doc)

    @property
    def outstanding_requests(self) -> List[InformationRequest]:
        return [r for r in self.requests() if r.outstanding]

    @property
    def unresolved_questions(self) -> List[Dict[str, Any]]:
        return [q for q in self.open_questions if not q.get("resolved_at")]

    # -- lifecycle ---------------------------------------------------------- #
    def can_transition(self, to_status: str) -> bool:
        return to_status in TRANSITIONS.get(self.status, ())

    def transition(self, to_status: str, *, actor: str, reason: str = "",
                   detail: Optional[Dict[str, Any]] = None) -> None:
        if to_status == self.status:
            return
        if not self.can_transition(to_status):
            raise CaseError(
                "OPS_ONBOARDING_BAD_TRANSITION",
                f"This onboarding cannot move from "
                f"{STATUS_LABELS.get(self.status, self.status).lower()} to "
                f"{STATUS_LABELS.get(to_status, to_status).lower()}.", 409)
        before = self.status
        self.status = to_status
        self.record(f"status_{to_status}", actor=actor, reason=reason,
                    before={"status": before}, after={"status": to_status},
                    detail=detail or {})

    def record(self, event: str, *, actor: str, reason: str = "",
               before: Optional[Dict[str, Any]] = None,
               after: Optional[Dict[str, Any]] = None,
               detail: Optional[Dict[str, Any]] = None) -> None:
        """Append to the case's own history. The hash-chained client audit trail
        is written separately by the service; this is the case-local narrative
        an operator reads on screen."""
        self.events.append({
            "event": event, "actor": actor, "at": now_iso(),
            "reason": reason, "before": before or {}, "after": after or {},
            "detail": detail or {}})
        self.updated_by = actor
        self.updated_at = now_iso()

    @property
    def writes_active_configuration(self) -> bool:
        return self.status == ACTIVATED


def new_entity_id() -> str:
    return new_id("ent")


def new_request_id() -> str:
    return new_id("req")


def source_key(portfolio_id: str, dataset: str) -> str:
    """One source registration per portfolio and dataset. Cadence is NOT part of
    the key: the existing registry matches on client/portfolio/dataset, because
    a book registered at one cadence is commonly reported at another."""
    return f"{portfolio_id}/{dataset}"
