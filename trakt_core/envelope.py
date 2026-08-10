"""trakt_core.envelope — the governed result envelope.

Every governed capability returns a :class:`GovernedResult`: one stable,
interface-neutral object carrying the answer plus the governance facts a caller
needs to trust it. HTTP adapters serialise it; Python callers consume it
directly; neither has to reconstruct any of it.

Compatibility note — the React and Copilot payloads are unchanged. The existing
response dictionary is carried verbatim in :attr:`GovernedResult.result`, and the
adapters return that dictionary with the governance block added additively (see
``mi_agent_api.presenters``). No existing field was renamed or removed.

Provenance is a *reference*, not a second model: :class:`ProvenanceRef` points at
the structures the MI engine already produces (``sourceNotes``, the
reconciliation footer) and at the dataset snapshot. Row-level provenance stays
where it is — in ``engine.provenance`` and the canonical data itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Generic, List, Optional, Tuple, TypeVar

from .errors import TraktError

T = TypeVar("T")

#: Envelope schema version. Bump the minor when adding a field; bump the major
#: only for a breaking change (and then version the route).
SCHEMA_VERSION = "1.1.0"  # 1.1: additive ScopeRef (portfolio scope + coverage)

STATUS_SUCCESS = "success"
STATUS_PARTIAL_SUCCESS = "partial_success"
STATUS_BLOCKED = "blocked"       # governance refused: policy / authorisation
STATUS_ERROR = "error"           # the capability could not produce an answer


@dataclass(frozen=True)
class SnapshotRef:
    """Identity of the data the answer was computed from."""

    source_kind: Optional[str] = None        # display kind (data_source.KIND_*)
    source_base: Optional[str] = None        # resolution base (policy.BASE_*)
    dataset_label: Optional[str] = None      # file name — never a full path
    reporting_date: Optional[str] = None
    snapshot_id: Optional[str] = None
    content_hash: Optional[str] = None
    row_count: Optional[int] = None
    approval_state: Optional[str] = None
    source_portfolios: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_kind": self.source_kind,
            "source_base": self.source_base,
            "dataset_label": self.dataset_label,
            "reporting_date": self.reporting_date,
            "snapshot_id": self.snapshot_id,
            "content_hash": self.content_hash,
            "row_count": self.row_count,
            "approval_state": self.approval_state,
            "source_portfolios": list(self.source_portfolios),
        }


@dataclass(frozen=True)
class ScopeRef:
    """Which portfolios the answer covers, and which of them actually answered.

    Produced by the governed backend from
    :class:`trakt_core.portfolio.ScopeCoverage`; every channel renders the same
    facts from it. A channel must never derive consolidation status, exclusions
    or field availability for itself — that is precisely the divergence this
    reference exists to prevent.
    """

    context_id: Optional[str] = None
    context_kind: Optional[str] = None
    label: Optional[str] = None
    portfolios_in_scope: Tuple[str, ...] = ()
    portfolios_used: Tuple[str, ...] = ()
    portfolios_excluded: Dict[str, Dict[str, str]] = field(default_factory=dict)
    is_fully_consolidated: bool = True
    field_coverage: Dict[str, Dict[str, List[str]]] = field(default_factory=dict)
    disclosure: Optional[str] = None

    @classmethod
    def from_coverage(cls, coverage: Any) -> "ScopeRef":
        """Build from a :class:`trakt_core.portfolio.ScopeCoverage` instance."""
        return cls.from_dict(coverage.to_dict())

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> Optional["ScopeRef"]:
        """Build from the serialised coverage block the MI engine produces."""
        if not data:
            return None
        requested = data.get("scope_requested") or {}
        return cls(
            context_id=requested.get("context_id"),
            context_kind=requested.get("context_kind"),
            label=requested.get("label"),
            portfolios_in_scope=tuple(data.get("portfolios_in_scope") or ()),
            portfolios_used=tuple(data.get("portfolios_used") or ()),
            portfolios_excluded=dict(data.get("portfolios_excluded") or {}),
            is_fully_consolidated=bool(data.get("is_fully_consolidated")),
            field_coverage=dict(data.get("field_coverage") or {}),
            disclosure=data.get("disclosure"))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scope_requested": {"context_id": self.context_id,
                                "context_kind": self.context_kind,
                                "label": self.label},
            "portfolios_in_scope": list(self.portfolios_in_scope),
            "portfolios_used": list(self.portfolios_used),
            "portfolios_excluded": {k: dict(v) for k, v in self.portfolios_excluded.items()},
            "is_fully_consolidated": self.is_fully_consolidated,
            "field_coverage": {k: dict(v) for k, v in self.field_coverage.items()},
            "disclosure": self.disclosure,
        }


@dataclass(frozen=True)
class PolicyState:
    """What governance decided, so a caller can show or log it."""

    data_approved: bool = False
    tenant_authorised: bool = False
    capability_allowed: bool = False
    runtime_mode: Optional[str] = None
    #: True when a non-production fixture source was permitted by the runtime
    #: mode. Always False in production.
    fixture_source: bool = False
    notes: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "data_approved": self.data_approved,
            "tenant_authorised": self.tenant_authorised,
            "capability_allowed": self.capability_allowed,
            "runtime_mode": self.runtime_mode,
            "fixture_source": self.fixture_source,
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class ProvenanceRef:
    """Reference to the provenance the capability already produced."""

    #: The MI engine's ``sourceNotes`` entries, unchanged.
    source_notes: Tuple[Dict[str, Any], ...] = ()
    #: The reconciliation / coverage footer, unchanged.
    reconciliation: Optional[Dict[str, Any]] = None
    #: The dataset the answer came from.
    snapshot: Optional[SnapshotRef] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_notes": [dict(n) for n in self.source_notes],
            "reconciliation": self.reconciliation,
            "snapshot": self.snapshot.to_dict() if self.snapshot else None,
        }


@dataclass(frozen=True)
class AuditMetadata:
    """Lightweight execution metadata. Never carries the answer body."""

    capability: str
    request_id: str
    tenant_id: str
    actor_id: str
    actor_type: str
    channel: str
    outcome: str
    started_at: str
    duration_ms: Optional[int] = None
    correlation_id: Optional[str] = None
    portfolio_id: Optional[str] = None
    snapshot_id: Optional[str] = None
    error_code: Optional[str] = None
    #: WHO ASKED, where the channel established it — see
    #: :mod:`trakt_core.organisation`. Optional and defaulted so every existing
    #: construction site is unchanged; ``None`` on the single-client deployment
    #: and on channels that carry no directory identity.
    organisation_id: Optional[str] = None
    microsoft_tenant_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "capability": self.capability,
            "request_id": self.request_id,
            "correlation_id": self.correlation_id,
            "tenant_id": self.tenant_id,
            "organisation_id": self.organisation_id,
            "microsoft_tenant_id": self.microsoft_tenant_id,
            "actor_id": self.actor_id,
            "actor_type": self.actor_type,
            "channel": self.channel,
            "portfolio_id": self.portfolio_id,
            "snapshot_id": self.snapshot_id,
            "outcome": self.outcome,
            "started_at": self.started_at,
            "duration_ms": self.duration_ms,
            "error_code": self.error_code,
        }


@dataclass(frozen=True)
class GovernedResult(Generic[T]):
    """The stable return type of every governed capability."""

    capability: str
    status: str
    request_id: str
    tenant_id: str
    schema_version: str = SCHEMA_VERSION
    correlation_id: Optional[str] = None
    portfolio_id: Optional[str] = None
    snapshot: Optional[SnapshotRef] = None
    result: Optional[T] = None
    warnings: Tuple[str, ...] = ()
    policy: PolicyState = field(default_factory=PolicyState)
    #: Portfolio scope + coverage for this answer (``None`` when the capability
    #: is not portfolio-scoped). Additive: existing consumers ignore it.
    scope: Optional[ScopeRef] = None
    provenance: Optional[ProvenanceRef] = None
    audit: Optional[AuditMetadata] = None
    error: Optional[TraktError] = None

    # -- predicates --------------------------------------------------------- #
    @property
    def ok(self) -> bool:
        return self.status in (STATUS_SUCCESS, STATUS_PARTIAL_SUCCESS)

    @property
    def blocked(self) -> bool:
        return self.status == STATUS_BLOCKED

    @property
    def error_code(self) -> Optional[str]:
        return self.error.code if self.error else None

    @property
    def http_status(self) -> int:
        """The HTTP status an adapter should return for this result."""
        if self.error is None:
            return 200
        return self.error.http_status

    # -- serialisation ------------------------------------------------------ #
    def governance_block(self) -> Dict[str, Any]:
        """The additive governance metadata adapters attach to their payloads."""
        return {
            "capability": self.capability,
            "schemaVersion": self.schema_version,
            "status": self.status,
            "requestId": self.request_id,
            "correlationId": self.correlation_id,
            "tenantId": self.tenant_id,
            "portfolioId": self.portfolio_id,
            "snapshot": self.snapshot.to_dict() if self.snapshot else None,
            "policy": self.policy.to_dict(),
            "scope": self.scope.to_dict() if self.scope else None,
            "error": self.error.to_dict() if self.error else None,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Full JSON-serialisable form. Contains no interface-specific types."""
        return {
            "capability": self.capability,
            "schema_version": self.schema_version,
            "status": self.status,
            "request_id": self.request_id,
            "correlation_id": self.correlation_id,
            "tenant_id": self.tenant_id,
            "portfolio_id": self.portfolio_id,
            "snapshot": self.snapshot.to_dict() if self.snapshot else None,
            "result": self.result,
            "warnings": list(self.warnings),
            "policy": self.policy.to_dict(),
            "scope": self.scope.to_dict() if self.scope else None,
            "provenance": self.provenance.to_dict() if self.provenance else None,
            "audit": self.audit.to_dict() if self.audit else None,
            "error": self.error.to_dict() if self.error else None,
        }
