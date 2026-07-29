"""
engine.annex_delivery_agent.contracts
=====================================

The annex-neutral vocabulary of the Annex Delivery Agent.

Nothing in this module knows about Annex 2. It defines the three things every
annex shares — what an annex *is* (:class:`AnnexDefinition`), what a caller
*asks for* (:class:`DeliveryRequest`), and what the agent *returns*
(:class:`DeliveryResult`) — plus the single status vocabulary the lifecycle
moves through.

Governance is **not** re-modelled here. A :class:`DeliveryResult` is the
capability payload that gets carried inside the repository's existing
:class:`trakt_core.envelope.GovernedResult`; see
:mod:`engine.annex_delivery_agent.agent`.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

# --------------------------------------------------------------------------- #
# Status model
# --------------------------------------------------------------------------- #


class DeliveryStatus:
    """The lifecycle states, shared by every annex.

    Deliberately one vocabulary. Terminal-for-now states (``PREPARED``,
    ``PREPARED_WITH_WARNINGS``, ``AWAITING_APPROVAL``) are *not* published
    states: preparing an artefact and publishing it are separated by an explicit
    human decision, never by a schema result.
    """

    RECEIVED = "RECEIVED"
    PREFLIGHT_BLOCKED = "PREFLIGHT_BLOCKED"
    READY = "READY"
    NORMALISING = "NORMALISING"
    NORMALISATION_FAILED = "NORMALISATION_FAILED"
    BUILDING = "BUILDING"
    BUILD_FAILED = "BUILD_FAILED"
    VALIDATING = "VALIDATING"
    VALIDATION_FAILED = "VALIDATION_FAILED"
    PREPARED_WITH_WARNINGS = "PREPARED_WITH_WARNINGS"
    PREPARED = "PREPARED"
    AWAITING_APPROVAL = "AWAITING_APPROVAL"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    PUBLISHED = "PUBLISHED"


#: States from which no further automatic progress is possible without a human.
TERMINAL_FAILURE_STATUSES = frozenset({
    DeliveryStatus.PREFLIGHT_BLOCKED,
    DeliveryStatus.NORMALISATION_FAILED,
    DeliveryStatus.BUILD_FAILED,
    DeliveryStatus.VALIDATION_FAILED,
    DeliveryStatus.REJECTED,
})

#: States in which a schema-valid artefact exists on disk.
PREPARED_STATUSES = frozenset({
    DeliveryStatus.PREPARED,
    DeliveryStatus.PREPARED_WITH_WARNINGS,
    DeliveryStatus.AWAITING_APPROVAL,
    DeliveryStatus.APPROVED,
    DeliveryStatus.PUBLISHED,
})

#: The only state from which :mod:`~engine.annex_delivery_agent.publication`
#: will move an artefact to ``PUBLISHED``.
PUBLISHABLE_STATUS = DeliveryStatus.APPROVED


#: The lifecycle stages, in order. Used by the checkpoint store for restart.
STAGE_PREFLIGHT = "preflight"
STAGE_NORMALISE = "normalise"
STAGE_BUILD = "build"
STAGE_VALIDATE = "validate"

ORDERED_STAGES: Tuple[str, ...] = (
    STAGE_PREFLIGHT, STAGE_NORMALISE, STAGE_BUILD, STAGE_VALIDATE,
)


# --------------------------------------------------------------------------- #
# Approval
# --------------------------------------------------------------------------- #

#: Approval vocabulary, deliberately identical to
#: ``apps.blob_trigger_app.approvals`` so the operator console speaks one
#: language across source promotion and annex delivery.
APPROVAL_NOT_REQUESTED = "not_requested"
APPROVAL_PENDING = "pending"
APPROVAL_APPROVED = "approved"
APPROVAL_REJECTED = "rejected"
APPROVAL_PROMOTED = "promoted"


@dataclass(frozen=True)
class ApprovalState:
    """Where the human decision has got to. Never inferred from XSD validity."""

    state: str = APPROVAL_NOT_REQUESTED
    decided_by: Optional[str] = None
    decided_at: Optional[str] = None
    reason: Optional[str] = None
    requested_at: Optional[str] = None

    @property
    def approved(self) -> bool:
        return self.state in (APPROVAL_APPROVED, APPROVAL_PROMOTED)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "state": self.state,
            "decided_by": self.decided_by,
            "decided_at": self.decided_at,
            "reason": self.reason,
            "requested_at": self.requested_at,
        }

    @classmethod
    def from_dict(cls, data: Optional[Mapping[str, Any]]) -> "ApprovalState":
        d = dict(data or {})
        return cls(
            state=str(d.get("state") or APPROVAL_NOT_REQUESTED),
            decided_by=d.get("decided_by"),
            decided_at=d.get("decided_at"),
            reason=d.get("reason"),
            requested_at=d.get("requested_at"),
        )


# --------------------------------------------------------------------------- #
# Annex definition — the plug-in descriptor
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class AnnexDefinition:
    """Declarative identity of one annex report at one version.

    This is *description*, not behaviour: the behaviour lives in the provider
    (:class:`engine.annex_delivery_agent.providers.base.AnnexProvider`). Keeping
    them apart is what lets the operator console, the registry and the audit
    record describe an annex the agent has never run.

    ``authoritative_projector`` / ``normaliser`` / ``builder`` / ``validator``
    are *references* (dotted paths or repo-relative file paths) recorded in the
    governed result so an auditor can see exactly which component produced an
    artefact — they are not import hooks.
    """

    annex_id: str
    annex_name: str
    jurisdiction: str
    regulatory_regime: str
    report_version: str
    artefact_type: str

    authoritative_projector: str
    normaliser: str
    builder: str
    validator: str

    schema_reference: str
    field_universe_reference: str = ""
    rules_reference: str = ""
    code_order_reference: str = ""
    configuration_references: Tuple[str, ...] = ()

    supported_input_contracts: Tuple[str, ...] = ()
    output_contract: str = ""
    publication_target: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "annex_id": self.annex_id,
            "annex_name": self.annex_name,
            "jurisdiction": self.jurisdiction,
            "regulatory_regime": self.regulatory_regime,
            "report_version": self.report_version,
            "artefact_type": self.artefact_type,
            "authoritative_projector": self.authoritative_projector,
            "normaliser": self.normaliser,
            "builder": self.builder,
            "validator": self.validator,
            "schema_reference": self.schema_reference,
            "field_universe_reference": self.field_universe_reference,
            "rules_reference": self.rules_reference,
            "code_order_reference": self.code_order_reference,
            "configuration_references": list(self.configuration_references),
            "supported_input_contracts": list(self.supported_input_contracts),
            "output_contract": self.output_contract,
            "publication_target": self.publication_target,
        }


# --------------------------------------------------------------------------- #
# Execution request
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class DeliveryRequest:
    """One governed request to prepare an annex artefact.

    Security note — ``client_id`` and ``portfolio_id`` here are *caller input*.
    They are never trusted: :class:`engine.annex_delivery_agent.agent.AnnexDeliveryAgent`
    authorises them against the trusted :class:`~trakt_core.context.ExecutionContext`
    via :func:`trakt_core.tenancy.authorise_portfolio_access` before any path is
    resolved. The tenant always comes from the context.
    """

    annex_id: str
    reporting_date: str
    projected_input: str
    output_root: str

    #: Caller-supplied selector within the authenticated tenant. ``None`` means
    #: "the tenant's default book".
    portfolio_id: Optional[str] = None
    #: Caller-supplied label only; the authoritative client is the tenant.
    client_id: Optional[str] = None

    annex_version: Optional[str] = None
    client_config_reference: Optional[str] = None
    workflow_run_reference: Optional[str] = None
    approval_context: Optional[str] = None

    #: Re-run an already-completed run from scratch. Governed: the agent records
    #: who forced it and why in the audit trail.
    force_rerun: bool = False
    force_reason: Optional[str] = None
    #: Provider-specific knobs (e.g. Annex 2 currency / performance mode).
    options: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "annex_id": self.annex_id,
            "annex_version": self.annex_version,
            "reporting_date": self.reporting_date,
            "projected_input": self.projected_input,
            "output_root": self.output_root,
            "portfolio_id": self.portfolio_id,
            "client_id": self.client_id,
            "client_config_reference": self.client_config_reference,
            "workflow_run_reference": self.workflow_run_reference,
            "approval_context": self.approval_context,
            "force_rerun": self.force_rerun,
            "force_reason": self.force_reason,
            "options": dict(self.options),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DeliveryRequest":
        d = dict(data)
        return cls(
            annex_id=str(d["annex_id"]),
            reporting_date=str(d["reporting_date"]),
            projected_input=str(d["projected_input"]),
            output_root=str(d["output_root"]),
            portfolio_id=d.get("portfolio_id"),
            client_id=d.get("client_id"),
            annex_version=d.get("annex_version"),
            client_config_reference=d.get("client_config_reference"),
            workflow_run_reference=d.get("workflow_run_reference"),
            approval_context=d.get("approval_context"),
            force_rerun=bool(d.get("force_rerun", False)),
            force_reason=d.get("force_reason"),
            options=dict(d.get("options") or {}),
        )


# --------------------------------------------------------------------------- #
# Result
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class StageTiming:
    """Wall-clock and memory for one lifecycle stage."""

    stage: str
    seconds: float
    peak_rss_mb: Optional[float] = None
    resumed_from_checkpoint: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage": self.stage,
            "seconds": round(self.seconds, 3),
            "peak_rss_mb": (round(self.peak_rss_mb, 1)
                            if self.peak_rss_mb is not None else None),
            "resumed_from_checkpoint": self.resumed_from_checkpoint,
        }


@dataclass(frozen=True)
class ArtefactRef:
    """A produced file, by reference. Never carries the bytes."""

    kind: str
    path: str
    size_bytes: int = 0
    sha256: str = ""

    @property
    def label(self) -> str:
        return Path(self.path).name

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "path": self.path,
            "label": self.label,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ArtefactRef":
        d = dict(data)
        return cls(kind=str(d.get("kind", "")), path=str(d.get("path", "")),
                   size_bytes=int(d.get("size_bytes") or 0),
                   sha256=str(d.get("sha256") or ""))

    @classmethod
    def of_file(cls, kind: str, path: Path, *, hash_bytes: bool = True) -> "ArtefactRef":
        p = Path(path)
        size = p.stat().st_size if p.exists() else 0
        digest = file_sha256(p) if (hash_bytes and p.exists()) else ""
        return cls(kind=kind, path=str(p), size_bytes=size, sha256=digest)


@dataclass(frozen=True)
class SchemaValidationResult:
    """The outcome of validating the artefact against its authoritative schema.

    ``valid`` means *schema-valid only*. It is deliberately never called
    "correct": see :attr:`DeliveryResult.why_it_matters`.
    """

    validator: str
    schema_reference: str
    schema_sha256: str = ""
    valid: bool = False
    executed: bool = False
    error_count: int = 0
    errors: Tuple[Dict[str, Any], ...] = ()
    seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "validator": self.validator,
            "schema_reference": self.schema_reference,
            "schema_sha256": self.schema_sha256,
            "valid": self.valid,
            "executed": self.executed,
            "error_count": self.error_count,
            "errors": [dict(e) for e in self.errors],
            "seconds": round(self.seconds, 3),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SchemaValidationResult":
        d = dict(data)
        return cls(
            validator=str(d.get("validator", "")),
            schema_reference=str(d.get("schema_reference", "")),
            schema_sha256=str(d.get("schema_sha256") or ""),
            valid=bool(d.get("valid")),
            executed=bool(d.get("executed")),
            error_count=int(d.get("error_count") or 0),
            errors=tuple(dict(e) for e in (d.get("errors") or ())),
            seconds=float(d.get("seconds") or 0.0),
        )


@dataclass(frozen=True)
class DeliveryResult:
    """The governed answer of one annex delivery run.

    Carried as the ``result`` payload of a
    :class:`trakt_core.envelope.GovernedResult`, so callers get delivery facts
    and governance facts from one object without a second model.
    """

    run_id: str
    annex_id: str
    annex_version: str
    status: str

    summary: str = ""
    why_it_matters: str = ""

    record_count: int = 0
    projected_field_count: int = 0
    delivered_field_count: int = 0

    artefact: Optional[ArtefactRef] = None
    intermediate_artefacts: Tuple[ArtefactRef, ...] = ()

    schema_validation: Optional[SchemaValidationResult] = None
    blocking_errors: Tuple[Dict[str, Any], ...] = ()
    warnings: Tuple[str, ...] = ()

    #: Instrumented builder behaviour. See
    #: :mod:`engine.annex_delivery_agent.evidence`.
    transformation_evidence: Optional[Dict[str, Any]] = None

    source_references: Tuple[Dict[str, Any], ...] = ()
    configuration_versions: Mapping[str, str] = field(default_factory=dict)
    component_versions: Mapping[str, str] = field(default_factory=dict)
    audit_references: Mapping[str, Any] = field(default_factory=dict)

    publication_eligible: bool = False
    publication_blockers: Tuple[str, ...] = ()
    approval: ApprovalState = field(default_factory=ApprovalState)
    publication: Optional[Dict[str, Any]] = None

    timings: Tuple[StageTiming, ...] = ()
    total_seconds: float = 0.0
    peak_rss_mb: Optional[float] = None

    tenant_id: str = ""
    portfolio_id: Optional[str] = None
    reporting_date: str = ""
    run_directory: str = ""

    # -- predicates --------------------------------------------------------- #
    @property
    def prepared(self) -> bool:
        return self.status in PREPARED_STATUSES

    @property
    def failed(self) -> bool:
        return self.status in TERMINAL_FAILURE_STATUSES

    @property
    def requires_operator_review(self) -> bool:
        ev = self.transformation_evidence or {}
        return bool(ev.get("review_required")) or bool(self.warnings)

    # -- serialisation ------------------------------------------------------ #
    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "annex_id": self.annex_id,
            "annex_version": self.annex_version,
            "status": self.status,
            "summary": self.summary,
            "why_it_matters": self.why_it_matters,
            "record_count": self.record_count,
            "projected_field_count": self.projected_field_count,
            "delivered_field_count": self.delivered_field_count,
            "artefact": self.artefact.to_dict() if self.artefact else None,
            "artefact_size_bytes": self.artefact.size_bytes if self.artefact else 0,
            "intermediate_artefacts": [a.to_dict() for a in self.intermediate_artefacts],
            "schema_used": (self.schema_validation.schema_reference
                            if self.schema_validation else None),
            "schema_validation": (self.schema_validation.to_dict()
                                  if self.schema_validation else None),
            "validation_errors": ([dict(e) for e in self.schema_validation.errors]
                                  if self.schema_validation else []),
            "blocking_errors": [dict(e) for e in self.blocking_errors],
            "warnings": list(self.warnings),
            "transformation_evidence": self.transformation_evidence,
            "source_references": [dict(s) for s in self.source_references],
            "configuration_versions": dict(self.configuration_versions),
            "component_versions": dict(self.component_versions),
            "audit_references": dict(self.audit_references),
            "publication_eligible": self.publication_eligible,
            "publication_blockers": list(self.publication_blockers),
            "approval": self.approval.to_dict(),
            "publication": self.publication,
            "timings": [t.to_dict() for t in self.timings],
            "total_seconds": round(self.total_seconds, 3),
            "peak_rss_mb": (round(self.peak_rss_mb, 1)
                            if self.peak_rss_mb is not None else None),
            "tenant_id": self.tenant_id,
            "portfolio_id": self.portfolio_id,
            "reporting_date": self.reporting_date,
            "run_directory": self.run_directory,
            "requires_operator_review": self.requires_operator_review,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DeliveryResult":
        d = dict(data)
        artefact = d.get("artefact")
        sv = d.get("schema_validation")
        return cls(
            run_id=str(d.get("run_id", "")),
            annex_id=str(d.get("annex_id", "")),
            annex_version=str(d.get("annex_version", "")),
            status=str(d.get("status", DeliveryStatus.RECEIVED)),
            summary=str(d.get("summary") or ""),
            why_it_matters=str(d.get("why_it_matters") or ""),
            record_count=int(d.get("record_count") or 0),
            projected_field_count=int(d.get("projected_field_count") or 0),
            delivered_field_count=int(d.get("delivered_field_count") or 0),
            artefact=ArtefactRef.from_dict(artefact) if artefact else None,
            intermediate_artefacts=tuple(
                ArtefactRef.from_dict(a) for a in (d.get("intermediate_artefacts") or ())),
            schema_validation=SchemaValidationResult.from_dict(sv) if sv else None,
            blocking_errors=tuple(dict(e) for e in (d.get("blocking_errors") or ())),
            warnings=tuple(str(w) for w in (d.get("warnings") or ())),
            transformation_evidence=d.get("transformation_evidence"),
            source_references=tuple(dict(s) for s in (d.get("source_references") or ())),
            configuration_versions=dict(d.get("configuration_versions") or {}),
            component_versions=dict(d.get("component_versions") or {}),
            audit_references=dict(d.get("audit_references") or {}),
            publication_eligible=bool(d.get("publication_eligible")),
            publication_blockers=tuple(str(b) for b in (d.get("publication_blockers") or ())),
            approval=ApprovalState.from_dict(d.get("approval")),
            publication=d.get("publication"),
            timings=tuple(
                StageTiming(stage=str(t.get("stage", "")),
                            seconds=float(t.get("seconds") or 0.0),
                            peak_rss_mb=t.get("peak_rss_mb"),
                            resumed_from_checkpoint=bool(t.get("resumed_from_checkpoint")))
                for t in (d.get("timings") or ())),
            total_seconds=float(d.get("total_seconds") or 0.0),
            peak_rss_mb=d.get("peak_rss_mb"),
            tenant_id=str(d.get("tenant_id") or ""),
            portfolio_id=d.get("portfolio_id"),
            reporting_date=str(d.get("reporting_date") or ""),
            run_directory=str(d.get("run_directory") or ""),
        )

    def with_status(self, status: str, **changes: Any) -> "DeliveryResult":
        return replace(self, status=status, **changes)


# --------------------------------------------------------------------------- #
# Hashing helpers — used for deterministic run ids and idempotency
# --------------------------------------------------------------------------- #

_HASH_CHUNK = 1024 * 1024


def file_sha256(path: Path) -> str:
    """Streaming SHA-256 of a file. Streaming matters: artefacts reach 200 MB+."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(_HASH_CHUNK)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stable_digest(parts: Sequence[Any]) -> str:
    """A short, order-sensitive digest over governed inputs."""
    joined = "\x1f".join("" if p is None else str(p) for p in parts)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]


def config_fingerprints(paths: Mapping[str, str]) -> Dict[str, str]:
    """``{label: sha256}`` for each existing config file, ``"missing"`` otherwise.

    This is what makes "the same governed inputs" checkable: a changed rules
    file changes the fingerprint, which changes the run id, which prevents a
    stale artefact being served as if it were current.
    """
    out: Dict[str, str] = {}
    for label, raw in paths.items():
        if not raw:
            out[label] = "unset"
            continue
        p = Path(raw)
        out[label] = file_sha256(p) if p.is_file() else "missing"
    return out


__all__ = [
    "DeliveryStatus", "TERMINAL_FAILURE_STATUSES", "PREPARED_STATUSES",
    "PUBLISHABLE_STATUS", "ORDERED_STAGES",
    "STAGE_PREFLIGHT", "STAGE_NORMALISE", "STAGE_BUILD", "STAGE_VALIDATE",
    "APPROVAL_NOT_REQUESTED", "APPROVAL_PENDING", "APPROVAL_APPROVED",
    "APPROVAL_REJECTED", "APPROVAL_PROMOTED", "ApprovalState",
    "AnnexDefinition", "DeliveryRequest", "DeliveryResult",
    "ArtefactRef", "SchemaValidationResult", "StageTiming",
    "file_sha256", "text_sha256", "stable_digest", "config_fingerprints",
]
