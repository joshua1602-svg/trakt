"""mi_agent.concentration_tests.models — typed contracts, no pandas.

Every document these models serialise carries ``schema_version`` and is written
through the append-only store; nothing here mutates history. The vocabularies
are closed on purpose: an unknown status, outcome or concern code is a bug in
the caller, not a new state the platform silently learns.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

SCHEMA_VERSION = "1.0.0"

# --------------------------------------------------------------------------- #
# Vocabularies (closed)
# --------------------------------------------------------------------------- #

#: Library-match outcomes for an extracted test.
MATCH_MATCHED = "matched"          # direct library metric, parameters resolved
MATCH_COMPOSED = "composed"        # representable via approved primitives
MATCH_AMBIGUOUS = "ambiguous"      # plausible metric, definitions unresolved
MATCH_UNSUPPORTED = "unsupported"  # cannot be represented safely today
MATCH_OUTCOMES = (MATCH_MATCHED, MATCH_COMPOSED, MATCH_AMBIGUOUS, MATCH_UNSUPPORTED)

#: Proposal lifecycle. A proposal can only become an active portfolio control
#: through PROPOSAL_APPROVED + an explicit activation that mints a version.
PROPOSAL_PROPOSED = "proposed"
PROPOSAL_PENDING_CONFIRMATION = "pending_confirmation"
PROPOSAL_PENDING_APPROVAL = "pending_approval"
PROPOSAL_APPROVED = "approved"
PROPOSAL_REJECTED = "rejected"
PROPOSAL_SUPERSEDED = "superseded"
PROPOSAL_UNSUPPORTED = "unsupported"
PROPOSAL_NOT_APPLICABLE = "not_applicable"
PROPOSAL_CLARIFICATION_REQUESTED = "clarification_requested"
PROPOSAL_STATUSES = (
    PROPOSAL_PROPOSED, PROPOSAL_PENDING_CONFIRMATION, PROPOSAL_PENDING_APPROVAL,
    PROPOSAL_APPROVED, PROPOSAL_REJECTED, PROPOSAL_SUPERSEDED,
    PROPOSAL_UNSUPPORTED, PROPOSAL_NOT_APPLICABLE,
    PROPOSAL_CLARIFICATION_REQUESTED,
)

#: Evaluation statuses. Fail closed: unknown / unsupported / missing-data tests
#: never show as ``pass``.
STATUS_PASS = "pass"
STATUS_WARNING = "warning"
STATUS_BREACH = "breach"
STATUS_UNAVAILABLE = "unavailable"
STATUS_INSUFFICIENT_DATA = "insufficient_data"
STATUS_PENDING_EFFECTIVE_DATE = "pending_effective_date"
STATUS_EXPIRED = "expired"
TEST_STATUSES = (
    STATUS_PASS, STATUS_WARNING, STATUS_BREACH, STATUS_UNAVAILABLE,
    STATUS_INSUFFICIENT_DATA, STATUS_PENDING_EFFECTIVE_DATE, STATUS_EXPIRED,
)

#: Data availability for one evaluation.
DATA_OK = "ok"
DATA_MISSING = "data_missing"
DATA_PARTIAL = "data_partial"
DATA_EXTERNAL_UNCONFIGURED = "external_reference_unconfigured"
DATA_STATUSES = (DATA_OK, DATA_MISSING, DATA_PARTIAL, DATA_EXTERNAL_UNCONFIGURED)

#: Comparison operators a test may declare. ``max`` / ``min`` mirror the
#: existing risk-limits contract; the symmetric forms are for completeness in
#: extraction and normalise onto max/min during validation.
OPERATOR_MAX = "max"   # actual must be <= threshold
OPERATOR_MIN = "min"   # actual must be >= threshold
OPERATORS = (OPERATOR_MAX, OPERATOR_MIN)


class ConcernCode:
    """Closed set of concern codes an extraction / mapping may raise."""

    DENOMINATOR_UNDEFINED = "denominator_undefined"
    NUMERATOR_UNDEFINED = "numerator_undefined"
    BALANCE_BASIS_UNDEFINED = "balance_basis_undefined"
    WEIGHTING_BASIS_UNDEFINED = "weighting_basis_undefined"
    AGGREGATION_BASIS_UNDEFINED = "aggregation_basis_undefined"
    REGION_MAPPING_UNCERTAIN = "region_mapping_uncertain"
    BORROWER_DEFINITION_UNCERTAIN = "borrower_definition_uncertain"
    JOINT_DEFINITION_UNCERTAIN = "joint_definition_uncertain"
    NET_WAC_DEFINITION_UNCERTAIN = "net_wac_definition_uncertain"
    INDEX_SERIES_MISSING = "index_series_missing"
    INDEX_BASE_DATE_MISSING = "index_base_date_missing"
    THRESHOLD_UNIT_UNCERTAIN = "threshold_unit_uncertain"
    COMPARISON_OPERATOR_UNCERTAIN = "comparison_operator_uncertain"
    DUPLICATED_TEST = "duplicated_test"
    CONFLICTING_THRESHOLDS = "conflicting_thresholds"
    EFFECTIVE_DATE_MISSING = "effective_date_missing"
    SOURCE_EVIDENCE_MISSING = "source_evidence_missing"
    CANONICAL_FIELD_UNAVAILABLE = "canonical_field_unavailable"
    UNSUPPORTED_METRIC = "unsupported_metric"

    ALL = (
        DENOMINATOR_UNDEFINED, NUMERATOR_UNDEFINED, BALANCE_BASIS_UNDEFINED,
        WEIGHTING_BASIS_UNDEFINED, AGGREGATION_BASIS_UNDEFINED,
        REGION_MAPPING_UNCERTAIN, BORROWER_DEFINITION_UNCERTAIN,
        JOINT_DEFINITION_UNCERTAIN, NET_WAC_DEFINITION_UNCERTAIN,
        INDEX_SERIES_MISSING, INDEX_BASE_DATE_MISSING,
        THRESHOLD_UNIT_UNCERTAIN, COMPARISON_OPERATOR_UNCERTAIN,
        DUPLICATED_TEST, CONFLICTING_THRESHOLDS, EFFECTIVE_DATE_MISSING,
        SOURCE_EVIDENCE_MISSING, CANONICAL_FIELD_UNAVAILABLE,
        UNSUPPORTED_METRIC,
    )


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def stable_hash(*parts: str) -> str:
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:16]


def _from_known(cls, d: Dict[str, Any]):
    known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
    return cls(**known)


# --------------------------------------------------------------------------- #
# Evidence / provenance
# --------------------------------------------------------------------------- #
@dataclass
class SourceEvidence:
    """Where a test came from. A REFERENCE to the evidence, never the document."""

    source_reference: str = ""       # document / response identifier
    source_text: str = ""            # the wording as supplied
    locator: str = ""                # page, sheet, cell range or line, if known
    supplied_at: str = ""            # ISO timestamp when the evidence arrived

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SourceEvidence":
        return _from_known(cls, d or {})


# --------------------------------------------------------------------------- #
# Proposal
# --------------------------------------------------------------------------- #
@dataclass
class TestProposal:
    """One extracted concentration test, proposed for operator review.

    ``parameters`` are the client-specific configuration for the proposed
    metric (threshold, operator, filters, buckets, denominators); they are
    validated against the metric's parameter schema before the proposal can be
    approved. ``concern_codes`` / ``confirmation_questions`` carry everything
    unresolved — the model may add to them, only an operator removes them.
    """

    proposal_id: str = field(default_factory=lambda: new_id("ctp"))
    onboarding_run_id: str = ""
    client_id: str = ""
    portfolio_id: str = ""
    evidence: SourceEvidence = field(default_factory=SourceEvidence)
    extracted_test_name: str = ""
    extracted_category: str = ""
    proposed_metric_id: str = ""
    match_outcome: str = MATCH_UNSUPPORTED
    extracted_operator: str = ""
    extracted_threshold: Optional[float] = None
    extracted_unit: str = ""
    numerator_basis: str = ""
    denominator_basis: str = ""
    weighting_basis: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    segment_filters: Dict[str, Any] = field(default_factory=dict)
    bucket_boundaries: List[float] = field(default_factory=list)
    effective_date: str = ""
    expiry_date: str = ""
    cure_or_waiver_notes: str = ""
    extraction_confidence: float = 0.0
    mapping_confidence: float = 0.0
    concern_codes: List[str] = field(default_factory=list)
    concern_messages: List[str] = field(default_factory=list)
    confirmation_questions: List[str] = field(default_factory=list)
    confirmation_answers: List[Dict[str, str]] = field(default_factory=list)
    status: str = PROPOSAL_PROPOSED
    #: The recorded operator decision, once one is taken (ApprovalRecord dict).
    approval: Dict[str, Any] = field(default_factory=dict)
    severity: str = "high"
    operator_notes: List[Dict[str, str]] = field(default_factory=list)
    #: Structured request raised when the test is unsupported — never an
    #: improvised formula.
    implementation_request: Dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
    updated_at: str = ""
    version: int = 1
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.match_outcome not in MATCH_OUTCOMES:
            raise ValueError(f"unknown match outcome: {self.match_outcome!r}")
        if self.status not in PROPOSAL_STATUSES:
            raise ValueError(f"unknown proposal status: {self.status!r}")
        for code in self.concern_codes:
            if code not in ConcernCode.ALL:
                raise ValueError(f"unknown concern code: {code!r}")

    @property
    def unresolved(self) -> bool:
        return bool(self.concern_codes) or bool(self.confirmation_questions)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["evidence"] = self.evidence.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TestProposal":
        d = dict(d or {})
        d["evidence"] = SourceEvidence.from_dict(d.get("evidence") or {})
        return _from_known(cls, d)


# --------------------------------------------------------------------------- #
# Approval / activation
# --------------------------------------------------------------------------- #
@dataclass
class ApprovalRecord:
    """The operator decision that let a proposal become (or stay out of) the
    active configuration. Immutable once written."""

    decision: str = ""               # approved | rejected | unsupported | not_applicable
    operator: str = ""
    decided_at: str = ""
    approved_metric_id: str = ""
    approved_parameters: Dict[str, Any] = field(default_factory=dict)
    source_version: int = 1          # proposal version the decision was taken on
    effective_date: str = ""
    comments: str = ""
    prior_active_test_id: str = ""   # when superseding an earlier approved test

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ApprovalRecord":
        return _from_known(cls, d or {})


@dataclass
class ActiveTest:
    """One approved test inside an immutable activated configuration version."""

    test_id: str = field(default_factory=lambda: new_id("ct"))
    proposal_id: str = ""
    metric_id: str = ""
    display_name: str = ""
    category: str = ""
    operator: str = OPERATOR_MAX
    threshold: Optional[float] = None
    unit: str = "percent"
    #: Warning at this fraction of the threshold (max) or threshold/fraction
    #: (min). Configurable per test; defaults follow the existing 0.9 amber
    #: convention.
    warning_fraction: float = 0.9
    parameters: Dict[str, Any] = field(default_factory=dict)
    effective_date: str = ""
    expiry_date: str = ""
    evidence: SourceEvidence = field(default_factory=SourceEvidence)
    approval: ApprovalRecord = field(default_factory=ApprovalRecord)
    severity: str = "high"           # high | critical (display/escalation only)
    definition_notes: str = ""       # the confirmed contractual definition
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.operator not in OPERATORS:
            raise ValueError(f"unknown operator: {self.operator!r}")

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["evidence"] = self.evidence.to_dict()
        d["approval"] = self.approval.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ActiveTest":
        d = dict(d or {})
        d["evidence"] = SourceEvidence.from_dict(d.get("evidence") or {})
        d["approval"] = ApprovalRecord.from_dict(d.get("approval") or {})
        return _from_known(cls, d)


@dataclass
class ActiveConfiguration:
    """An immutable, versioned set of active tests for one tenant/portfolio.

    Activation writes a new version and flips the previous one to superseded;
    history is never overwritten. ``content_hash`` makes re-activation of the
    same content idempotent (the same guarantee ``OnboardingStore.commit``
    gives configuration versions).
    """

    client_id: str = ""
    portfolio_id: str = ""           # "" = client-level (all portfolios)
    version: int = 1
    status: str = "active"           # active | superseded
    tests: List[ActiveTest] = field(default_factory=list)
    activated_by: str = ""
    activated_at: str = ""
    based_on_version: Optional[int] = None
    reason: str = ""
    content_hash: str = ""
    library_version: str = ""
    schema_version: str = SCHEMA_VERSION

    def compute_content_hash(self) -> str:
        payload = [t.to_dict() for t in self.tests]
        for t in payload:  # identity fields must not defeat idempotency
            t.pop("test_id", None)
        return stable_hash(canonical_json(payload))

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["tests"] = [t.to_dict() for t in self.tests]
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ActiveConfiguration":
        d = dict(d or {})
        d["tests"] = [ActiveTest.from_dict(t) for t in (d.get("tests") or [])]
        return _from_known(cls, d)
