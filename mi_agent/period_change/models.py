"""mi_agent.period_change.models — the workflow's typed contracts.

Everything the workflow produces is a frozen dataclass with a ``to_dict()``.
Two reasons: the numbers are computed once and cannot be mutated by a presenter,
and the serialised form is the governed result contract that the API adapter
carries inside ``GovernedResult.result`` — so a channel renders the structure it
is given rather than recomputing anything.

Nothing in this module calculates. It names.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

#: Stable workflow / capability identifier. Follows the repository's snake_case
#: route naming (``period_movement``, ``temporal_compare``, ``cohort_progression``).
WORKFLOW_ID = "period_change_analysis"

#: Bumped when a calculation changes in a way that would alter a published
#: number. Recorded in the audit block so a stored result can be reproduced.
CALCULATION_VERSION = "1.0.0"

#: The result contract's own version, separate from the calculation version:
#: a purely additive shape change bumps this and not the calculation.
RESULT_SCHEMA_VERSION = "1.0.0"


# --------------------------------------------------------------------------- #
# Controlled statuses — §17 of the workflow specification
# --------------------------------------------------------------------------- #
STATUS_AVAILABLE = "available"
STATUS_PARTIALLY_AVAILABLE = "partially_available"
STATUS_NOT_AVAILABLE = "not_available"
STATUS_NOT_COMPARABLE = "not_comparable"
STATUS_NOT_COMPARABLE_DUE_TO_AVAILABILITY = "not_comparable_due_to_availability"
STATUS_INSUFFICIENT_HISTORY = "insufficient_history"
STATUS_INVALID_WEIGHT_POPULATION = "invalid_weight_population"
STATUS_ZERO_DENOMINATOR = "zero_denominator"
STATUS_SOURCE_CONVENTION_UNCERTAIN = "source_convention_uncertain"

ALL_STATUSES: Tuple[str, ...] = (
    STATUS_AVAILABLE, STATUS_PARTIALLY_AVAILABLE, STATUS_NOT_AVAILABLE,
    STATUS_NOT_COMPARABLE, STATUS_NOT_COMPARABLE_DUE_TO_AVAILABILITY,
    STATUS_INSUFFICIENT_HISTORY, STATUS_INVALID_WEIGHT_POPULATION,
    STATUS_ZERO_DENOMINATOR, STATUS_SOURCE_CONVENTION_UNCERTAIN,
)


# --------------------------------------------------------------------------- #
# Movement units — §11. A unit is a fact about the field, not a display choice.
# --------------------------------------------------------------------------- #
UNIT_CURRENCY = "currency"
UNIT_COUNT = "count"
UNIT_PERCENTAGE_POINT = "percentage_point"
UNIT_RATIO = "ratio"
UNIT_YEARS = "years"
UNIT_NOT_APPLICABLE = "not_applicable"

#: Units for which a percentage-point (and basis-point) movement is the correct
#: expression of change, rather than a percentage of the starting value.
POINT_UNITS: frozenset = frozenset({UNIT_PERCENTAGE_POINT})

#: Units for which no numerical percentage movement is meaningful at all.
NON_NUMERIC_UNITS: frozenset = frozenset({UNIT_NOT_APPLICABLE})


# --------------------------------------------------------------------------- #
# Movement basis — how the movement number was derived from the two snapshots.
# The distinction the specification insists on: a differenced cumulative is not
# the same statement as a compared pair of period flows.
# --------------------------------------------------------------------------- #
BASIS_POINT_IN_TIME_DIFFERENCE = "point_in_time_difference"
BASIS_FLOW_LEVEL_COMPARISON = "flow_level_comparison"
BASIS_CUMULATIVE_DIFFERENCE = "cumulative_difference"
BASIS_STATIC_BASELINE_REFERENCE = "static_baseline_reference"
BASIS_SHARE_SHIFT = "share_shift"

#: Attached to every ``period_flow`` metric change so the number can never be
#: read as an amount accumulated between the two dates.
FLOW_COMPARISON_NOTE = (
    "Period flows are compared as independent reporting-period totals. The "
    "movement is the change in the periodic flow level between the two "
    "reporting periods, not an amount arising between the two dates.")

#: Attached when a cumulative field falls between snapshots (§7).
CUMULATIVE_DECLINE_NOTE = (
    "The cumulative value fell between the two snapshots. A cumulative measure "
    "is not expected to decrease; this may indicate a restatement or a source "
    "convention difference. The observed difference is retained unmodified.")


# --------------------------------------------------------------------------- #
# Controlled interpretation — §12
# --------------------------------------------------------------------------- #
INTERPRETATION_IMPROVEMENT = "improvement"
INTERPRETATION_DETERIORATION = "deterioration"
INTERPRETATION_NEUTRAL = "neutral"
INTERPRETATION_NO_MOVEMENT = "no_movement"
#: Used for ``neutral`` / ``context_dependent`` directionality: the movement is
#: reported, no better/worse judgement is asserted.
INTERPRETATION_NOT_ASSESSED = "not_assessed"


# --------------------------------------------------------------------------- #
# Controlled significance language — §11. No thresholds are invented here.
# --------------------------------------------------------------------------- #
SIGNIFICANCE_LARGEST_INCREASE = "largest_observed_increase"
SIGNIFICANCE_LARGEST_DECREASE = "largest_observed_decrease"
SIGNIFICANCE_NOTABLE_BY_RANK = "notable_by_rank"
SIGNIFICANCE_MAIN_OBSERVED_MOVEMENT = "main_observed_movement"
SIGNIFICANCE_RELATIVELY_STABLE = "relatively_stable"
SIGNIFICANCE_NO_BASIS = "insufficient_basis_for_materiality_assessment"

#: The single sentence the workflow uses wherever a caller might expect a
#: materiality verdict. Materiality requires a configured rule; there is none.
MATERIALITY_DISCLAIMER = (
    "No governed materiality threshold is configured for this portfolio, so "
    "movements are ranked by observed size only. No movement is described as "
    "material, significant, a breach or high risk.")


# --------------------------------------------------------------------------- #
# Workflow modes — §10
# --------------------------------------------------------------------------- #
MODE_REQUESTED_METRIC = "requested_metric"
MODE_CONCEPT = "concept"
MODE_PORTFOLIO_OVERVIEW = "portfolio_overview"

ALL_MODES: Tuple[str, ...] = (
    MODE_REQUESTED_METRIC, MODE_CONCEPT, MODE_PORTFOLIO_OVERVIEW)


# --------------------------------------------------------------------------- #
# Failure reasons — §5. Mapped onto the existing error taxonomy by the adapter.
# --------------------------------------------------------------------------- #
FAIL_INSUFFICIENT_SNAPSHOTS = "insufficient_snapshots"
FAIL_PORTFOLIO_ABSENT_AT_PERIOD = "portfolio_absent_at_period"
FAIL_AMBIGUOUS_PERIOD_RANGE = "ambiguous_period_range"
FAIL_REVERSED_PERIOD_RANGE = "reversed_period_range"
FAIL_IDENTICAL_SNAPSHOTS = "identical_snapshots"
FAIL_CROSS_TENANT_ACCESS = "cross_tenant_access"
FAIL_NO_ELIGIBLE_FIELDS = "no_eligible_fields"
FAIL_REGISTRY_UNAVAILABLE = "registry_unavailable"

#: Human wording for each failure. One place, so every channel says the same.
FAILURE_MESSAGES: Mapping[str, str] = {
    FAIL_INSUFFICIENT_SNAPSHOTS: (
        "A period-change analysis needs two governed portfolio snapshots; "
        "fewer than two are available for this scope."),
    FAIL_PORTFOLIO_ABSENT_AT_PERIOD: (
        "The requested portfolio does not exist at both reporting dates, so "
        "the two periods cannot be compared."),
    FAIL_AMBIGUOUS_PERIOD_RANGE: (
        "The requested date range could not be resolved to two governed "
        "snapshots without guessing which periods were meant."),
    FAIL_REVERSED_PERIOD_RANGE: (
        "The requested start date is later than the requested end date."),
    FAIL_IDENTICAL_SNAPSHOTS: (
        "Both requested periods resolve to the same governed snapshot, so "
        "there is no movement to report."),
    FAIL_CROSS_TENANT_ACCESS: (
        "The requested comparison would require data outside the authorised "
        "tenant scope."),
    FAIL_NO_ELIGIBLE_FIELDS: (
        "No governed field tagged for period-change analysis is available in "
        "both snapshots for this portfolio."),
    FAIL_REGISTRY_UNAVAILABLE: (
        "The governed Business Semantics Registry could not be loaded, so no "
        "period-change analysis can be performed."),
}


class PeriodChangeFailure(Exception):
    """A controlled, explicit workflow failure.

    Carries a stable ``reason`` from the list above plus optional structured
    detail. The adapter maps the reason onto the repository's existing
    ``trakt_core.errors.ErrorCode`` taxonomy — this module deliberately does not
    import it, so the workflow core stays dependency-free.
    """

    def __init__(self, reason: str, message: Optional[str] = None,
                 detail: Optional[Mapping[str, Any]] = None) -> None:
        self.reason = reason
        self.message = message or FAILURE_MESSAGES.get(
            reason, "The period-change analysis could not be performed.")
        self.detail: Dict[str, Any] = dict(detail or {})
        super().__init__(self.message)

    def to_dict(self) -> Dict[str, Any]:
        return {"reason": self.reason, "message": self.message,
                "detail": dict(self.detail)}


# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SnapshotFrame:
    """One resolved portfolio snapshot and the frame that backs it.

    ``dataset_label`` is a file NAME or governed label, never a full path — the
    same discipline ``trakt_core.envelope.SnapshotRef`` applies, so nothing
    leaks a storage layout into a result or a log.
    """

    snapshot_id: str
    reporting_date: Optional[str]
    frame: Any = None                       # pandas.DataFrame (kept untyped here)
    dataset_label: Optional[str] = None
    dataset_reference: Optional[str] = None
    row_count: Optional[int] = None
    portfolio_ids: Tuple[str, ...] = ()

    @property
    def period_label(self) -> str:
        """``YYYY-MM`` where a reporting date exists, else the snapshot id."""
        if self.reporting_date:
            return str(self.reporting_date)[:7]
        return self.snapshot_id

    def reference(self) -> Dict[str, Any]:
        """The provenance reference recorded against every calculated figure."""
        return {
            "snapshot_id": self.snapshot_id,
            "reporting_date": self.reporting_date,
            "dataset_label": self.dataset_label,
            "dataset_reference": self.dataset_reference,
            "row_count": self.row_count,
        }


@dataclass(frozen=True)
class PortfolioScopeRef:
    """The portfolio/client scope a period-change analysis was run for.

    ``tenant_id`` comes from the execution context, never from the question —
    the workflow only records it so the audit block can prove which tenant the
    numbers belong to.
    """

    tenant_id: Optional[str] = None
    context_id: Optional[str] = None
    label: Optional[str] = None
    portfolio_ids: Tuple[str, ...] = ()
    asset_classes: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tenant_id": self.tenant_id,
            "context_id": self.context_id,
            "label": self.label,
            "portfolio_ids": list(self.portfolio_ids),
            "asset_classes": list(self.asset_classes),
        }


# --------------------------------------------------------------------------- #
# Period resolution — §5
# --------------------------------------------------------------------------- #
METHOD_EXPLICIT_DATES = "explicit_dates"
METHOD_CURRENT_VS_PREVIOUS = "current_vs_previous"
METHOD_MONTH_ON_MONTH = "month_on_month"
METHOD_QUARTER_ON_QUARTER = "quarter_on_quarter"
METHOD_YEAR_ON_YEAR = "year_on_year"
METHOD_YEAR_TO_DATE = "year_to_date"
METHOD_LATEST_AVAILABLE_PAIR = "latest_available_pair"

ALL_RESOLUTION_METHODS: Tuple[str, ...] = (
    METHOD_EXPLICIT_DATES, METHOD_CURRENT_VS_PREVIOUS, METHOD_MONTH_ON_MONTH,
    METHOD_QUARTER_ON_QUARTER, METHOD_YEAR_ON_YEAR, METHOD_YEAR_TO_DATE,
    METHOD_LATEST_AVAILABLE_PAIR,
)


@dataclass(frozen=True)
class PeriodResolution:
    """What was asked for, what it resolved to, and whether it was adjusted."""

    requested_start: Optional[str]
    requested_end: Optional[str]
    resolution_method: str
    start_snapshot: SnapshotFrame
    end_snapshot: SnapshotFrame
    start_adjusted: bool = False
    end_adjusted: bool = False
    adjustment_notes: Tuple[str, ...] = ()
    available_snapshots: Tuple[str, ...] = ()
    portfolio_scope: PortfolioScopeRef = field(default_factory=PortfolioScopeRef)

    @property
    def any_adjusted(self) -> bool:
        return self.start_adjusted or self.end_adjusted

    def to_dict(self) -> Dict[str, Any]:
        return {
            "requested_start_period": self.requested_start,
            "requested_end_period": self.requested_end,
            "resolution_method": self.resolution_method,
            "resolved_start_snapshot": self.start_snapshot.reference(),
            "resolved_end_snapshot": self.end_snapshot.reference(),
            "start_adjusted_to_available_snapshot": self.start_adjusted,
            "end_adjusted_to_available_snapshot": self.end_adjusted,
            "adjustment_notes": list(self.adjustment_notes),
            "available_snapshots": list(self.available_snapshots),
            "portfolio_scope": self.portfolio_scope.to_dict(),
        }


# --------------------------------------------------------------------------- #
# Per-snapshot aggregation outcome
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class AggregateOutcome:
    """One governed aggregate at one snapshot, with its population evidence."""

    value: Optional[float]
    status: str
    aggregation: str
    valid_population: int = 0
    excluded_population: int = 0
    numerator: Optional[float] = None
    denominator: Optional[float] = None
    total_weight: Optional[float] = None
    weight_field: Optional[str] = None
    share_basis: Optional[str] = None
    notes: Tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return self.value is not None and self.status in (
            STATUS_AVAILABLE, STATUS_PARTIALLY_AVAILABLE)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "value": self.value,
            "status": self.status,
            "aggregation": self.aggregation,
            "valid_population": self.valid_population,
            "excluded_population": self.excluded_population,
            "numerator": self.numerator,
            "denominator": self.denominator,
            "total_weight": self.total_weight,
            "weight_field": self.weight_field,
            "share_basis": self.share_basis,
            "notes": list(self.notes),
        }


# --------------------------------------------------------------------------- #
# Metric change — §15
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class MetricChange:
    """One governed field's movement between the two resolved snapshots."""

    field: str
    display_name: str
    analytical_concept: str
    analytical_role: str
    temporality: str
    aggregation: str
    weight_field: Optional[str]
    share_basis: Optional[str]
    start_value: Optional[float]
    end_value: Optional[float]
    movement_value: Optional[float]
    movement_unit: str
    movement_basis: str
    relative_change: Optional[float]
    basis_point_change: Optional[float]
    directionality: str
    interpretation: str
    status: str
    start: AggregateOutcome
    end: AggregateOutcome
    confidence: str
    rationale: str
    evidence: Tuple[Dict[str, Any], ...] = ()
    notes: Tuple[str, ...] = ()
    significance: str = SIGNIFICANCE_NO_BASIS
    movement_rank: Optional[int] = None
    contribution_to_balance_change: Optional[float] = None

    @property
    def comparable(self) -> bool:
        return self.movement_value is not None and self.status == STATUS_AVAILABLE

    def to_dict(self) -> Dict[str, Any]:
        return {
            "canonical_field": self.field,
            "display_name": self.display_name,
            "analytical_concept": self.analytical_concept,
            "analytical_role": self.analytical_role,
            "temporality": self.temporality,
            "aggregation": self.aggregation,
            "weight_field": self.weight_field,
            "share_basis": self.share_basis,
            "start_value": self.start_value,
            "end_value": self.end_value,
            "movement_value": self.movement_value,
            "movement_unit": self.movement_unit,
            "movement_basis": self.movement_basis,
            "relative_change": self.relative_change,
            "basis_point_change": self.basis_point_change,
            "directionality": self.directionality,
            "interpretation": self.interpretation,
            "status": self.status,
            "valid_population": {"start": self.start.valid_population,
                                 "end": self.end.valid_population},
            "excluded_population": {"start": self.start.excluded_population,
                                    "end": self.end.excluded_population},
            "start_detail": self.start.to_dict(),
            "end_detail": self.end.to_dict(),
            "confidence": self.confidence,
            "rationale": self.rationale,
            "evidence": [dict(e) for e in self.evidence],
            "notes": list(self.notes),
            "significance": self.significance,
            "movement_rank": self.movement_rank,
            "contribution_to_balance_change": self.contribution_to_balance_change,
        }


# --------------------------------------------------------------------------- #
# Distribution change — §13
# --------------------------------------------------------------------------- #
#: The label existing analytics use for a missing categorical value. Reused so a
#: distribution bucket matches every other stratification in the platform.
UNKNOWN_CATEGORY = "Unknown"


@dataclass(frozen=True)
class CategoryShift:
    """One category's movement inside a distribution."""

    category: str
    start_count: int
    end_count: int
    start_count_share: Optional[float]
    end_count_share: Optional[float]
    count_share_movement: Optional[float]
    start_balance: Optional[float] = None
    end_balance: Optional[float] = None
    start_balance_share: Optional[float] = None
    end_balance_share: Optional[float] = None
    balance_share_movement: Optional[float] = None
    presence: str = "both"      # both | start_only | end_only

    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category,
            "start_count": self.start_count,
            "end_count": self.end_count,
            "start_count_share": self.start_count_share,
            "end_count_share": self.end_count_share,
            "count_share_movement": self.count_share_movement,
            "start_balance": self.start_balance,
            "end_balance": self.end_balance,
            "start_balance_share": self.start_balance_share,
            "end_balance_share": self.end_balance_share,
            "balance_share_movement": self.balance_share_movement,
            "presence": self.presence,
        }


@dataclass(frozen=True)
class DistributionChange:
    """A dimension's composition shift between the two snapshots."""

    field: str
    display_name: str
    analytical_concept: str
    status: str
    categories: Tuple[CategoryShift, ...]
    balance_field: Optional[str]
    start_total_count: int
    end_total_count: int
    start_total_balance: Optional[float]
    end_total_balance: Optional[float]
    largest_increases: Tuple[str, ...] = ()
    largest_decreases: Tuple[str, ...] = ()
    confidence: str = ""
    rationale: str = ""
    evidence: Tuple[Dict[str, Any], ...] = ()
    notes: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "canonical_field": self.field,
            "display_name": self.display_name,
            "analytical_concept": self.analytical_concept,
            "analytical_role": "dimension",
            "aggregation": "distribution",
            "status": self.status,
            "balance_field": self.balance_field,
            "start_total_count": self.start_total_count,
            "end_total_count": self.end_total_count,
            "start_total_balance": self.start_total_balance,
            "end_total_balance": self.end_total_balance,
            "categories": [c.to_dict() for c in self.categories],
            "largest_increases": list(self.largest_increases),
            "largest_decreases": list(self.largest_decreases),
            "confidence": self.confidence,
            "rationale": self.rationale,
            "evidence": [dict(e) for e in self.evidence],
            "notes": list(self.notes),
        }


# --------------------------------------------------------------------------- #
# Balance bridge — §14
# --------------------------------------------------------------------------- #
BRIDGE_STATUS_AVAILABLE = "available"
BRIDGE_STATUS_UNAVAILABLE_NO_IDENTIFIER = "unavailable_no_loan_identifier"
BRIDGE_STATUS_UNAVAILABLE_DUPLICATE_IDENTIFIERS = "unavailable_duplicate_identifiers"
BRIDGE_STATUS_UNAVAILABLE_NO_BALANCE_FIELD = "unavailable_no_balance_field"
BRIDGE_STATUS_UNAVAILABLE_MISSING_IDENTIFIERS = "unavailable_missing_identifiers"
BRIDGE_STATUS_DOES_NOT_RECONCILE = "does_not_reconcile"

#: Absolute currency tolerance the reconciliation is allowed to miss by, to
#: absorb float accumulation across a large book. Documented, not silent.
BRIDGE_ROUNDING_TOLERANCE = 0.01


@dataclass(frozen=True)
class BalanceBridge:
    """Opening → closing balance reconciliation over a stable loan identifier."""

    status: str
    balance_field: Optional[str] = None
    identifier_field: Optional[str] = None
    opening_balance: Optional[float] = None
    closing_balance: Optional[float] = None
    new_loan_balance: Optional[float] = None
    exited_loan_balance: Optional[float] = None
    continuing_movement: Optional[float] = None
    new_loan_count: Optional[int] = None
    exited_loan_count: Optional[int] = None
    continuing_loan_count: Optional[int] = None
    reconciles: Optional[bool] = None
    residual: Optional[float] = None
    tolerance: float = BRIDGE_ROUNDING_TOLERANCE
    limitation: Optional[str] = None
    evidence: Tuple[Dict[str, Any], ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "balance_field": self.balance_field,
            "identifier_field": self.identifier_field,
            "opening_balance": self.opening_balance,
            "new_loan_closing_balance": self.new_loan_balance,
            "exited_loan_opening_balance": self.exited_loan_balance,
            "movement_on_continuing_loans": self.continuing_movement,
            "closing_balance": self.closing_balance,
            "new_loan_count": self.new_loan_count,
            "exited_loan_count": self.exited_loan_count,
            "continuing_loan_count": self.continuing_loan_count,
            "reconciles": self.reconciles,
            "residual": self.residual,
            "rounding_tolerance": self.tolerance,
            "limitation": self.limitation,
            "evidence": [dict(e) for e in self.evidence],
        }


# --------------------------------------------------------------------------- #
# Field selection audit — §16
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ExcludedCandidate:
    """A BSR entry that was considered and rejected, with the reason."""

    field: str
    reason: str
    detail: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {"canonical_field": self.field, "reason": self.reason,
                "detail": self.detail}


@dataclass(frozen=True)
class FieldSelection:
    """The outcome of the governed field-selection policy."""

    mode: str
    measures: Tuple[str, ...]
    dimensions: Tuple[str, ...]
    excluded: Tuple[ExcludedCandidate, ...]
    concepts_covered: Tuple[str, ...]
    policy_name: str
    policy_version: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "selected_measures": list(self.measures),
            "selected_dimensions": list(self.dimensions),
            "excluded_candidates": [e.to_dict() for e in self.excluded],
            "concepts_covered": list(self.concepts_covered),
            "policy_name": self.policy_name,
            "policy_version": self.policy_version,
        }


# --------------------------------------------------------------------------- #
# The governed result — §15
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PeriodChangeResult:
    """The workflow's complete, auditable output.

    ``summary`` is generated deterministically from the tables below it; it
    never introduces a fact that is not present in ``metric_changes``,
    ``distribution_changes`` or ``balance_bridge``.
    """

    workflow_id: str
    result_schema_version: str
    request_interpretation: Mapping[str, Any]
    portfolio_scope: PortfolioScopeRef
    period_resolution: PeriodResolution
    dataset_provenance: Tuple[Dict[str, Any], ...]
    summary: Mapping[str, Any]
    metric_changes: Tuple[MetricChange, ...]
    distribution_changes: Tuple[DistributionChange, ...]
    balance_bridge: Optional[BalanceBridge]
    field_selection: FieldSelection
    warnings: Tuple[str, ...]
    limitations: Tuple[str, ...]
    evidence: Tuple[Dict[str, Any], ...]
    audit: Mapping[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "workflow_id": self.workflow_id,
            "result_schema_version": self.result_schema_version,
            "request_interpretation": dict(self.request_interpretation),
            "portfolio_scope": self.portfolio_scope.to_dict(),
            "period_resolution": self.period_resolution.to_dict(),
            "dataset_provenance": [dict(d) for d in self.dataset_provenance],
            "summary": dict(self.summary),
            "metric_changes": [m.to_dict() for m in self.metric_changes],
            "distribution_changes": [d.to_dict() for d in self.distribution_changes],
            "balance_bridge": (self.balance_bridge.to_dict()
                               if self.balance_bridge else None),
            "field_selection": self.field_selection.to_dict(),
            "warnings": list(self.warnings),
            "limitations": list(self.limitations),
            "evidence": [dict(e) for e in self.evidence],
            "audit": dict(self.audit),
        }


def evidence_ref(kind: str, snapshot: SnapshotFrame, *,
                 field_name: Optional[str] = None,
                 detail: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """One evidence reference. Never carries loan-level values (§16)."""
    ref: Dict[str, Any] = {"kind": kind, **snapshot.reference()}
    if field_name:
        ref["canonical_field"] = field_name
    if detail:
        ref.update(dict(detail))
    return ref


def dedupe(values: Sequence[str]) -> Tuple[str, ...]:
    """Order-preserving de-duplication, used for warnings and limitations."""
    seen: List[str] = []
    for value in values:
        if value and value not in seen:
            seen.append(value)
    return tuple(seen)
