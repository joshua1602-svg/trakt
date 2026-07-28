"""mi_agent.period_change.workflow — the governed Period Change Analysis.

The orchestrator. It resolves two snapshots, selects governed fields, calculates
movements, ranks them, builds a deterministic summary and assembles the audit
record. It calculates nothing itself: every number comes from
:mod:`~mi_agent.period_change.calculations`,
:mod:`~mi_agent.period_change.distribution` or
:mod:`~mi_agent.period_change.bridge`.

Two invariants hold throughout:

* **the workflow is the numerical source of truth.** The ``summary`` block is
  generated from the calculated tables; it cannot introduce a fact that is not
  already in ``metric_changes``, ``distribution_changes`` or ``balance_bridge``.
  A presenter turns that structure into prose — it never computes;
* **nothing is asserted that is not measured.** There is no materiality
  threshold, so no movement is called material, significant, a breach or high
  risk. Movements are ranked by observed size and described in the controlled
  significance vocabulary in :mod:`~mi_agent.period_change.models`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field as dc_field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..business_semantics import (
    BusinessSemanticsError,
    BusinessSemanticsRegistry,
    WORKFLOW_PERIOD_CHANGE,
    load_business_semantics,
)
from .bridge import balance_bridge
from .calculations import BALANCE_FIELD, build_pair_context, metric_change
from .distribution import distribution_change
from .models import (
    BRIDGE_STATUS_AVAILABLE,
    CALCULATION_VERSION,
    FAIL_CROSS_TENANT_ACCESS,
    FAIL_NO_ELIGIBLE_FIELDS,
    FAIL_REGISTRY_UNAVAILABLE,
    INTERPRETATION_DETERIORATION,
    INTERPRETATION_IMPROVEMENT,
    MATERIALITY_DISCLAIMER,
    RANK_SCOPE_NOTE,
    MODE_PORTFOLIO_OVERVIEW,
    RESULT_SCHEMA_VERSION,
    SIGNIFICANCE_LARGEST_DECREASE,
    SIGNIFICANCE_LARGEST_INCREASE,
    SIGNIFICANCE_NO_BASIS,
    SIGNIFICANCE_NOTABLE_BY_RANK,
    SIGNIFICANCE_RELATIVELY_STABLE,
    STATUS_NOT_COMPARABLE_DUE_TO_AVAILABILITY,
    STATUS_SOURCE_CONVENTION_UNCERTAIN,
    UNIT_CURRENCY,
    WORKFLOW_ID,
    BalanceBridge,
    DistributionChange,
    FlowBasis,
    MetricChange,
    PeriodChangeFailure,
    PeriodChangeResult,
    PeriodResolution,
    PortfolioScopeRef,
    SnapshotFrame,
    dedupe,
)
from .periods import PeriodRequest, resolve_periods
from .selection import (
    SelectionInputs,
    SelectionPolicy,
    load_policy,
    populated_fields,
    select_fields,
)

logger = logging.getLogger("mi_agent.period_change")

#: How many ranked movements the summary calls out. Ranking is objective;
#: this is only how much of the ranking is repeated in the summary block.
SUMMARY_TOP_N = 3


@dataclass(frozen=True)
class PeriodChangeRequest:
    """One period-change analysis request, already interpreted.

    Produced by :func:`mi_agent.period_change.recognition.recognise` (chat) or
    constructed directly (API / job). ``tenant_id`` and ``authorised_portfolio_ids``
    come from the execution context, never from the question.
    """

    question: str = ""
    mode: str = MODE_PORTFOLIO_OVERVIEW
    period_request: PeriodRequest = dc_field(default_factory=PeriodRequest)
    requested_fields: Tuple[str, ...] = ()
    requested_concepts: Tuple[str, ...] = ()
    scope: PortfolioScopeRef = dc_field(default_factory=PortfolioScopeRef)
    include_bridge: bool = True
    composition_focus: bool = False
    #: Portfolio ids the caller is authorised to see. When set, the requested
    #: scope may only narrow within it — never widen beyond it.
    authorised_portfolio_ids: Tuple[str, ...] = ()
    #: Governed source identifier whose BSR overrides apply, when configured.
    source_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "mode": self.mode,
            "period_request": self.period_request.to_dict(),
            "requested_fields": list(self.requested_fields),
            "requested_concepts": list(self.requested_concepts),
            "include_balance_bridge": self.include_bridge,
            "composition_focus": self.composition_focus,
        }


# --------------------------------------------------------------------------- #
# Governance
# --------------------------------------------------------------------------- #
def check_scope_access(request: PeriodChangeRequest) -> PortfolioScopeRef:
    """Fail closed when the requested scope is not inside the authorised one.

    A caller-supplied portfolio list may NARROW the authorised scope; it may
    never widen it. When the two disagree the request fails rather than being
    silently reduced to the intersection — a caller that asked for a portfolio it
    cannot see must be told, not quietly given a different answer.
    """
    scope = request.scope
    authorised = set(request.authorised_portfolio_ids)
    if not authorised:
        return scope
    requested = tuple(scope.portfolio_ids)
    if not requested:
        return PortfolioScopeRef(
            tenant_id=scope.tenant_id, context_id=scope.context_id,
            label=scope.label, portfolio_ids=tuple(sorted(authorised)),
            asset_classes=scope.asset_classes)
    outside = tuple(p for p in requested if p not in authorised)
    if outside:
        raise PeriodChangeFailure(
            FAIL_CROSS_TENANT_ACCESS,
            detail={"requested_outside_scope": list(outside),
                    "tenant_id": scope.tenant_id})
    return scope


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def run_period_change_analysis(
    request: PeriodChangeRequest,
    snapshots: Sequence[SnapshotFrame], *,
    registry: Optional[BusinessSemanticsRegistry] = None,
    policy: Optional[SelectionPolicy] = None,
) -> PeriodChangeResult:
    """Run the workflow. Raises :class:`PeriodChangeFailure` on an explicit failure."""
    scope = check_scope_access(request)

    if registry is None:
        try:
            registry = load_business_semantics()
        except BusinessSemanticsError as exc:
            raise PeriodChangeFailure(FAIL_REGISTRY_UNAVAILABLE,
                                      detail={"error": str(exc)}) from exc
    registry = registry.for_source(request.source_id)
    policy = policy or load_policy()

    resolution = resolve_periods(
        snapshots, request.period_request, scope=scope,
        max_gap_days=policy.max_snapshot_gap_days(scope.context_id),
        flow_basis_tolerance_days=policy.flow_basis_tolerance_days)
    start_snapshot, end_snapshot = resolution.start_snapshot, resolution.end_snapshot

    selection = select_fields(SelectionInputs(
        mode=request.mode, registry=registry,
        start_fields=populated_fields(start_snapshot.frame),
        end_fields=populated_fields(end_snapshot.frame),
        asset_classes=tuple(scope.asset_classes),
        requested_fields=tuple(request.requested_fields),
        requested_concepts=tuple(request.requested_concepts)),
        policy=policy)

    if not selection.measures and not selection.dimensions:
        raise PeriodChangeFailure(
            FAIL_NO_ELIGIBLE_FIELDS,
            detail={"mode": request.mode,
                    "requested_fields": list(request.requested_fields),
                    "requested_concepts": list(request.requested_concepts),
                    "excluded_count": len(selection.excluded)})

    warnings: List[str] = []
    limitations: List[str] = []

    metrics = _calculate_metrics(registry, selection.measures,
                                 start_snapshot, end_snapshot, warnings,
                                 resolution.flow_basis)
    distributions = _calculate_distributions(registry, selection.dimensions,
                                             start_snapshot, end_snapshot, warnings)
    bridge = _calculate_bridge(request, start_snapshot, end_snapshot, limitations)
    metrics = _rank(metrics, bridge)

    _collect_period_warnings(resolution, warnings)
    _collect_limitations(selection, metrics, limitations)

    summary = build_summary(metrics, distributions, bridge, resolution)
    evidence = _evidence(metrics, distributions, bridge)
    audit = _audit(registry, policy, selection, resolution, request,
                   metrics, distributions, bridge)

    return PeriodChangeResult(
        workflow_id=WORKFLOW_ID,
        result_schema_version=RESULT_SCHEMA_VERSION,
        request_interpretation=request.to_dict(),
        portfolio_scope=scope,
        period_resolution=resolution,
        dataset_provenance=(start_snapshot.reference(), end_snapshot.reference()),
        summary=summary, metric_changes=tuple(metrics),
        distribution_changes=tuple(distributions), balance_bridge=bridge,
        field_selection=selection,
        warnings=dedupe(warnings), limitations=dedupe(limitations),
        evidence=tuple(evidence), audit=audit)


# --------------------------------------------------------------------------- #
# Calculation stages
# --------------------------------------------------------------------------- #
def _calculate_metrics(registry: BusinessSemanticsRegistry,
                       fields: Sequence[str], start: SnapshotFrame,
                       end: SnapshotFrame, warnings: List[str],
                       flow_basis: Optional[FlowBasis] = None
                       ) -> List[MetricChange]:
    out: List[MetricChange] = []
    for name in fields:
        entry = registry.get(name)
        if entry is None:
            continue
        try:
            ctx = build_pair_context(entry, [start.frame, end.frame])
            change = metric_change(entry, start, end, ctx=ctx,
                                   flow_basis=flow_basis)
        except Exception as exc:  # noqa: BLE001 - one bad field never fails the analysis
            logger.warning("period-change metric %s failed: %s", name, exc)
            warnings.append(
                f"{name} could not be calculated and has been omitted.")
            continue
        if change.status == STATUS_SOURCE_CONVENTION_UNCERTAIN:
            warnings.extend(change.notes)
        out.append(change)
    return out


def _calculate_distributions(registry: BusinessSemanticsRegistry,
                             fields: Sequence[str], start: SnapshotFrame,
                             end: SnapshotFrame, warnings: List[str]
                             ) -> List[DistributionChange]:
    out: List[DistributionChange] = []
    for name in fields:
        entry = registry.get(name)
        if entry is None:
            continue
        try:
            out.append(distribution_change(entry, start, end))
        except Exception as exc:  # noqa: BLE001
            logger.warning("period-change distribution %s failed: %s", name, exc)
            warnings.append(
                f"The composition shift for {name} could not be calculated.")
    return out


def _calculate_bridge(request: PeriodChangeRequest, start: SnapshotFrame,
                      end: SnapshotFrame, limitations: List[str]
                      ) -> Optional[BalanceBridge]:
    if not request.include_bridge:
        return None
    try:
        bridge = balance_bridge(start, end)
    except Exception as exc:  # noqa: BLE001
        logger.warning("period-change balance bridge failed: %s", exc)
        limitations.append(
            "The balance bridge could not be calculated and is not reported.")
        return None
    if bridge.limitation:
        limitations.append(bridge.limitation)
    return bridge


# --------------------------------------------------------------------------- #
# Ranking — objective features only (§11)
# --------------------------------------------------------------------------- #
def _rank(metrics: Sequence[MetricChange],
          bridge: Optional[BalanceBridge]) -> List[MetricChange]:
    """Rank comparable movements WITHIN each unit, and label them accordingly.

    Ranking is per ``movement_unit``. A currency movement and a percentage-point
    movement have no common scale: putting them in one sequence made a +0.1%
    balance drift outrank a +15-point arrears rise and be labelled the largest
    observed increase. Each unit therefore has its own rank sequence and its own
    largest increase and decrease. Nothing here decides materiality.

    Within a unit, ordering is by |relative change| where the unit supports one
    (currency, count, ratio) and by |movement| where it does not (percentage
    points) — both are meaningful comparisons inside a single unit.
    """
    import dataclasses

    comparable = [m for m in metrics if m.comparable and m.movement_value is not None]

    ranks: Dict[str, int] = {}
    populations: Dict[str, int] = {}
    largest_increase: Dict[str, str] = {}
    largest_decrease: Dict[str, str] = {}

    for unit in sorted({m.movement_unit for m in comparable}):
        group = [m for m in comparable if m.movement_unit == unit]
        ordered = sorted(
            group,
            key=lambda m: (-abs(m.relative_change) if m.relative_change is not None
                           else -abs(m.movement_value), m.field))
        for index, metric in enumerate(ordered):
            ranks[metric.field] = index + 1
            populations[metric.field] = len(ordered)
        increases = [m for m in ordered if (m.movement_value or 0) > 0]
        decreases = [m for m in ordered if (m.movement_value or 0) < 0]
        if increases:
            largest_increase[unit] = increases[0].field
        if decreases:
            largest_decrease[unit] = decreases[0].field

    total_balance_change = None
    if bridge and bridge.status == BRIDGE_STATUS_AVAILABLE \
            and bridge.opening_balance is not None \
            and bridge.closing_balance is not None:
        total_balance_change = bridge.closing_balance - bridge.opening_balance

    out: List[MetricChange] = []
    for metric in metrics:
        rank = ranks.get(metric.field)
        significance = SIGNIFICANCE_NO_BASIS
        if rank is not None:
            unit = metric.movement_unit
            if largest_increase.get(unit) == metric.field:
                significance = SIGNIFICANCE_LARGEST_INCREASE
            elif largest_decrease.get(unit) == metric.field:
                significance = SIGNIFICANCE_LARGEST_DECREASE
            elif metric.movement_value == 0:
                significance = SIGNIFICANCE_RELATIVELY_STABLE
            elif rank <= SUMMARY_TOP_N:
                significance = SIGNIFICANCE_NOTABLE_BY_RANK
        contribution = None
        if (total_balance_change not in (None, 0)
                and metric.movement_unit == UNIT_CURRENCY
                and metric.movement_value is not None):
            contribution = metric.movement_value / total_balance_change
        out.append(dataclasses.replace(
            metric, movement_rank=rank, significance=significance,
            rank_population=populations.get(metric.field),
            contribution_to_balance_change=contribution))
    return out


# --------------------------------------------------------------------------- #
# Warnings and limitations
# --------------------------------------------------------------------------- #
def _collect_period_warnings(resolution: PeriodResolution,
                             warnings: List[str]) -> None:
    warnings.extend(resolution.adjustment_notes)
    if resolution.any_adjusted and not resolution.adjustment_notes:
        warnings.append(
            "At least one requested period was adjusted to the nearest "
            "available governed snapshot.")


def _collect_limitations(selection, metrics: Sequence[MetricChange],
                         limitations: List[str]) -> None:
    limitations.append(MATERIALITY_DISCLAIMER)
    single_date = [m.field for m in metrics
                   if m.status == STATUS_NOT_COMPARABLE_DUE_TO_AVAILABILITY]
    if single_date:
        limitations.append(
            f"{len(single_date)} field(s) are reported at only one of the two "
            f"dates and are marked not comparable: {', '.join(single_date[:5])}.")
    if selection.mode == MODE_PORTFOLIO_OVERVIEW:
        limitations.append(
            f"This overview reports a governed subset of the registry selected "
            f"by policy {selection.policy_name} v{selection.policy_version}; it "
            f"is not every period-change field in the registry.")


# --------------------------------------------------------------------------- #
# Deterministic summary — generated FROM the tables, never beyond them
# --------------------------------------------------------------------------- #
def build_summary(metrics: Sequence[MetricChange],
                  distributions: Sequence[DistributionChange],
                  bridge: Optional[BalanceBridge],
                  resolution: PeriodResolution) -> Dict[str, Any]:
    """The structured summary. Every value here also appears in a table above."""
    comparable = [m for m in metrics if m.comparable]
    improvements = [m.field for m in comparable
                    if m.interpretation == INTERPRETATION_IMPROVEMENT]
    deteriorations = [m.field for m in comparable
                      if m.interpretation == INTERPRETATION_DETERIORATION]
    def _headline(metric: MetricChange) -> Dict[str, Any]:
        return {
            "canonical_field": metric.field,
            "display_name": metric.display_name,
            "analytical_concept": metric.analytical_concept,
            "start_value": metric.start_value,
            "end_value": metric.end_value,
            "movement_value": metric.movement_value,
            "movement_unit": metric.movement_unit,
            "relative_change": metric.relative_change,
            "interpretation": metric.interpretation,
            "significance": metric.significance,
            "movement_rank": metric.movement_rank,
            "movement_rank_scope": metric.movement_unit,
            "rank_population": metric.rank_population,
        }

    # Top movements are reported PER UNIT. A single flattened list would put a
    # currency movement above a percentage-point one purely because currency
    # sorts first, reintroducing the cross-unit comparison this avoids.
    units = sorted({m.movement_unit for m in comparable if m.movement_rank})
    top_by_unit = {
        unit: [_headline(m) for m in sorted(
            (x for x in comparable
             if x.movement_unit == unit and x.movement_rank),
            key=lambda x: x.movement_rank)[:SUMMARY_TOP_N]]
        for unit in units
    }
    largest_increases = [_headline(m) for m in metrics
                         if m.significance == SIGNIFICANCE_LARGEST_INCREASE]
    largest_decreases = [_headline(m) for m in metrics
                         if m.significance == SIGNIFICANCE_LARGEST_DECREASE]

    composition: List[Dict[str, Any]] = []
    for dist in distributions:
        for category in dist.categories:
            if category.count_share_movement is None:
                continue
            composition.append({
                "canonical_field": dist.field,
                "display_name": dist.display_name,
                "category": category.category,
                "count_share_movement": category.count_share_movement,
                "balance_share_movement": category.balance_share_movement,
            })
    composition.sort(key=lambda row: (-abs(row["count_share_movement"] or 0.0),
                                      row["canonical_field"], row["category"]))

    return {
        "period": {
            "start": resolution.start_snapshot.reporting_date
            or resolution.start_snapshot.snapshot_id,
            "end": resolution.end_snapshot.reporting_date
            or resolution.end_snapshot.snapshot_id,
            "resolution_method": resolution.resolution_method,
            "adjusted": resolution.any_adjusted,
        },
        "metrics_analysed": len(metrics),
        "metrics_comparable": len(comparable),
        "distributions_analysed": len(distributions),
        "improvements": improvements,
        "deteriorations": deteriorations,
        "not_assessed": [m.field for m in comparable
                         if m.interpretation not in (INTERPRETATION_IMPROVEMENT,
                                                     INTERPRETATION_DETERIORATION)],
        # Keyed by unit — never one cross-unit ordering. See RANK_SCOPE_NOTE.
        "top_movements_by_unit": top_by_unit,
        "largest_observed_increases": largest_increases,
        "largest_observed_decreases": largest_decreases,
        "rank_scope": RANK_SCOPE_NOTE,
        "largest_composition_shifts": composition[:SUMMARY_TOP_N],
        "balance_bridge": (
            {"status": bridge.status,
             "opening_balance": bridge.opening_balance,
             "closing_balance": bridge.closing_balance,
             "new_loan_closing_balance": bridge.new_loan_balance,
             "exited_loan_opening_balance": bridge.exited_loan_balance,
             "movement_on_continuing_loans": bridge.continuing_movement,
             "reconciles": bridge.reconciles}
            if bridge else None),
        "materiality": MATERIALITY_DISCLAIMER,
    }


# --------------------------------------------------------------------------- #
# Evidence and audit — §16
# --------------------------------------------------------------------------- #
def _evidence(metrics: Sequence[MetricChange],
              distributions: Sequence[DistributionChange],
              bridge: Optional[BalanceBridge]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for metric in metrics:
        out.extend(dict(e) for e in metric.evidence)
    for dist in distributions:
        out.extend(dict(e) for e in dist.evidence)
    if bridge:
        out.extend(dict(e) for e in bridge.evidence)
    return out


def _audit(registry: BusinessSemanticsRegistry, policy: SelectionPolicy,
           selection, resolution: PeriodResolution,
           request: PeriodChangeRequest, metrics: Sequence[MetricChange],
           distributions: Sequence[DistributionChange],
           bridge: Optional[BalanceBridge]) -> Dict[str, Any]:
    """Everything needed to reproduce the result. No loan-level data (§16)."""
    return {
        "workflow_id": WORKFLOW_ID,
        "calculation_version": CALCULATION_VERSION,
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "business_semantics": registry.to_provenance(),
        "selection_policy": policy.to_dict(),
        "selected_entries": [
            {"canonical_field": m.field,
             "analytical_role": m.analytical_role,
             "temporality": m.temporality,
             "aggregation": m.aggregation,
             "weight_field": m.weight_field,
             "share_basis": m.share_basis,
             "movement_basis": m.movement_basis,
             "movement_unit": m.movement_unit,
             "numerator": {"start": m.start.numerator, "end": m.end.numerator},
             "denominator": {"start": m.start.denominator,
                             "end": m.end.denominator},
             "valid_rows": {"start": m.start.valid_population,
                            "end": m.end.valid_population},
             "invalid_rows": {"start": m.start.excluded_population,
                              "end": m.end.excluded_population},
             "status": m.status,
             "confidence": m.confidence}
            for m in metrics
        ] + [
            {"canonical_field": d.field, "analytical_role": "dimension",
             "aggregation": "distribution", "status": d.status,
             "balance_field": d.balance_field, "confidence": d.confidence}
            for d in distributions
        ],
        "excluded_candidates": [e.to_dict() for e in selection.excluded],
        "resolved_snapshots": {
            "start": resolution.start_snapshot.reference(),
            "end": resolution.end_snapshot.reference()},
        "filters": {
            "portfolio_ids": list(resolution.portfolio_scope.portfolio_ids),
            "context_id": resolution.portfolio_scope.context_id,
            "asset_classes": list(resolution.portfolio_scope.asset_classes)},
        "period_resolution_method": resolution.resolution_method,
        "balance_bridge_status": bridge.status if bridge else None,
        "rounding": {"balance_bridge_tolerance":
                     bridge.tolerance if bridge else None},
        "data_source_identifiers": [
            resolution.start_snapshot.dataset_reference
            or resolution.start_snapshot.dataset_label,
            resolution.end_snapshot.dataset_reference
            or resolution.end_snapshot.dataset_label],
        "workflow_tag": WORKFLOW_PERIOD_CHANGE,
        "balance_field": BALANCE_FIELD,
    }
