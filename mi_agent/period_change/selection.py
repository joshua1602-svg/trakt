"""mi_agent.period_change.selection — the governed field-selection policy.

§10. The Business Semantics Registry tags 106 fields for ``period_change``. An
overview that returned all of them would not be an overview, and a policy hidden
inside the workflow could not be reviewed. So the policy is CONFIGURATION —
``config/period_change_selection.yaml`` — and this module applies it
deterministically, recording every exclusion and its reason.

The selection inputs, in the order the policy applies them:

1. ``workflow_tags`` — an entry not tagged ``period_change`` is never a candidate;
2. ``analytical_role`` — measures become metrics, dimensions become distributions,
   derived inputs and supporting attributes are excluded from overview metrics;
3. asset applicability — an asset-specific entry only applies to a matching book;
4. availability — a field absent from BOTH snapshots is omitted; a field present
   in only one is retained and marked ``not_comparable_due_to_availability``;
5. confidence — below the configured floor, an entry is not an overview metric;
6. existing MI core/extended tier — the product's own view of what is core MI;
7. concept coverage — the overview spreads across the configured concepts;
8. non-duplication — a cap on entries sharing a (concept, aggregation,
   temporality) signature, so one idea is not reported five ways.

Requested-metric mode applies (1)–(4) only: the caller named the field, so the
workflow must analyse THAT field or explain why it cannot — never quietly
substitute another.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Set, Tuple

import yaml

from ..business_semantics import (
    CONFIDENCE_RANK,
    ROLE_DIMENSION,
    ROLE_MEASURE,
    WORKFLOW_PERIOD_CHANGE,
    BusinessSemanticsEntry,
    BusinessSemanticsRegistry,
)
from .models import (
    MODE_CONCEPT,
    MODE_PORTFOLIO_OVERVIEW,
    MODE_REQUESTED_METRIC,
    ExcludedCandidate,
    FieldSelection,
)
from .units import mi_tier

_REPO_ROOT = Path(__file__).resolve().parents[2]
POLICY_PATH = _REPO_ROOT / "config" / "period_change_selection.yaml"

# Exclusion reason codes. Stable strings — they are part of the audit contract.
EXCLUDED_NOT_TAGGED = "not_tagged_for_period_change"
EXCLUDED_ROLE_NOT_ELIGIBLE = "role_not_eligible_for_overview_metric"
EXCLUDED_ASSET_NOT_APPLICABLE = "asset_class_not_applicable"
EXCLUDED_ABSENT_FROM_BOTH = "absent_from_both_snapshots"
EXCLUDED_BELOW_CONFIDENCE = "below_minimum_confidence"
EXCLUDED_STATIC_BASELINE = "static_baseline_excluded_from_ranking"
EXCLUDED_CONCEPT_NOT_IN_SCOPE = "concept_not_in_overview_scope"
EXCLUDED_CONCEPT_CAP = "concept_measure_cap_reached"
EXCLUDED_SIGNATURE_CAP = "duplicate_signature_cap_reached"
EXCLUDED_TOTAL_CAP = "overview_measure_cap_reached"
EXCLUDED_DIMENSION_CAP = "dimension_cap_reached"
EXCLUDED_NOT_IN_REGISTRY = "not_in_business_semantics_registry"
EXCLUDED_NOT_REQUESTED = "not_requested"


@dataclass(frozen=True)
class SelectionPolicy:
    """The loaded, validated selection policy."""

    name: str
    version: str
    workflow_tag: str
    measure_roles: Tuple[str, ...]
    dimension_roles: Tuple[str, ...]
    excluded_overview_roles: Tuple[str, ...]
    excluded_from_ranking: Tuple[str, ...]
    overview_concepts: Tuple[str, ...]
    distribution_concepts: Tuple[str, ...]
    max_measures_per_concept: int
    max_measures_total: int
    max_dimensions: int
    min_confidence: str
    max_measures_per_signature: int
    mi_tier_preference: Tuple[str, ...]
    concept_max_measures: int
    concept_max_dimensions: int
    concept_min_confidence: str
    reserve_one_measure_per_concept: bool = True
    default_max_snapshot_gap_days: Optional[int] = None
    gap_days_by_portfolio: Mapping[str, int] = dc_field(default_factory=dict)
    flow_basis_tolerance_days: int = 5

    def max_snapshot_gap_days(self, context_id: Optional[str] = None
                              ) -> Optional[int]:
        """The gap ceiling for one portfolio, falling back to the default.

        A book on a quarterly cadence legitimately needs a wider ceiling than a
        monthly one, so the value is per-portfolio configuration rather than a
        constant.
        """
        if context_id and context_id in self.gap_days_by_portfolio:
            return int(self.gap_days_by_portfolio[context_id])
        return self.default_max_snapshot_gap_days

    def tier_rank(self, field: str) -> int:
        tier = mi_tier(field)
        if tier and tier in self.mi_tier_preference:
            return self.mi_tier_preference.index(tier)
        return len(self.mi_tier_preference)

    def confidence_ok(self, entry: BusinessSemanticsEntry, floor: str) -> bool:
        return (CONFIDENCE_RANK.get(entry.confidence, 99)
                <= CONFIDENCE_RANK.get(floor, 99))

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "version": self.version,
                "workflow_tag": self.workflow_tag,
                "overview_concepts": list(self.overview_concepts),
                "max_measures_total": self.max_measures_total,
                "max_measures_per_concept": self.max_measures_per_concept,
                "max_measures_per_signature": self.max_measures_per_signature,
                "max_dimensions": self.max_dimensions,
                "min_confidence": self.min_confidence,
                "mi_tier_preference": list(self.mi_tier_preference)}


def _tuple(value: Any) -> Tuple[str, ...]:
    if not value:
        return ()
    return tuple(str(v) for v in value)


def parse_policy(data: Mapping[str, Any]) -> SelectionPolicy:
    policy = data.get("policy") or {}
    roles = data.get("roles") or {}
    temporality = data.get("temporality") or {}
    overview = data.get("overview") or {}
    concept = data.get("concept_mode") or {}
    resolution = data.get("period_resolution") or {}
    gap_default = resolution.get("max_snapshot_gap_days")
    return SelectionPolicy(
        name=str(policy.get("name") or "period_change_overview"),
        version=str(policy.get("version") or "0"),
        workflow_tag=str(policy.get("workflow_tag") or WORKFLOW_PERIOD_CHANGE),
        measure_roles=_tuple(roles.get("measure_roles")) or (ROLE_MEASURE,),
        dimension_roles=_tuple(roles.get("dimension_roles")) or (ROLE_DIMENSION,),
        excluded_overview_roles=_tuple(roles.get("excluded_overview_roles")),
        excluded_from_ranking=_tuple(temporality.get("excluded_from_ranking")),
        overview_concepts=_tuple(overview.get("concepts")),
        distribution_concepts=_tuple(overview.get("distribution_concepts")),
        max_measures_per_concept=int(overview.get("max_measures_per_concept") or 3),
        max_measures_total=int(overview.get("max_measures_total") or 18),
        max_dimensions=int(overview.get("max_dimensions") or 4),
        min_confidence=str(overview.get("min_confidence") or "medium"),
        max_measures_per_signature=int(
            overview.get("max_measures_per_signature") or 1),
        mi_tier_preference=_tuple(overview.get("mi_tier_preference")) or
        ("core", "extended"),
        concept_max_measures=int(concept.get("max_measures") or 12),
        concept_max_dimensions=int(concept.get("max_dimensions") or 4),
        concept_min_confidence=str(concept.get("min_confidence") or "medium"),
        reserve_one_measure_per_concept=bool(
            overview.get("reserve_one_measure_per_concept", True)),
        default_max_snapshot_gap_days=(None if gap_default is None
                                       else int(gap_default)),
        gap_days_by_portfolio={
            str(k): int(v) for k, v in
            (resolution.get("max_snapshot_gap_days_by_portfolio") or {}).items()},
        flow_basis_tolerance_days=int(
            resolution.get("flow_basis_tolerance_days") or 5))


@lru_cache(maxsize=4)
def _load_cached(path_text: str) -> SelectionPolicy:
    path = Path(path_text)
    if not path.exists():
        raise FileNotFoundError(
            f"Period-change selection policy not found at {path}")
    with path.open(encoding="utf-8") as fh:
        return parse_policy(yaml.safe_load(fh) or {})


def load_policy(path: Optional[Path | str] = None) -> SelectionPolicy:
    """The governed selection policy, cached per path."""
    return _load_cached(str(Path(path) if path else POLICY_PATH))


def reset_cache() -> None:
    _load_cached.cache_clear()


# --------------------------------------------------------------------------- #
# Availability
# --------------------------------------------------------------------------- #
AVAILABILITY_BOTH = "both"
AVAILABILITY_START_ONLY = "start_only"
AVAILABILITY_END_ONLY = "end_only"
AVAILABILITY_NEITHER = "neither"


def populated_fields(frame: Any) -> Set[str]:
    """Columns that exist AND carry at least one non-null value.

    A column of nothing but nulls is not an available field: reporting it as
    available produces a metric with no population at either date, which reads
    as a real zero.
    """
    if frame is None:
        return set()
    available: Set[str] = set()
    for column in getattr(frame, "columns", []):
        try:
            if bool(frame[column].notna().any()):
                available.add(str(column))
        except Exception:  # noqa: BLE001 - an unreadable column is simply unavailable
            continue
    return available


def availability(field: str, start_fields: Set[str], end_fields: Set[str]) -> str:
    in_start, in_end = field in start_fields, field in end_fields
    if in_start and in_end:
        return AVAILABILITY_BOTH
    if in_start:
        return AVAILABILITY_START_ONLY
    if in_end:
        return AVAILABILITY_END_ONLY
    return AVAILABILITY_NEITHER


# --------------------------------------------------------------------------- #
# Selection
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SelectionInputs:
    """Everything the policy needs, separated from how it was obtained."""

    mode: str
    registry: BusinessSemanticsRegistry
    start_fields: Set[str]
    end_fields: Set[str]
    asset_classes: Tuple[str, ...] = ()
    requested_fields: Tuple[str, ...] = ()
    requested_concepts: Tuple[str, ...] = ()


def _sort_key(entry: BusinessSemanticsEntry, policy: SelectionPolicy,
              availability_state: str) -> Tuple[Any, ...]:
    """Deterministic candidate ordering inside one concept.

    Comparable first (a field present at both dates outranks one present at a
    single date), then governed confidence, then the product's own MI tier, then
    whether the registry says the field supports materiality assessment, then the
    field name so the order never depends on dictionary iteration.
    """
    return (
        0 if availability_state == AVAILABILITY_BOTH else 1,
        CONFIDENCE_RANK.get(entry.confidence, 99),
        policy.tier_rank(entry.field),
        0 if entry.supports_materiality_assessment else 1,
        entry.field,
    )


def _base_filter(entry: BusinessSemanticsEntry, inputs: SelectionInputs,
                 policy: SelectionPolicy, excluded: List[ExcludedCandidate]
                 ) -> Optional[str]:
    """Apply the filters common to every mode. Returns the availability state."""
    if not entry.has_workflow_tag(policy.workflow_tag):
        excluded.append(ExcludedCandidate(entry.field, EXCLUDED_NOT_TAGGED))
        return None
    if not entry.applies_to_asset(inputs.asset_classes):
        excluded.append(ExcludedCandidate(
            entry.field, EXCLUDED_ASSET_NOT_APPLICABLE,
            f"applies to {', '.join(entry.asset_applicability)}; portfolio is "
            f"{', '.join(inputs.asset_classes) or 'unidentified'}"))
        return None
    state = availability(entry.field, inputs.start_fields, inputs.end_fields)
    if state == AVAILABILITY_NEITHER:
        excluded.append(ExcludedCandidate(entry.field, EXCLUDED_ABSENT_FROM_BOTH))
        return None
    return state


def select_fields(inputs: SelectionInputs, *,
                  policy: Optional[SelectionPolicy] = None) -> FieldSelection:
    """Apply the governed policy and return the selection plus every exclusion."""
    policy = policy or load_policy()
    if inputs.mode == MODE_REQUESTED_METRIC:
        return _select_requested(inputs, policy)
    if inputs.mode == MODE_CONCEPT:
        return _select_concept(inputs, policy)
    return _select_overview(inputs, policy)


def _select_requested(inputs: SelectionInputs, policy: SelectionPolicy
                      ) -> FieldSelection:
    """Requested-metric mode: exactly what was asked for, or an explicit refusal."""
    excluded: List[ExcludedCandidate] = []
    measures: List[str] = []
    dimensions: List[str] = []
    concepts: List[str] = []

    for field in inputs.requested_fields:
        entry = inputs.registry.get(field)
        if entry is None:
            excluded.append(ExcludedCandidate(field, EXCLUDED_NOT_IN_REGISTRY))
            continue
        if _base_filter(entry, inputs, policy, excluded) is None:
            continue
        if entry.analytical_role in policy.dimension_roles:
            dimensions.append(field)
        else:
            # A requested derived_input or supporting_attribute is NOT silently
            # dropped and NOT silently replaced: it is analysed as asked, and the
            # result carries the role so a presenter can caveat it. What §6
            # forbids is such a field appearing as a standalone OVERVIEW metric.
            measures.append(field)
        if entry.analytical_concept:
            concepts.append(entry.analytical_concept)

    return FieldSelection(
        mode=MODE_REQUESTED_METRIC, measures=tuple(measures),
        dimensions=tuple(dimensions), excluded=tuple(excluded),
        concepts_covered=tuple(sorted(set(concepts))),
        policy_name=policy.name, policy_version=policy.version)


def _select_concept(inputs: SelectionInputs, policy: SelectionPolicy
                    ) -> FieldSelection:
    """Concept mode: every eligible entry under the requested concept(s)."""
    wanted = set(inputs.requested_concepts)
    excluded: List[ExcludedCandidate] = []
    measure_pool: List[Tuple[Tuple[Any, ...], str]] = []
    dimension_pool: List[Tuple[Tuple[Any, ...], str]] = []

    for entry in inputs.registry.for_workflow(policy.workflow_tag):
        if entry.analytical_concept not in wanted:
            excluded.append(ExcludedCandidate(
                entry.field, EXCLUDED_CONCEPT_NOT_IN_SCOPE,
                entry.analytical_concept))
            continue
        state = _base_filter(entry, inputs, policy, excluded)
        if state is None:
            continue
        if entry.temporality in policy.excluded_from_ranking:
            excluded.append(ExcludedCandidate(
                entry.field, EXCLUDED_STATIC_BASELINE, entry.temporality))
            continue
        if entry.analytical_role in policy.dimension_roles:
            dimension_pool.append((_sort_key(entry, policy, state), entry.field))
            continue
        if entry.analytical_role in policy.excluded_overview_roles:
            excluded.append(ExcludedCandidate(
                entry.field, EXCLUDED_ROLE_NOT_ELIGIBLE, entry.analytical_role))
            continue
        if not policy.confidence_ok(entry, policy.concept_min_confidence):
            excluded.append(ExcludedCandidate(
                entry.field, EXCLUDED_BELOW_CONFIDENCE, entry.confidence))
            continue
        measure_pool.append((_sort_key(entry, policy, state), entry.field))

    measure_pool.sort()
    dimension_pool.sort()
    measures = [f for _, f in measure_pool[:policy.concept_max_measures]]
    dimensions = [f for _, f in dimension_pool[:policy.concept_max_dimensions]]
    for _, field in measure_pool[policy.concept_max_measures:]:
        excluded.append(ExcludedCandidate(field, EXCLUDED_TOTAL_CAP))
    for _, field in dimension_pool[policy.concept_max_dimensions:]:
        excluded.append(ExcludedCandidate(field, EXCLUDED_DIMENSION_CAP))

    return FieldSelection(
        mode=MODE_CONCEPT, measures=tuple(measures), dimensions=tuple(dimensions),
        excluded=tuple(excluded), concepts_covered=tuple(sorted(wanted)),
        policy_name=policy.name, policy_version=policy.version)


def _select_overview(inputs: SelectionInputs, policy: SelectionPolicy
                     ) -> FieldSelection:
    """Portfolio-overview mode: a governed spread across the policy's concepts."""
    excluded: List[ExcludedCandidate] = []
    per_concept: Dict[str, List[Tuple[Tuple[Any, ...], BusinessSemanticsEntry]]] = {}
    dimension_pool: List[Tuple[Tuple[Any, ...], str]] = []

    for entry in inputs.registry.for_workflow(policy.workflow_tag):
        state = _base_filter(entry, inputs, policy, excluded)
        if state is None:
            continue
        if entry.temporality in policy.excluded_from_ranking:
            excluded.append(ExcludedCandidate(
                entry.field, EXCLUDED_STATIC_BASELINE, entry.temporality))
            continue

        if entry.analytical_role in policy.dimension_roles:
            if entry.analytical_concept not in policy.distribution_concepts:
                excluded.append(ExcludedCandidate(
                    entry.field, EXCLUDED_CONCEPT_NOT_IN_SCOPE,
                    entry.analytical_concept))
                continue
            dimension_pool.append((_sort_key(entry, policy, state), entry.field))
            continue

        if entry.analytical_role in policy.excluded_overview_roles:
            excluded.append(ExcludedCandidate(
                entry.field, EXCLUDED_ROLE_NOT_ELIGIBLE, entry.analytical_role))
            continue
        if entry.analytical_role not in policy.measure_roles:
            excluded.append(ExcludedCandidate(
                entry.field, EXCLUDED_ROLE_NOT_ELIGIBLE, entry.analytical_role))
            continue
        if entry.analytical_concept not in policy.overview_concepts:
            excluded.append(ExcludedCandidate(
                entry.field, EXCLUDED_CONCEPT_NOT_IN_SCOPE,
                entry.analytical_concept))
            continue
        if not policy.confidence_ok(entry, policy.min_confidence):
            excluded.append(ExcludedCandidate(
                entry.field, EXCLUDED_BELOW_CONFIDENCE, entry.confidence))
            continue
        per_concept.setdefault(entry.analytical_concept, []).append(
            (_sort_key(entry, policy, state), entry))

    measures: List[str] = []
    taken_per_concept: Dict[str, int] = {}
    signatures: Dict[Tuple[str, str, str], int] = {}
    deferred: List[Tuple[str, Any]] = []

    def _take(concept: str, entry) -> bool:
        """Select ``entry`` unless a cap forbids it. Records the refusal reason."""
        if len(measures) >= policy.max_measures_total:
            excluded.append(ExcludedCandidate(entry.field, EXCLUDED_TOTAL_CAP))
            return False
        if taken_per_concept.get(concept, 0) >= policy.max_measures_per_concept:
            excluded.append(ExcludedCandidate(entry.field, EXCLUDED_CONCEPT_CAP))
            return False
        signature = (concept, entry.default_aggregation, entry.temporality)
        if signatures.get(signature, 0) >= policy.max_measures_per_signature:
            excluded.append(ExcludedCandidate(
                entry.field, EXCLUDED_SIGNATURE_CAP,
                f"{signature[1]}/{signature[2]}"))
            return False
        signatures[signature] = signatures.get(signature, 0) + 1
        taken_per_concept[concept] = taken_per_concept.get(concept, 0) + 1
        measures.append(entry.field)
        return True

    # Pass 1 — reserve ONE measure for every eligible concept before spending the
    # budget on second and third measures. Without this the concepts listed last
    # (coverage, liquidity) are consumed by the total cap and disappear from the
    # overview entirely, which a reader takes to mean "nothing changed there".
    # Concept order is the policy's order, so the overview always reads the same
    # way: exposure first, then performance, then quality, and so on.
    for concept in policy.overview_concepts:
        candidates = sorted(per_concept.get(concept, []), key=lambda item: item[0])
        if not candidates:
            continue
        if policy.reserve_one_measure_per_concept:
            head, tail = candidates[:1], candidates[1:]
        else:
            head, tail = candidates, []
        for _key, entry in head:
            _take(concept, entry)
        deferred.extend((concept, entry) for _key, entry in tail)

    # Pass 2 — fill the remaining budget in the same concept order.
    for concept, entry in deferred:
        _take(concept, entry)

    covered = [c for c in policy.overview_concepts if taken_per_concept.get(c)]

    dimension_pool.sort()
    dimensions = [f for _, f in dimension_pool[:policy.max_dimensions]]
    for _, field in dimension_pool[policy.max_dimensions:]:
        excluded.append(ExcludedCandidate(field, EXCLUDED_DIMENSION_CAP))

    return FieldSelection(
        mode=MODE_PORTFOLIO_OVERVIEW, measures=tuple(measures),
        dimensions=tuple(dimensions), excluded=tuple(excluded),
        concepts_covered=tuple(covered),
        policy_name=policy.name, policy_version=policy.version)
