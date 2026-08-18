#!/usr/bin/env python3
"""
mi_query_spec.py

v1 MIQuerySpec — a small, serialisable description of an MI query / chart
request.  Implemented with the standard library `dataclasses` (no pydantic
dependency) so it is safe to import anywhere in the repo.

A spec is purely declarative: it names *semantic field keys* (keys in
mi_semantics_field_registry.yaml) and how they should be combined.  It does
NOT contain data and never executes anything.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping, Optional

# Allowed enumerations (kept as module constants so the validator and parser
# can share them).
INTENTS = {"chart", "table", "summary"}
CHART_TYPES = {"bar", "line", "scatter", "bubble", "heatmap", "treemap", "none"}
AGGREGATIONS = {
    "sum", "avg", "weighted_avg", "count", "count_distinct",
    "median", "distribution", "loan_level", "balance_sum",
    # P1A — a filtered population expressed as a share of the whole book.
    # Distinct from the aggregations above because it needs TWO populations.
    "share",
    # P1D — each group's CONTRIBUTION to a portfolio weighted aggregate
    # (share of weight x the group's own value). Distinct from a per-group
    # weighted average because a small group with a high value contributes
    # little, and ranking the two the same way answers a different question.
    "contribution",
}
OUTPUT_FORMATS = {"chart", "table", "text", "chart_and_table"}

# --------------------------------------------------------------------------- #
# Phase 7 — MIQuerySpec v2 controlled vocabularies (shared with the validator
# and the LLM interpretation contract). All additive.
# --------------------------------------------------------------------------- #
SPEC_VERSION = "2.0"
EXECUTION_MODES = {"flat", "snapshot", "state", "temporal", "risk"}
STATES = {
    "total_funded", "total_pipeline", "total_forecast_funded",
    "cohort_by_date", "cohort_by_portfolio", "cohort_by_spv",
    "cohort_by_acquired_portfolio",
    # descriptive cohort aliases (resolve to cohort_by_date at the runtime)
    "cohort_by_origination_date", "cohort_by_funding_date",
    "cohort_by_acquisition_date",
}
TEMPORAL_MODES = {"latest", "as_of", "compare", "trend"}
RISK_MONITOR_MODES = {"migration", "concentration", "trajectory", "flags"}
BUCKET_STRATEGIES = {"configured", "quantile", "none"}
TREND_GRAINS = {"daily", "weekly", "monthly", "quarterly"}
FORECAST_PROBABILITY_SOURCES = {"row", "config", "explicit_balance"}
OUTPUT_TYPES = {"table", "chart", "both"}
SEGMENTS = {"portfolio", "spv", "acquired_portfolio"}

# Bare, ambiguous natural-language concepts that must be RESOLVED to a concrete
# field before reaching a spec (see the interpretation contract). A spec whose
# dimension is one of these is rejected by the validator.
AMBIGUOUS_DIMENSION_TERMS = {"stage", "portfolio", "region", "rate", "balance"}

# Semantic-field-bearing fields on the spec (used by referenced_fields and the
# validator).  `dimensions`/`hierarchy` are list-valued and handled separately.
_SCALAR_FIELD_SLOTS = ("metric", "dimension", "x", "y", "size", "color", "weight_field", "sort_by")
_LIST_FIELD_SLOTS = ("dimensions", "hierarchy")

# --------------------------------------------------------------------------- #
# Filter normalisation
# --------------------------------------------------------------------------- #
# ``filters`` is canonically ``{field_key: condition}`` (see
# mi_query_executor._apply_filters). Callers that build a spec from JSON — the
# LLM parser above all — routinely express the same intent as a LIST of predicate
# objects, or under a singular ``filter`` key, because that is the more natural
# JSON shape and nothing in the contract ruled it out. Both shapes previously
# reached the dataclass unconverted: a list made ``referenced_fields`` raise
# ``TypeError: unhashable type: 'dict'``, and a singular ``filter`` key was
# dropped as unknown — so a correctly-identified predicate was either a crash or
# a silently unfiltered answer.
#
# These helpers fold every recognised shape into the canonical mapping. A
# predicate that cannot be folded is NEVER discarded: it is recorded in
# ``unavailable_filters``, which the workflow already surfaces in warnings and
# the query-audit panel.

#: Keys a predicate object may use to name its field, in precedence order.
_PREDICATE_FIELD_KEYS = ("field", "field_key", "key", "column", "name")
#: Keys a predicate object may use to name its comparison operator.
_PREDICATE_OP_KEYS = ("operator", "op", "comparator", "comparison")
#: Keys a predicate object may use to carry its value.
_PREDICATE_VALUE_KEYS = ("value", "values", "val")
#: Operator spellings that mean plain equality/membership, for which the
#: canonical condition is the bare value rather than an ``{"op", "value"}`` dict.
_EQUALITY_OPS = {"", "=", "==", "eq", "equal", "equals", "equal_to", "equalto",
                 "is", "is_equal_to", "in", "one_of"}


def _predicate_to_condition(pred: Dict[str, Any]) -> Optional[tuple]:
    """``{field, operator, value}`` -> ``(field_key, condition)``, or None.

    Returns None when the object carries no usable field name, so the caller can
    record it as unavailable rather than guess which column it meant.
    """
    field_key = None
    for key in _PREDICATE_FIELD_KEYS:
        candidate = pred.get(key)
        if isinstance(candidate, str) and candidate.strip():
            field_key = candidate.strip()
            break
    if field_key is None:
        return None

    operator = ""
    for key in _PREDICATE_OP_KEYS:
        candidate = pred.get(key)
        if isinstance(candidate, str) and candidate.strip():
            operator = candidate.strip().lower()
            break

    has_value = any(k in pred for k in _PREDICATE_VALUE_KEYS)
    value: Any = None
    for key in _PREDICATE_VALUE_KEYS:
        if key in pred:
            value = pred[key]
            break
    # ``between`` is also written as separate bounds.
    if not has_value and ("min" in pred or "max" in pred):
        lo, hi = pred.get("min"), pred.get("max")
        if lo is not None and hi is not None:
            return field_key, {"op": "between", "value": [lo, hi]}
        return field_key, {"op": "ge" if lo is not None else "le",
                           "value": lo if lo is not None else hi}
    if not has_value:
        return None

    # Plain equality/membership is expressed as the bare value: the executor
    # matches a string case-insensitively and a list as membership, which is what
    # a categorical predicate means.
    if operator in _EQUALITY_OPS:
        if isinstance(value, (list, tuple, set)):
            return field_key, list(value)
        return field_key, value
    return field_key, {"op": operator, "value": value}


def _describe_unfoldable(pred: Any) -> str:
    """A short, user-facing note for a predicate that could not be folded."""
    if isinstance(pred, dict):
        parts = [f"{k}={v!r}" for k, v in list(pred.items())[:3]]
        body = ", ".join(parts) if parts else "empty predicate"
    else:
        body = repr(pred)[:80]
    return (f"a filter was requested ({body}) but could not be interpreted, "
            "so it was NOT applied")


#: The most measures one question may carry. A management answer with more than
#: four figures stops being readable, and an unbounded set would let a vague
#: question quietly become an expensive one. A request beyond this is reported,
#: never silently truncated.
MAX_MEASURES = 4


def normalise_measures(raw: Any, *, metric: Optional[str] = None,
                       aggregation: Optional[str] = None,
                       weight_field: Optional[str] = None) -> tuple:
    """``(measures, notes)`` for any recognised measure shape.

    Accepts the canonical list of ``{"field", "aggregation"}`` mappings, a list
    of bare field-name strings, and the singular ``metric``/``aggregation``
    pair. Never raises: anything unrecognised is reported through the notes so a
    caller surfaces it instead of dropping it — the same discipline
    ``normalise_filters`` follows.

    Duplicates are folded: "balance and total balance" resolve to one governed
    measure and must be reported once, not twice.
    """
    notes: List[str] = []
    out: List[Dict[str, Any]] = []
    seen: set = set()

    def _add(field_name: Any, agg: Any = None, weight: Any = None) -> None:
        name = str(field_name or "").strip()
        if not name:
            return
        key = (name, str(agg or "").strip().lower() or None)
        if key in seen or any(m["field"] == name and not agg for m in out):
            return
        seen.add(key)
        entry: Dict[str, Any] = {"field": name}
        if agg:
            entry["aggregation"] = str(agg).strip().lower()
        if weight:
            entry["weight_field"] = str(weight).strip()
        out.append(entry)

    items: Any = raw
    if isinstance(items, (str, Mapping)):
        items = [items]
    if items is None:
        items = []
    if not isinstance(items, (list, tuple)):
        notes.append(f"measure specification {type(raw).__name__} was not "
                     f"recognised and has not been applied")
        items = []

    for item in items:
        if isinstance(item, str):
            _add(item)
        elif isinstance(item, Mapping):
            name = (item.get("field") or item.get("metric")
                    or item.get("name") or item.get("measure"))
            if not name:
                notes.append("a requested measure carried no field name and has "
                             "not been applied")
                continue
            _add(name, item.get("aggregation") or item.get("agg"),
                 item.get("weight_field") or item.get("weight"))
        else:
            notes.append(f"a requested measure of type {type(item).__name__} "
                         f"was not recognised and has not been applied")

    if not out and metric:
        _add(metric, aggregation, weight_field)
    return out, notes


def normalise_filters(raw: Any) -> tuple:
    """``(filters_dict, unavailable_notes)`` for any recognised filter shape.

    Accepts the canonical mapping, a list of predicate objects, and a single
    predicate object. Never raises: anything unrecognised is reported through the
    notes so the caller can surface it instead of dropping it.
    """
    notes: List[str] = []
    if raw is None:
        return {}, notes
    if isinstance(raw, dict):
        # A lone predicate object — ``{"field": …, "operator": …, "value": …}`` —
        # is a predicate, not a one-entry mapping of field to condition.
        if any(k in raw for k in _PREDICATE_FIELD_KEYS) and \
                any(k in raw for k in _PREDICATE_OP_KEYS + _PREDICATE_VALUE_KEYS):
            folded = _predicate_to_condition(raw)
            if folded is None:
                notes.append(_describe_unfoldable(raw))
                return {}, notes
            return {folded[0]: folded[1]}, notes
        # Canonical mapping. Keys must be usable as field references.
        out: Dict[str, Any] = {}
        for key, value in raw.items():
            if isinstance(key, str) and key.strip():
                out[key] = value
            else:
                notes.append(_describe_unfoldable({key: value}))
        return out, notes
    if isinstance(raw, (list, tuple)):
        out = {}
        for item in raw:
            if not isinstance(item, dict):
                notes.append(_describe_unfoldable(item))
                continue
            folded = _predicate_to_condition(item)
            if folded is None:
                notes.append(_describe_unfoldable(item))
                continue
            out[folded[0]] = folded[1]
        return out, notes
    notes.append(_describe_unfoldable(raw))
    return {}, notes


@dataclass
class MIQuerySpec:
    """v1 specification for an MI query/chart request."""

    intent: str = "chart"                       # chart | table | summary
    chart_type: str = "none"                    # bar|line|scatter|bubble|heatmap|treemap|none

    # Semantic field keys (all optional; which are required depends on chart_type)
    metric: Optional[str] = None
    dimension: Optional[str] = None
    x: Optional[str] = None
    y: Optional[str] = None
    size: Optional[str] = None
    color: Optional[str] = None

    aggregation: str = "count"                  # see AGGREGATIONS
    weight_field: Optional[str] = None

    # P1E — MULTI-MEASURE COMPOSITION.
    #
    # A CFO asks one analytical question containing several governed measures
    # ("balance, loan count, WA LTV and WA rate for the London book"), not four
    # unrelated questions. ``measures`` is that set: an ordered list of
    # ``{"field", "aggregation", "weight_field"}`` entries sharing ONE
    # population, ONE filter set, ONE optional grouping and ONE reporting period.
    #
    # Backward compatibility is by NORMALISATION, not by a second code path:
    # ``normalise_measures`` folds the singular ``metric``/``aggregation`` into a
    # one-element list, and keeps ``metric``/``aggregation`` pointing at
    # ``measures[0]``. Every existing single-metric query therefore produces
    # exactly the spec it produced before, and the executor has one contract to
    # consume rather than two.
    measures: List[Dict[str, Any]] = field(default_factory=list)

    filters: Dict[str, Any] = field(default_factory=dict)
    top_n: Optional[int] = None

    # Ranking / "largest" grammar (ADDITIVE; all optional, no-op by default).
    #   ranking_mode : "loan_level" (top loans table) | "grouped" (ranked bar) | None
    #   sort_by      : semantic field key to rank by (falls back to ``metric``)
    #   sort_direction: "desc" (default) | "asc"
    #   limit        : loan-level row cap for ranked tables (falls back to top_n/10)
    ranking_mode: Optional[str] = None
    sort_by: Optional[str] = None
    sort_direction: str = "desc"
    limit: Optional[int] = None

    # Multi-dimension support (heatmap / treemap)
    dimensions: List[str] = field(default_factory=list)
    hierarchy: List[str] = field(default_factory=list)

    title: Optional[str] = None
    explanation: Optional[str] = None
    output_format: str = "chart"                # chart | table | text | chart_and_table

    # ------------------------------------------------------------------ #
    # Phase 6 — runtime integration fields (ADDITIVE; all optional).
    # Existing flat single-CSV queries leave these defaulted and behave
    # exactly as before. They are NOT semantic-field slots and do not affect
    # referenced_fields()/validation of the v1 chart structure.
    # ------------------------------------------------------------------ #
    route_id: str = "mi"                         # mi | mna | regulatory_annex2 | ...
    execution_mode: Optional[str] = None         # flat|snapshot|state|temporal|risk
    state: Optional[str] = None                  # state-library key (e.g. total_funded)
    temporal_mode: Optional[str] = None          # latest|as_of|compare|trend
    as_of_date: Optional[str] = None
    baseline_date: Optional[str] = None
    current_date: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    segment: Optional[str] = None                # portfolio|spv|acquired_portfolio
    risk_monitor: Optional[Any] = None           # bool | migration|concentration|trajectory
    snapshot_client_id: Optional[str] = None
    snapshot_store_root: Optional[str] = None    # local/dev/test only

    # ------------------------------------------------------------------ #
    # Phase 7 — MIQuerySpec v2 expansion (ADDITIVE; all optional/defaulted).
    # The single governed contract for BOTH the flat path and the snapshot/
    # state/temporal/risk runtime path. Convenience fields are normalised onto
    # the canonical runtime fields by ``normalized()`` so run_mi_query is
    # unchanged.
    # ------------------------------------------------------------------ #
    query_id: Optional[str] = None

    # State
    state_filters: Dict[str, Any] = field(default_factory=dict)

    # Snapshot metadata / segmentation keys
    reporting_date: Optional[str] = None
    cut_off_date: Optional[str] = None
    portfolio_id: Optional[str] = None
    spv_id: Optional[str] = None
    acquired_portfolio_id: Optional[str] = None

    # Source-portfolio lens (total | direct | acquired | cohort) — set by the
    # portfolio_lens resolver; carried into output metadata/titles.
    portfolio_lens: Optional[Dict[str, Any]] = None

    # Temporal
    comparison_basis: Optional[str] = None
    trend_grain: Optional[str] = None            # daily|weekly|monthly|quarterly

    # Forecast
    forecast_mode: Optional[str] = None
    forecast_probability_source: Optional[str] = None  # row|config|explicit_balance
    allow_config_probability: Optional[bool] = None

    # Balance bridge (attribution waterfall between two funded periods)
    bridge_query: bool = False
    bridge_dimension: Optional[str] = None       # semantic dimension key to attribute by

    # Cohort progression (static-pool seasoning across reporting periods)
    cohort_progression: bool = False
    cohort_vintage: Optional[str] = None         # origination vintage (e.g. 2023 / 2023-Q2)
    cohort_grain: Optional[str] = None           # Y|Q|M vintage grain

    # Risk monitor
    risk_monitor_mode: Optional[str] = None      # migration|concentration|trajectory|flags
    migration_dimension: Optional[str] = None
    concentration_dimension: Optional[str] = None
    risk_dimension: Optional[str] = None
    baseline_risk_field: Optional[str] = None
    current_risk_field: Optional[str] = None

    # Buckets
    bucket_strategy: Optional[str] = None        # configured|quantile|none
    bucket_count: int = 4
    bucket_field: Optional[str] = None
    bucket_config_key: Optional[str] = None

    # Chart / output
    output_type: Optional[str] = None            # table|chart|both
    chart_preference: Optional[str] = None
    allow_chart_fallback: Optional[bool] = None

    # Governance
    require_structured_issues: bool = True
    allow_partial_result: Optional[bool] = None
    strict_mode: Optional[bool] = None

    # ------------------------------------------------------------------ #
    # ERE securitisation sprint — analytical intents (ADDITIVE; optional).
    # These are NOT semantic-field slots: they carry period tokens, target
    # values and a question-kind, never registry field names, so they are
    # excluded from referenced_fields()/validation. They mark a governed
    # temporal-compare, forecast-extrapolation or risk-limit plan that the
    # runtime / API layer resolves against evolution / risk-monitor data.
    # ------------------------------------------------------------------ #
    compare_periods: List[str] = field(default_factory=list)
    forecast_question: Optional[str] = None      # reach_threshold|run_rate|run_rate_annualised|scenario|conversion|pipeline_needed|compare_models|extrapolation_curve
    forecast_target_value: Optional[float] = None
    risk_limit_query: Optional[bool] = None
    # Risk-limit category filter ("geographic concentration limits" -> only the
    # geographic category). None = all categories.
    risk_limit_category: Optional[str] = None
    # Human-readable predicates the user asked for that COULD NOT be applied
    # (e.g. a joint-borrower filter when no borrower-structure field exists). These
    # are surfaced in warnings + the query-audit panel and NEVER silently dropped.
    unavailable_filters: List[str] = field(default_factory=list)

    # ------------------------------------------------------------------ #
    # Serialisation
    # ------------------------------------------------------------------ #
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=False)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MIQuerySpec":
        if not isinstance(data, dict):
            raise TypeError("MIQuerySpec.from_dict expects a dict")
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        kwargs = {k: v for k, v in data.items() if k in known}
        # Defensive normalisation of list/dict slots.
        for slot in _LIST_FIELD_SLOTS:
            if slot in kwargs and kwargs[slot] is None:
                kwargs[slot] = []
        # ``filters`` accepts the canonical mapping, a list of predicate objects
        # or a single predicate object; a producer that wrote the singular
        # ``filter`` key meant the same thing. Fold them all, and carry anything
        # unfoldable into unavailable_filters rather than dropping it. Without
        # this, a list reached referenced_fields() and raised TypeError, and a
        # singular ``filter`` was discarded as an unknown key — both of which
        # turned a correctly-identified predicate into a wrong answer.
        raw_filters = kwargs.get("filters")
        if not raw_filters and isinstance(data.get("filter"), (dict, list, tuple)):
            raw_filters = data["filter"]
        folded, notes = normalise_filters(raw_filters)
        kwargs["filters"] = folded
        # P1E: fold the measure set, accepting the singular metric/aggregation
        # as a one-element list so the two shapes are one contract downstream.
        measures, measure_notes = normalise_measures(
            kwargs.get("measures") if kwargs.get("measures") is not None
            else data.get("measure"),
            metric=kwargs.get("metric"), aggregation=kwargs.get("aggregation"),
            weight_field=kwargs.get("weight_field"))
        kwargs["measures"] = measures
        notes = list(notes) + measure_notes
        if notes:
            existing = list(kwargs.get("unavailable_filters") or [])
            kwargs["unavailable_filters"] = existing + [
                n for n in notes if n not in existing]
        return cls(**kwargs)

    @classmethod
    def from_json(cls, text: str) -> "MIQuerySpec":
        return cls.from_dict(json.loads(text))

    # ------------------------------------------------------------------ #
    # Phase 7 — mode inference & normalisation (keeps run_mi_query unchanged)
    # ------------------------------------------------------------------ #
    def effective_execution_mode(self) -> str:
        """Infer the execution mode (mirrors mi_runtime.infer_execution_mode)."""
        if self.execution_mode:
            return self.execution_mode
        if self.risk_monitor or self.risk_monitor_mode:
            return "risk"
        if self.temporal_mode in ("compare", "trend"):
            return "temporal"
        if self.state:
            return "state"
        return "flat"

    def effective_risk_dimension(self) -> Optional[str]:
        """Resolve the dimension a risk query should group/migrate on."""
        mode = self.risk_monitor_mode
        if mode == "migration" and self.migration_dimension:
            return self.migration_dimension
        if mode == "concentration" and self.concentration_dimension:
            return self.concentration_dimension
        return self.risk_dimension or self.dimension

    def normalized(self) -> "MIQuerySpec":
        """Return a copy with v2 convenience fields mapped onto the canonical
        runtime fields (risk_monitor / dimension / as_of_date / execution_mode).

        Idempotent for v1/Phase-6 specs, so ``run_mi_query`` is unaffected.
        """
        import dataclasses as _dc
        out = _dc.replace(self)
        # risk_monitor_mode -> risk_monitor (+ dimension)
        if out.risk_monitor_mode and not out.risk_monitor:
            out.risk_monitor = out.risk_monitor_mode
        if (out.risk_monitor or out.risk_monitor_mode) and not out.dimension:
            rdim = self.effective_risk_dimension()
            if rdim:
                out.dimension = rdim
        # reporting_date -> as_of_date for as_of temporal selection
        if out.temporal_mode == "as_of" and not out.as_of_date and out.reporting_date:
            out.as_of_date = out.reporting_date
        # fill execution_mode for downstream clarity
        if not out.execution_mode:
            out.execution_mode = out.effective_execution_mode()
        return out

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #
    def referenced_fields(self) -> List[str]:
        """Return every semantic field key referenced by this spec (unique, ordered)."""
        out: List[str] = []
        for slot in _SCALAR_FIELD_SLOTS:
            value = getattr(self, slot)
            if value:
                out.append(value)
        for slot in _LIST_FIELD_SLOTS:
            for value in getattr(self, slot) or []:
                if value:
                    out.append(value)
        # P1E: every measure in the set is a field reference, so validation sees
        # the whole request rather than only its first measure. ``loan_count``
        # is a governed count, not a semantic field, and is excluded.
        for measure in self.measures or []:
            if not isinstance(measure, dict):
                continue
            name = measure.get("field")
            if name and name not in ("loan_count", "count"):
                out.append(name)
        # filter keys are field references too. Defensive about the container and
        # its keys: a spec built by a producer that bypassed from_dict (a direct
        # constructor call, an older pickle) may still carry a non-mapping here,
        # and introspection must never be the thing that raises.
        filters = self.filters
        if isinstance(filters, dict):
            keys: List[Any] = list(filters.keys())
        elif isinstance(filters, (list, tuple)):
            keys = [f.get("field") if isinstance(f, dict) else f for f in filters]
        else:
            keys = []
        for key in keys:
            if isinstance(key, str) and key:
                out.append(key)
        # de-duplicate, preserve order. Only hashable, string-valued references
        # reach here, so membership testing is safe.
        seen = set()
        unique = []
        for f in out:
            if not isinstance(f, str):
                continue
            if f not in seen:
                seen.add(f)
                unique.append(f)
        return unique
