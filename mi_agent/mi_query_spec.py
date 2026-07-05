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
from typing import Any, Dict, List, Optional

# Allowed enumerations (kept as module constants so the validator and parser
# can share them).
INTENTS = {"chart", "table", "summary"}
CHART_TYPES = {"bar", "line", "scatter", "bubble", "heatmap", "treemap", "none"}
AGGREGATIONS = {
    "sum", "avg", "weighted_avg", "count", "count_distinct",
    "median", "distribution", "loan_level", "balance_sum",
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
        if "filters" in kwargs and kwargs["filters"] is None:
            kwargs["filters"] = {}
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
        # filter keys are field references too
        for key in self.filters or {}:
            if key:
                out.append(key)
        # de-duplicate, preserve order
        seen = set()
        unique = []
        for f in out:
            if f not in seen:
                seen.add(f)
                unique.append(f)
        return unique
