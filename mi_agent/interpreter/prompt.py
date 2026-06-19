"""mi_agent.interpreter.prompt — constrained Claude prompt for MIQuerySpec v2.

Phase 8B. Builds a single, controlled prompt instructing Claude to return JSON
only — one MIQuerySpec-v2 object, or a clarification object — and never code,
pandas, SQL, computed results, invented fields, or chart types outside the
governed library. Built entirely from code constants (no markdown read at
runtime).
"""

from __future__ import annotations

import json
from typing import Optional

from mi_agent.mi_query_spec import (
    AGGREGATIONS,
    BUCKET_STRATEGIES,
    CHART_TYPES,
    EXECUTION_MODES,
    FORECAST_PROBABILITY_SOURCES,
    MIQuerySpec,
    OUTPUT_TYPES,
    RISK_MONITOR_MODES,
    SEGMENTS,
    STATES,
    TEMPORAL_MODES,
    TREND_GRAINS,
)

# Known route ids (mirror config/routes/*.yaml stems).
ROUTE_IDS = ("mi", "mna", "regulatory_annex2", "regulatory_and_mi")

# Allowed top-level spec fields the model may emit (+ clarification keys).
ALLOWED_SPEC_FIELDS = tuple(MIQuerySpec().__dataclass_fields__.keys())
CLARIFICATION_FIELDS = ("clarification_required", "clarification_question")

_DEFAULT_RESOLUTION_RULES = """
Default resolution rules (PREFER THESE DEFAULTS OVER CLARIFYING — for the
controlled MI route these are unambiguous and must NOT trigger a clarification):
- State defaulting. "show total funded" / "total funded" -> execution_mode
  "state", state "total_funded". "show total pipeline" / "total pipeline" /
  "pipeline total" -> execution_mode "state", state "total_pipeline". "show
  forecast funded" / "forecast funded" -> execution_mode "state", state
  "total_forecast_funded". For a plain current-state question use as_of_date =
  context.as_of and output_type "table"; you may set aggregation "balance_sum".
  Only clarify a pipeline (or funded/forecast) question if the user gives NO
  indication of total, trend, comparison, breakdown, or forecast.
- Region defaulting. A bare "region" on the MI route defaults to the obligor
  region dimension geographic_region_obligor. Do NOT ask collateral-vs-obligor.
  Only clarify (or use collateral) if the user EXPLICITLY says "collateral
  region"/"security geography" (-> geographic_region_collateral) or "obligor
  region"/"borrower geography" (-> geographic_region_obligor).
    * "<state> by region" -> execution_mode "risk", risk_monitor_mode
      "concentration", state (e.g. total_funded), dimension
      geographic_region_obligor.
    * "concentration by region" -> execution_mode "risk", risk_monitor_mode
      "concentration", state total_funded, dimension geographic_region_obligor.
- Migration/comparison date defaulting. When the question asks for a migration or
  a comparison and the context provides prev_period AND as_of, USE
  baseline_date = context.prev_period and current_date = context.as_of. Do NOT
  ask which dates to compare when these anchors are present.
""".strip()

_INTERPRETATION_RULES = """
Interpretation rules (resolve concepts to concrete fields; never emit a bare
ambiguous term as a dimension VALUE — always emit the resolved canonical field):
- "portfolio" -> portfolio_id (the Trakt portfolio reference). Requires the
  client to have a portfolio reference config; if context.portfolio_config_available
  is false, return a clarification instead.
- "acquired portfolio" -> acquired_portfolio_id
- "SPV" -> spv_id
- "region" -> geographic_region_obligor by default (see Region defaulting above).
- "stage" / "pipeline stage" -> pipeline_stage, ONLY in a pipeline context
  (state total_pipeline / total_forecast_funded). A bare "stage" with no pipeline
  context is ambiguous -> clarify.
- "IFRS stage" / "IFRS 9 stage" -> ifrs9_stage
- "risk stage" / "internal risk stage" -> internal_risk_stage
- "balance" / "interest rate" / "time on book" buckets -> the corresponding
  *_bucket / *_band dimension with bucket_strategy "quantile" (unless a client
  config defines fixed bands).
- "LTV" / "borrower age" buckets may use bucket_strategy "configured".
- A vague "risk" question must return a clarification (which risk view?).
- A vague "changes" question must return a clarification asking for the period,
  unless the context provides date anchors.
- Never invent dates: use ONLY context anchors (as_of, prev_period, range_start),
  and apply the migration/comparison date defaulting rule above instead of asking.
- Bare ambiguous terms still requiring resolution-or-clarification as a dimension:
  "stage" (outside a pipeline context), "rate", "balance". Emit the resolved
  canonical field, or clarify; never emit the bare term as the dimension value.
""".strip()

_HARD_RULES = """
You translate a business MI question into a single governed query specification.
You DO NOT compute any analytics, numbers, aggregates, or results.

Output rules (MUST follow exactly):
1. Return JSON ONLY. No prose, no explanation, no markdown code fences.
2. Return EITHER one MIQuerySpec-v2 object, OR a clarification object of the
   form {"clarification_required": true, "clarification_question": "..."}.
3. Do NOT return Python, pandas, SQL, or any code.
4. Do NOT calculate or return any data values/results.
5. Do NOT invent field names — use only the allowed fields and enum values.
6. Do NOT invent dates — use only the context date anchors. When a comparison or
   migration is requested and the context provides prev_period and as_of, USE
   them (baseline_date=prev_period, current_date=as_of) rather than asking.
7. Do NOT use chart types outside the allowed chart list.
8. Only clarify when a question is GENUINELY ambiguous. First apply the default
   resolution rules below; if a documented default applies, produce the spec and
   do NOT clarify. Reserve clarification for genuinely ambiguous questions (e.g.
   bare "risk", "changes" without dates, "stage" outside a pipeline context,
   "portfolio" without a portfolio config, "rate").
""".strip()


def _sorted(values) -> list:
    return sorted(values)


def build_mi_spec_prompt(question: str,
                         context: "object" = None,
                         *, semantics: Optional[dict] = None,
                         max_semantic_fields: int = 120) -> str:
    """Construct the constrained Claude prompt for *question*."""
    ctx_block = {}
    if context is not None:
        for attr in ("snapshot_client_id", "route_id", "as_of", "prev_period",
                     "range_start", "portfolio_config_available"):
            if hasattr(context, attr):
                ctx_block[attr] = getattr(context, attr)

    semantic_fields = []
    if semantics:
        semantic_fields = sorted((semantics.get("fields") or {}).keys())[
            :max_semantic_fields]

    allowed = {
        "route_id": list(ROUTE_IDS),
        "execution_mode": _sorted(EXECUTION_MODES),
        "state": _sorted(STATES),
        "temporal_mode": _sorted(TEMPORAL_MODES),
        "risk_monitor_mode": _sorted(RISK_MONITOR_MODES),
        "bucket_strategy": _sorted(BUCKET_STRATEGIES),
        "trend_grain": _sorted(TREND_GRAINS),
        "forecast_probability_source": _sorted(FORECAST_PROBABILITY_SOURCES),
        "output_type": _sorted(OUTPUT_TYPES),
        "chart_type": _sorted(CHART_TYPES),
        "segment": _sorted(SEGMENTS),
        "aggregation": _sorted(AGGREGATIONS),
    }

    parts = [
        _HARD_RULES,
        "",
        "Allowed top-level spec fields:",
        json.dumps(list(ALLOWED_SPEC_FIELDS)),
        "",
        "Allowed enum values:",
        json.dumps(allowed, indent=2),
        "",
        "Known semantic field keys (dimensions/metrics you may reference):",
        json.dumps(semantic_fields),
        "",
        _DEFAULT_RESOLUTION_RULES,
        "",
        _INTERPRETATION_RULES,
        "",
        "Context (use these anchors; do not invent others):",
        json.dumps(ctx_block, indent=2, default=str),
        "",
        "Business question:",
        json.dumps(str(question)),
        "",
        "Return the JSON object now:",
    ]
    return "\n".join(parts)
