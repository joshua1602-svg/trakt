/**
 * MI domain vocabulary — mirrors the Python MI Agent semantic layer so the
 * React front end speaks the same language as the analytics engine.
 *
 * Sources:
 *   - mi_agent/mi_query_spec.py        (STATES, AGGREGATIONS, CHART_TYPES, …)
 *   - mi_agent/mi_semantics_field_registry.yaml (dimensions / measures)
 *   - config/mi/stratification_catalogue.yaml, config/mi/buckets.yaml
 *
 * Keeping these as const tuples (not free strings) gives us exhaustiveness and
 * autocomplete, and makes the future query-builder UI trivial.
 */

/** Portfolio "states" the analytics engine can assemble (state_library.yaml). */
export const MI_STATES = [
  "total_funded",
  "total_pipeline",
  "total_forecast_funded",
  "cohort_by_date",
  "cohort_by_portfolio",
  "cohort_by_spv",
  "cohort_by_acquired_portfolio",
] as const;
export type MIState = (typeof MI_STATES)[number];

/** Aggregation vocabulary (mi_query_spec.AGGREGATIONS). */
export const AGGREGATIONS = [
  "sum",
  "avg",
  "weighted_avg",
  "count",
  "count_distinct",
  "median",
  "distribution",
  "loan_level",
  "balance_sum",
] as const;
export type Aggregation = (typeof AGGREGATIONS)[number];

/** Chart types the MI Agent can actually emit (mi_query_spec.CHART_TYPES,
 *  excluding the sentinel "none"). This is the authoritative set — the
 *  /mi/catalogue endpoint serves exactly these from the Python side. */
export const MI_CHART_TYPES = ["bar", "line", "scatter", "bubble", "heatmap", "treemap"] as const;
export type MIChartType = (typeof MI_CHART_TYPES)[number];

/**
 * Render chart types: the MI Agent set PLUS front-end presentation-only
 * variants (`area`, `waterfall`) used for mock/derived visuals. These two are
 * NOT part of the MIQuerySpec contract — they are renderer conveniences and
 * must never be sent back to the agent as a spec chart_type.
 */
export const CHART_TYPES = [...MI_CHART_TYPES, "area", "waterfall"] as const;
export type ChartType = (typeof CHART_TYPES)[number];

/** Risk-monitor modes (mi_query_spec.RISK_MONITOR_MODES). */
export const RISK_MODES = ["concentration", "migration", "trajectory", "flags", "limits"] as const;
export type RiskMode = (typeof RISK_MODES)[number];

/** A catalogue dimension entry (subset of the semantic registry metadata). */
export interface DimensionDef {
  key: string;
  label: string;
  bucketed: boolean;
  /** e.g. NUTS region codes, IFRS9 stages — for future filter UIs. */
  group: "geography" | "channel" | "product" | "risk" | "performance" | "structure" | "segmentation" | "vintage";
}

/** A catalogue measure entry. */
export interface MeasureDef {
  key: string;
  label: string;
  format: ValueFormat;
  defaultAggregation: Aggregation;
}

export type ValueFormat = "gbp" | "pct" | "number" | "decimal" | "text" | "date";

/** A resolved field as returned by the executor (`resolved_fields`). */
export interface ResolvedField {
  canonicalField: string;
  role: "dimension" | "measure" | "date" | "identifier" | "flag";
  format: ValueFormat;
}

/**
 * A partial MIQuerySpec — the agent request/echo. We keep it partial and
 * loosely typed on purpose: the React layer never *builds* a full spec yet, it
 * only displays the one the agent interpreted and forwards context.
 */
export interface MIQuerySpec {
  intent: "chart" | "table" | "summary";
  chartType?: ChartType;
  metric?: string;
  dimension?: string;
  dimensions?: string[];
  aggregation?: Aggregation;
  state?: MIState;
  riskMode?: RiskMode;
  topN?: number;
  filters?: Record<string, unknown>;
}

/** RAG status used across concentration / limit artifacts. */
export type RagStatus = "green" | "amber" | "red" | "below_minimum";

/** Risk thresholds from config/mi/risk_monitor.yaml. */
export const RISK_THRESHOLDS = {
  concentration: { amber: 0.2, red: 0.3 },
  limitUsage: { amber: 0.8, red: 1.0 },
} as const;
