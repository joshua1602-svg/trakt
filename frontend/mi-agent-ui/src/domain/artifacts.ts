/**
 * Artifact schema — the contract between the analytics engine and the React
 * renderer. The UI is artifact-driven: the agent returns a list of typed
 * artifacts and `ArtifactRenderer` dispatches by `type`.
 *
 * Each artifact carries lineage (`source`) mirroring the metadata that
 * `run_mi_agent_query` / `RiskMonitorResult` / `scenario_engine` already emit,
 * so a future HTTP backend maps onto this shape 1:1.
 */

import type {
  ChartType,
  MIChartType,
  MIQuerySpec,
  MIState,
  RagStatus,
  ResolvedField,
  RiskMode,
  ValueFormat,
} from "./mi";

export const ARTIFACT_TYPES = [
  "kpi",
  "chart",
  "table",
  "validation",
  "risk",
  "scenario",
  "unsupported",
] as const;
export type ArtifactType = (typeof ARTIFACT_TYPES)[number];

/** Lineage / provenance shown on every artifact card. */
export interface ArtifactSource {
  /** "mi_agent.workflow" | "risk_monitor" | "scenario_engine" | "stratify" */
  engine: string;
  state?: MIState;
  spec?: Partial<MIQuerySpec>;
  resolvedFields?: Record<string, ResolvedField>;
  /** Reporting / as-of date (ISO). */
  asOf?: string;
  portfolio?: string;
  /** Short human label, e.g. "Stratification · NUTS1". */
  label: string;
  /** Backend-native MIQuerySpec chart type (distinct from the render type). */
  nativeChartType?: MIChartType;
  /** Raw Plotly figure JSON from the chart factory, when present. */
  figure?: unknown;
  /** Originating NL question (stamped client-side) so a drill-through can
   * re-run the same query with an added filter against the full dataset. */
  question?: string;
}

interface ArtifactBase {
  id: string;
  type: ArtifactType;
  title: string;
  description?: string;
  source: ArtifactSource;
  createdAt: string;
  /** Mock-data disclosure — true until wired to a live backend. */
  mock: boolean;
  warnings?: string[];
  pinned?: boolean;
}

/* ------------------------------- KPI -------------------------------- */

export type Trend = "up" | "down" | "flat";

export interface KPI {
  id: string;
  label: string;
  value: string;
  delta?: string;
  trend?: Trend;
  deltaIntent?: "positive" | "negative" | "neutral";
  hint?: string;
}

export interface KPIArtifact extends ArtifactBase {
  type: "kpi";
  kpis: KPI[];
}

/* ------------------------------ Chart ------------------------------- */

export interface ChartSeries {
  key: string;
  label: string;
  color: string;
}

/** Per-column display hint from the API dataset contract (format + storage scale). */
export interface DisplayHint {
  format?: ValueFormat;
  /** "percent_fraction" (0.51) | "percent_points" (51) | null. */
  scale?: "percent_fraction" | "percent_points" | null;
}

export interface ChartArtifact extends ArtifactBase {
  type: "chart";
  chartType: ChartType;
  rows: Array<Record<string, string | number>>;
  xKey: string;
  series: ChartSeries[];
  valueFormat?: ValueFormat;
  unit?: string;
  /** Loan-level y axis (scatter/bubble) and the heatmap y dimension. */
  yKey?: string;
  /** Loan-level bubble size measure (explicit, never inferred from series order). */
  sizeKey?: string;
  /** Measure column for grid/area charts (heatmap intensity, treemap size). */
  valueKey?: string;
  /** Explicit axis labels for loan-level charts. */
  xLabel?: string;
  yLabel?: string;
  sizeLabel?: string;
  /** Per-column {format, scale} so the renderer formats without guessing. */
  displayHints?: Record<string, DisplayHint>;
}

/* ------------------------------ Table ------------------------------- */

export interface TableColumn {
  key: string;
  label: string;
  align?: "left" | "right";
  format?: ValueFormat;
  /** Percent storage scale from the dataset contract (fraction vs points). */
  scale?: "percent_fraction" | "percent_points" | null;
  /** Render an inline magnitude bar scaled to the column max. */
  bar?: boolean;
}

export interface TableArtifact extends ArtifactBase {
  type: "table";
  columns: TableColumn[];
  rows: Array<Record<string, string | number>>;
}

/* ---------------------------- Validation ---------------------------- */

export type ValidationSeverity = "blocker" | "warning" | "info" | "pass";

export interface ValidationIssue {
  id: string;
  code: string;
  title: string;
  severity: ValidationSeverity;
  scope: string;
  detail: string;
  affected?: number;
}

export interface ValidationArtifact extends ArtifactBase {
  type: "validation";
  summary: { blockers: number; warnings: number; passed: number; coverage: number };
  issues: ValidationIssue[];
}

/* ------------------------------- Risk ------------------------------- */

/** One concentration / limit group row (mirrors RiskMonitorResult frame). */
export interface RiskGroup {
  name: string;
  balance: number;
  share: number; // 0–1
  status: RagStatus;
  limit?: number; // 0–1, for limit mode
  approaching?: boolean;
}

/** One migration cell (mirrors migration_matrix frame). */
export interface MigrationCell {
  from: string;
  to: string;
  balance: number;
  share: number;
  movement: "improved" | "deteriorated" | "unchanged" | "new" | "exited" | "changed";
}

export interface RiskArtifact extends ArtifactBase {
  type: "risk";
  mode: RiskMode;
  dimension: string;
  /** concentration / limits mode. */
  groups?: RiskGroup[];
  /** migration mode. */
  matrix?: MigrationCell[];
  /** ordered axis values for the migration matrix. */
  axis?: string[];
}

/* ----------------------------- Scenario ----------------------------- */

export interface ScenarioPoint {
  year: number;
  balance: number;
  propertyValue?: number;
  ltv: number;
  nnegLoss: number;
  cumulativeNneg: number;
}

export interface ScenarioArtifact extends ArtifactBase {
  type: "scenario";
  assumptions: Record<string, string>;
  projection: ScenarioPoint[];
}

/* ---------------------------- Unsupported ---------------------------- */

/**
 * A response the renderer cannot natively draw (e.g. a future chart type, or a
 * payload arriving before a renderer exists). Carries enough context to explain
 * the gap and, where available, the raw Plotly figure for a future renderer.
 * This makes "we received it but can't draw it yet" an explicit, visible state
 * rather than a silent drop.
 */
export interface UnsupportedArtifact extends ArtifactBase {
  type: "unsupported";
  /** What was received but couldn't be rendered (e.g. "heatmap"). */
  reason: string;
}

/* --------------------------- Discriminated --------------------------- */

export type Artifact =
  | KPIArtifact
  | ChartArtifact
  | TableArtifact
  | ValidationArtifact
  | RiskArtifact
  | ScenarioArtifact
  | UnsupportedArtifact;
