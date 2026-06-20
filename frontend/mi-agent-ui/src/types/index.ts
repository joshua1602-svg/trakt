/**
 * Shared domain types for the MI Agent front end.
 *
 * These intentionally mirror the analytical concepts in the existing Python
 * MI stack (portfolio stratifications, pipeline / forward exposure, static
 * pools, validation / governance) so the UI is trivial to wire to a future
 * MI Agent API. All data is currently mocked — see `src/data/`.
 */

export type Intent =
  | "portfolio_overview"
  | "concentration_risk"
  | "pipeline"
  | "static_pools"
  | "validation"
  | "unknown";

export type ArtifactKind =
  | "kpi"
  | "chart"
  | "table"
  | "validation";

export type ChartType = "bar" | "line" | "area" | "waterfall";

export type Trend = "up" | "down" | "flat";

export interface KPI {
  id: string;
  label: string;
  value: string;
  delta?: string;
  trend?: Trend;
  /** A semantic hint: positive delta good (mint) or bad (rose). */
  deltaIntent?: "positive" | "negative" | "neutral";
  hint?: string;
}

export interface ChartSeries {
  key: string;
  label: string;
  color: string;
}

export interface ChartArtifactData {
  kind: "chart";
  chartType: ChartType;
  /** Recharts-friendly rows. */
  rows: Array<Record<string, string | number>>;
  xKey: string;
  series: ChartSeries[];
  /** Optional value-axis formatting. */
  valueFormat?: "gbp" | "pct" | "number";
  unit?: string;
}

export interface TableColumn {
  key: string;
  label: string;
  align?: "left" | "right";
  format?: "gbp" | "pct" | "number" | "text";
  /** Render a coloured bar in the cell scaled to this column's max. */
  bar?: boolean;
}

export interface TableArtifactData {
  kind: "table";
  columns: TableColumn[];
  rows: Array<Record<string, string | number>>;
}

export interface KPIArtifactData {
  kind: "kpi";
  kpis: KPI[];
}

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

export interface ValidationArtifactData {
  kind: "validation";
  summary: {
    blockers: number;
    warnings: number;
    passed: number;
    coverage: number; // % fields validated
  };
  issues: ValidationIssue[];
}

export type ArtifactData =
  | ChartArtifactData
  | TableArtifactData
  | KPIArtifactData
  | ValidationArtifactData;

export interface Artifact {
  id: string;
  kind: ArtifactKind;
  title: string;
  description: string;
  source: string;
  createdAt: string;
  pinned?: boolean;
  data: ArtifactData;
}

export type ChatRole = "user" | "assistant";

export interface ChatMessage {
  id: string;
  role: ChatRole;
  content: string;
  createdAt: string;
  pending?: boolean;
  /** For assistant messages: assumptions surfaced to the analyst. */
  assumptions?: string[];
  /** Ids of artifacts produced by this turn. */
  artifactRefs?: { id: string; title: string }[];
  intent?: Intent;
}

export interface AgentResponse {
  intent: Intent;
  narrative: string;
  assumptions: string[];
  artifacts: Artifact[];
}
