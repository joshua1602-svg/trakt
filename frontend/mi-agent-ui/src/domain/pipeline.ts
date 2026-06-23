/**
 * Pipeline + forecast-bridge snapshot shapes — mirror `mi_agent_api`
 * `pipeline_contract.compute_pipeline_snapshot` and
 * `forecast_bridge.compute_forecast_bridge`.
 *
 * Pipeline is a governed single source of truth, SEPARATE from the funded book.
 * The forecast bridge is a deterministic, backend-derived aggregate composition
 * (funded balance + Σ expected × probability) — never computed in React state.
 */

/** One row of the pipeline stage breakdown. */
export interface PipelineStageBucket {
  stage: string;
  caseCount: number;
  pipelineAmount: number;
  weightedExpectedFundedAmount: number | null;
}

/** One month of the expected-completion breakdown. */
export interface ExpectedCompletionBucket {
  month: string;
  caseCount: number;
  expectedFundedAmount: number;
  weightedExpectedFundedAmount: number | null;
}

/** A generic dimension breakdown row (broker / region). */
export interface DimensionBucket {
  key: string;
  caseCount: number;
  pipelineAmount: number;
  weightedExpectedFundedAmount: number | null;
}

/** A backend data-quality diagnostic (blocker | warning | info). */
export interface PipelineDiagnostic {
  check: string;
  severity: "blocker" | "warning" | "info";
  detail: string;
  count?: number;
  [k: string]: unknown;
}

/** Per-field correlation of a pipeline field to the funded book. */
export interface FieldCorrelation {
  funded_correlation: string[];
  available: boolean;
}

/** The Phase 1 pipeline snapshot block. */
export interface PipelineSnapshot {
  ok: boolean;
  recordType: "pipeline";
  error?: string;
  portfolioId: string;
  client_id: string;
  runId: string;
  reportingDate: string | null;
  pipelineRowCount: number;
  pipelineAmount: number | null;
  expectedFundedAmount: number | null;
  weightedExpectedFundedAmount: number | null;
  stageBreakdown: PipelineStageBucket[];
  expectedCompletionBreakdown: ExpectedCompletionBucket[];
  brokerBreakdown?: DimensionBucket[];
  regionBreakdown?: DimensionBucket[];
  availableMetrics: string[];
  availableDimensions: string[];
  missingDimensions: { dimension: string; reason: string; detail: string }[];
  dataQuality: PipelineDiagnostic[];
  fieldCorrelationToFunded: Record<string, FieldCorrelation>;
  forecastReadiness: Record<string, unknown>;
}

/** Forecast readiness summary. */
export interface ForecastReadiness {
  status: "ready" | "partial" | "blocked";
  missingRequiredFields: string[];
  warnings: string[];
}

/** Diagnostics grouped by severity for the forecast bridge. */
export interface GroupedDataQuality {
  blockers: PipelineDiagnostic[];
  warnings: PipelineDiagnostic[];
  info: PipelineDiagnostic[];
}

/** The deterministic funded + pipeline forecast bridge. */
export interface ForecastBridge {
  portfolioId: string;
  client_id: string;
  runId: string;
  reportingDate: string | null;
  fundedBalance: number;
  fundedLoanCount: number;
  pipelineAvailable: boolean;
  pipelineAmount: number;
  pipelineCaseCount: number;
  weightedExpectedFundedAmount: number;
  forecastFundedBalance: number;
  forecastLoanCount: number;
  completionProbabilityBasis: string;
  expectedCompletionBreakdown: ExpectedCompletionBucket[];
  stageBreakdown: PipelineStageBucket[];
  forecastReadiness: ForecastReadiness;
  dataQuality: GroupedDataQuality;
}

/** One early-warning / watchlist item. */
export interface WatchlistItem {
  category: string;
  severity: "blocker" | "warning" | "info";
  title: string;
  detail: string;
  count?: number;
  [k: string]: unknown;
}

/** The full forecast-snapshot envelope from `GET /mi/forecast/snapshot`. */
export interface ForecastSnapshot {
  ok: boolean;
  error?: string;
  portfolioId: string;
  client_id: string;
  runId: string;
  reportingDate: string | null;
  fundedBalance: number;
  fundedLoanCount: number;
  pipelineAvailable: boolean;
  pipelineSnapshot: PipelineSnapshot | null;
  forecastBridge: ForecastBridge | null;
  watchlist: WatchlistItem[];
}
