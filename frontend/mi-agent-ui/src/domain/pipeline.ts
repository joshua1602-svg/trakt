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

/** Completion-month classification relative to the pipeline as-of month. */
export interface ExpectedCompletionSummary {
  asOfMonth: string | null;
  overdueExpectedCompletionCount: number;
  overdueExpectedCompletionWeightedAmount: number;
  currentMonthExpectedCompletionCount: number;
  currentMonthExpectedCompletionWeightedAmount: number;
  nextExpectedCompletionMonth: string | null;
  nextExpectedCompletionCount: number;
  nextExpectedCompletionWeightedAmount: number;
}

/** A generic dimension breakdown row (broker / region). */
export interface DimensionBucket {
  key: string;
  caseCount: number;
  pipelineAmount: number;
  weightedExpectedFundedAmount: number | null;
  /** Set on an aggregated "Other" row when a breakdown is capped to top 10. */
  isOther?: boolean;
  categoriesIncluded?: number;
  sharePct?: number;
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

/** The Phase 1 pipeline snapshot block.
 *
 * Pipeline dates are weekly-operational and DISTINCT from the funded reporting
 * date: `pipelineAsOfDate` follows the latest weekly extract, `pipelineExtractDate`
 * is parsed from that file, and `pipelineSourceFolderDate` is the source scope
 * folder (e.g. the monthly `2025-11-01`). There is no ambiguous `reportingDate`.
 */
export interface PipelineSnapshot {
  ok: boolean;
  recordType: "pipeline";
  error?: string;
  portfolioId: string;
  client_id: string;
  runId: string;
  pipelineAsOfDate: string | null;
  pipelineExtractDate: string | null;
  pipelineSourceFolderDate: string | null;
  pipelineSourceFolder?: string | null;
  sourceFile?: string | null;
  /** CURRENT pipeline snapshot — the latest weekly extract — kept DISTINCT from
   * the source-folder date and from the historical observation window. */
  currentPipelineSnapshotDate?: string | null;
  currentPipelineSourceFile?: string | null;
  historicalObservationWindowStart?: string | null;
  historicalObservationWindowEnd?: string | null;
  uniqueWeeklyExtractsUsed?: number | null;
  sourceFilesScanned?: number | null;
  duplicatesExcluded?: number | null;
  primarySourcePreference?: string | null;
  sourceFoldersIncluded?: string[];
  pipelineRowCount: number;
  pipelineAmount: number | null;
  expectedFundedAmount: number | null;
  weightedExpectedFundedAmount: number | null;
  completionProbabilityBasis?: string;
  completionProbabilitySummary?: Record<string, unknown>;
  historicalCompletionModel?: Record<string, unknown>;
  historicalModelEvidence?: HistoricalModelEvidence;
  stageBreakdown: PipelineStageBucket[];
  /** Unchanged — drives the completion-month chart. */
  expectedCompletionBreakdown: ExpectedCompletionBucket[];
  /** Completion months classified vs the as-of month (overdue / current / next). */
  expectedCompletionSummary?: ExpectedCompletionSummary;
  nextExpectedCompletionMonth?: string | null;
  overdueExpectedCompletionCount?: number;
  overdueExpectedCompletionWeightedAmount?: number;
  currentMonthExpectedCompletionCount?: number;
  /** Capped to top 10 (+ Other) for the landing-page visual. */
  brokerBreakdown?: DimensionBucket[];
  regionBreakdown?: DimensionBucket[];
  /** Uncapped detail (API / agent), present when the breakdown was capped. */
  brokerBreakdownFull?: DimensionBucket[];
  regionBreakdownFull?: DimensionBucket[];
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

/** The deterministic funded + pipeline forecast bridge.
 *
 * `fundedReportingDate` is the funded book's cut-off for the run; the pipeline
 * dates describe the selected weekly extract. They are deliberately separate.
 */
export interface ForecastBridge {
  portfolioId: string;
  client_id: string;
  runId: string;
  fundedReportingDate: string | null;
  pipelineAsOfDate: string | null;
  pipelineExtractDate: string | null;
  pipelineSourceFolderDate: string | null;
  sourceFile?: string | null;
  fundedBalance: number;
  fundedLoanCount: number;
  pipelineAvailable: boolean;
  pipelineAmount: number;
  pipelineCaseCount: number;
  weightedExpectedFundedAmount: number;
  forecastFundedBalance: number;
  forecastLoanCount: number;
  completionProbabilityBasis: string;
  /** Governed probability disclosure. */
  grossPipelineAmount?: number;
  excludedFromWeightingAmount?: number;
  excludedCaseCount?: number;
  activeGrossPipelineAmount?: number | null;
  amountWeightedHistorical?: number | null;
  amountWeightedConfig?: number | null;
  blendedWeightedConversion?: number | null;
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
  byStage?: Record<string, number>;
  excluded?: boolean;
  weighted?: boolean;
  [k: string]: unknown;
}

/** One forecast-by-dimension row (funded actual + weighted pipeline). */
export interface ForecastDimensionRow {
  key: string;
  fundedAmount: number;
  weightedPipelineAmount: number;
  forecastAmount: number;
}

/** Forecast-by-dimension breakdowns (derived: funded + weighted pipeline). */
export interface ForecastBreakdowns {
  byRegion: ForecastDimensionRow[];
  byLtvBucket: ForecastDimensionRow[];
  byCompletionMonth: { month: string; weightedExpectedFundedAmount: number }[];
  byRegionCapped?: DimensionBucket[];
  byLtvBucketCapped?: DimensionBucket[];
}

/** Evidence for the historical completion-rate model (weekly snapshots used). */
export interface HistoricalModelEvidence {
  weeklyFilesUsed: number;
  weeklyFileNames: string[];
  observationWindowStart: string | null;
  observationWindowEnd: string | null;
  historicalRowsUsed: number;
  trackedCaseCount: number;
  observedCompletionCount: number;
  stableIdentifierUsed: string | null;
  stagesUsingHistoricalRates: string[];
  stagesUsingConfigFallback: string[];
  excludedStageCounts: Record<string, number>;
  completionProbabilityBasis: string | null;
  /** Dedup provenance — files scanned vs unique weekly extracts actually used. */
  sourceFilesScanned?: number;
  uniqueWeeklyExtractsUsed?: number;
  duplicatesExcluded?: number;
  primarySourcePreference?: string | null;
  available: boolean;
}

/** "How calculated" lineage for a view. */
export interface ViewLineage {
  view: string;
  source?: string;
  metric?: string;
  weightedMetric?: string;
  formula?: string;
  fundedReportingDate?: string | null;
  pipelineAsOfDate?: string | null;
  pipelineSourceFolderDate?: string | null;
  currentPipelineSnapshotDate?: string | null;
  currentPipelineSourceFile?: string | null;
  historicalObservationWindowStart?: string | null;
  historicalObservationWindowEnd?: string | null;
  uniqueWeeklyExtractsUsed?: number | null;
  sourceFilesScanned?: number | null;
  observationWindowStart?: string | null;
  observationWindowEnd?: string | null;
  completionProbabilityBasis?: string | null;
  historicalModelEvidence?: HistoricalModelEvidence;
  explanation?: string;
  [k: string]: unknown;
}

/** The full forecast-snapshot envelope from `GET /mi/forecast/snapshot`. */
export interface ForecastSnapshot {
  ok: boolean;
  error?: string;
  portfolioId: string;
  client_id: string;
  runId: string;
  fundedReportingDate: string | null;
  pipelineAsOfDate: string | null;
  pipelineExtractDate: string | null;
  pipelineSourceFolderDate: string | null;
  fundedBalance: number;
  fundedLoanCount: number;
  pipelineAvailable: boolean;
  pipelineSnapshot: PipelineSnapshot | null;
  forecastBridge: ForecastBridge | null;
  forecastBreakdowns?: ForecastBreakdowns;
  lineage?: ViewLineage;
  watchlist: WatchlistItem[];
}
