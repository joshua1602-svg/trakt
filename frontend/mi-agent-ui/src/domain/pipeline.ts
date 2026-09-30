/**
 * Pipeline + forecast-bridge snapshot shapes — mirror `mi_agent_api`
 * `pipeline_contract.compute_pipeline_snapshot` and
 * `forecast_bridge.compute_forecast_bridge`.
 *
 * Pipeline is a governed single source of truth, SEPARATE from the funded book.
 * The forecast bridge is a deterministic, backend-derived aggregate composition
 * (funded balance + Σ expected × probability) — never computed in React state.
 */

/**
 * Non-blocking disclosure of the funded-vs-pipeline date difference — mirrors
 * `mi_agent_api.pipeline_timing.timing_disclosure`. Pipeline defaults to its
 * latest weekly extract; when that is later than the selected funded reporting
 * date we DISCLOSE the gap (info, or a stronger warning above the threshold)
 * rather than hide/truncate the pipeline.
 */
export interface TimingDisclosure {
  fundedActualsAsOf: string | null;
  pipelineExtractAsOf: string | null;
  lagDays: number | null;
  level: "none" | "info" | "warning";
  message: string | null;
  warnThresholdDays: number;
}

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
  /** D21: null when a case's stage has no measured validity window, so
   *  whether it has lapsed is unknown — never counted as zero. */
  caseCount: number | null;
  expectedFundedAmount: number | null;
  weightedExpectedFundedAmount: number | null;
}

/** Completion-month classification relative to the pipeline as-of month. */
export interface ExpectedCompletionSummary {
  asOfMonth: string | null;
  overdueExpectedCompletionCount: number | null;
  overdueExpectedCompletionWeightedAmount: number | null;
  currentMonthExpectedCompletionCount: number | null;
  currentMonthExpectedCompletionWeightedAmount: number | null;
  nextExpectedCompletionMonth: string | null;
  nextExpectedCompletionCount: number | null;
  nextExpectedCompletionWeightedAmount: number | null;
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

/** Which region the region breakdown is: the client's reporting taxonomy
 *  where the extract's regions resolve to it, and the live cases whose region
 *  has no governed mapping (left out of the chart, disclosed here). */
export interface PipelineRegionBasis {
  field: string;
  taxonomy: string | null;
  unmappedCaseCount: number;
  unmappedAmount: number;
  unmappedValues: Record<string, number>;
}

/** Prior weekly pipeline snapshot aggregates, for week-on-week tile deltas.
 *
 * Additive + optional: present only when a genuine prior weekly extract exists.
 * The UI must never synthesise a prior week — when this is absent the tiles show
 * "No prior week". Each metric is independently optional so partial history
 * degrades gracefully. */
export interface PipelineWeeklyPrior {
  /** The prior weekly extract date (ISO), for the "vs prior week" label. */
  snapshotDate: string | null;
  sourceFile?: string | null;
  pipelineRowCount?: number | null;
  pipelineAmount?: number | null;
  weightedExpectedFundedAmount?: number | null;
}

/** The pipeline's credit profile, on the funded tiles' definitions (amount-
 *  weighted averages; single-borrower share of cases). A null measure means the
 *  extract does not carry its inputs — the tile is then omitted. */
export interface PipelineProfile {
  waLtvPct: number | null;
  waInterestRatePct: number | null;
  waYoungestAge: number | null;
  waPropertyValue: number | null;
  singleBorrowerPct: number | null;
  singleBorrowerCount: number | null;
  borrowerTypeKnownCount: number | null;
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
  /** D21: false when a live case's stage has no rate measured from the client's
   *  history; the weighted figures it affects are then null, never zero. */
  weightingComplete?: boolean;
  weightingIncompleteReason?: string | null;
  /** Prior weekly extract aggregates for week-on-week tile deltas (optional). */
  priorWeek?: PipelineWeeklyPrior | null;
  completionProbabilityBasis?: string;
  completionProbabilitySummary?: Record<string, unknown>;
  historicalCompletionModel?: Record<string, unknown>;
  historicalModelEvidence?: HistoricalModelEvidence;
  stageBreakdown: PipelineStageBucket[];
  /** "open": every figure is KFI / Application / Offer only. */
  pipelinePopulation?: "open";
  openStages?: string[];
  /** What the open-pipeline figures leave out (still in the weekly extract). */
  excludedFromOpenPipeline?: {
    stages: { stage: string; caseCount: number; amount: number }[];
    cases: number;
    amount: number;
  } | null;
  /** Every row in the weekly extract, open or not. */
  extractRowCount?: number;
  /** Credit profile tiles (additive; absent on older payloads). */
  profile?: PipelineProfile | null;
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
  /** The region basis of the region breakdown (additive). */
  regionBasis?: PipelineRegionBasis;
  /** The extract's own region spelling, kept for audit (additive). */
  regionSourceBreakdownFull?: DimensionBucket[];
  /** Product (capped top 10 + Other) and LTV band breakdowns — additive. */
  productBreakdown?: DimensionBucket[];
  productBreakdownFull?: DimensionBucket[];
  ltvBreakdown?: DimensionBucket[];
  availableMetrics: string[];
  availableDimensions: string[];
  missingDimensions: { dimension: string; reason: string; detail: string }[];
  dataQuality: PipelineDiagnostic[];
  fieldCorrelationToFunded: Record<string, FieldCorrelation>;
  forecastReadiness: Record<string, unknown>;
  /** Funded-vs-pipeline timing disclosure (pipeline shown as of its latest extract). */
  pipelineTiming?: TimingDisclosure;
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
  /** D21: null — with `forecastWithheldReason` — when the weighted pipeline
   *  cannot be stated; the forecast built on it is then null too. */
  weightedExpectedFundedAmount: number | null;
  forecastFundedBalance: number | null;
  forecastWithheldReason?: string | null;
  forecastLoanCount: number;
  completionProbabilityBasis: string;
  /** Governed probability disclosure. */
  grossPipelineAmount?: number;
  excludedFromWeightingAmount?: number;
  excludedCaseCount?: number;
  /** The same exclusion by governed reason (completed, withdrawn,
   *  not_forecast, lapsed, missing_stage, missing_probability). Additive. */
  excludedByReason?: Record<string, { count: number; amount: number }>;
  activeGrossPipelineAmount?: number | null;
  amountWeightedHistorical?: number | null;
  amountWeightedConfig?: number | null;
  blendedWeightedConversion?: number | null;
  expectedCompletionBreakdown: ExpectedCompletionBucket[];
  stageBreakdown: PipelineStageBucket[];
  forecastReadiness: ForecastReadiness;
  dataQuality: GroupedDataQuality;
  /** Funded actuals vs latest-pipeline timing disclosure (both anchors + level). */
  pipelineTiming?: TimingDisclosure;
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
/** Which region the forecast-by-region breakdown adds the two books up in
 *  (the reporting taxonomy when both books carry it), and what it cannot
 *  place. Additive. */
export interface ForecastRegionBasis {
  field: string;
  unplacedFundedAmount: number;
  unplacedWeightedPipelineAmount: number;
  /** The two above together — the forecast the chart cannot place. */
  unplacedForecastAmount: number;
}

export interface ForecastBreakdowns {
  byRegion: ForecastDimensionRow[];
  regionBasis?: ForecastRegionBasis;
  byLtvBucket: ForecastDimensionRow[];
  byCompletionMonth: { month: string; weightedExpectedFundedAmount: number | null }[];
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
  /** Funded actuals vs latest-pipeline timing disclosure (both anchors + level). */
  pipelineTiming?: TimingDisclosure;
}
