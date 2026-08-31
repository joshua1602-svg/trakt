/**
 * Evolution (time-series) shapes — mirror `mi_agent_api.evolution`.
 * Funded / pipeline / forecast metrics across governed monthly runs and weekly
 * pipeline extracts, each period carrying its own reconciliation + lineage.
 */

import type { TimingDisclosure } from "./pipeline";

export interface EvolutionPeriod {
  period: string;
  run_id?: string;
  reporting_date?: string | null;
  extract_date?: string | null;
  week?: string | null;
  metrics: Record<string, number | null>;
  reconciliation?: Record<string, unknown> | null;
  source_file?: string;
}

export interface BreakdownPoint {
  period: string;
  key: string;
  value: number;
}

export interface StagePoint {
  period: string;
  stage: string;
  value: number | null;
  /** Case count for this stage/extract (drives the count + conversion views). */
  count?: number;
  /** Day-level extract date (preferred x label over a month period). */
  week?: string | null;
}

export interface FundedEvolution {
  dataset: "funded";
  portfolioId: string;
  toRunId: string | null;
  availableRunIds: string[];
  reportingDates: (string | null)[];
  sourceFiles: (string | null)[];
  periods: EvolutionPeriod[];
  breakdowns: Record<string, BreakdownPoint[]>;
  lineage?: Record<string, unknown>;
  singlePeriod: boolean;
  error?: string;
}

export interface PipelineEvolution {
  dataset: "pipeline";
  portfolioId: string;
  toRunId: string | null;
  availableExtractDates?: (string | null)[];
  sourceFiles?: string[];
  uniqueWeeklyExtractsUsed?: number | null;
  periods: EvolutionPeriod[];
  byStage: StagePoint[];
  lineage?: Record<string, unknown>;
  singlePeriod: boolean;
  /** Funded-vs-pipeline timing disclosure (pipeline history not capped by funded date). */
  pipelineTiming?: TimingDisclosure;
  error?: string;
}

/** Where a case that left a stage actually went. */
export interface StageDeparture {
  /** The destination stage, or ABSENT for a case no longer in the extract. */
  stage: string;
  caseCount: number;
  amount: number;
}

/**
 * One live stage's opening-to-closing reconciliation, on counts AND amounts:
 *
 *   opening live + arrivals - departures +/- amount change on stayers
 *     = closing live
 */
export interface PipelineStageMovement {
  stage: string;
  openingCaseCount: number;
  openingAmount: number;
  arrivalCaseCount: number;
  arrivalAmount: number;
  departureCaseCount: number;
  departureAmount: number;
  persistingCaseCount: number;
  /** Movement on cases present in BOTH extracts — an amendment, not an exit. */
  amountChangeOnPersisting: number;
  closingCaseCount: number;
  closingAmount: number;
  departuresByDestination: StageDeparture[];
  residual: number;
  reconciles: boolean;
}

/**
 * What happened to pipeline cases between two governed weekly extracts.
 *
 * `available: false` carries the engine's own reason. There is deliberately no
 * fallback: without a stable case identifier in both extracts the only honest
 * answer is that this cannot be reported.
 */
export interface PipelineMovement {
  dataset: "pipeline_movement";
  portfolioId: string;
  available: boolean;
  reason?: string;
  openingWeek?: string | null;
  closingWeek?: string | null;
  identifierField?: string;
  openingCaseCount?: number;
  closingCaseCount?: number;
  persistingCaseCount?: number;
  stages: PipelineStageMovement[];
  reconciles?: boolean;
  lineage?: Record<string, unknown>;
}

export interface ForecastEvolution {
  dataset: "forecast";
  portfolioId: string;
  toRunId: string | null;
  periods: EvolutionPeriod[];
  lineage?: Record<string, unknown>;
  singlePeriod: boolean;
  error?: string;
}

// --------------------------------------------------------------------------- //
// Weekly origination funnel trends — KFI / Application / Offer / Completion
// value + count per governed weekly extract (mirrors evolution.pipeline_funnel).
// --------------------------------------------------------------------------- //
/** Per-week STOCK level of a funnel stage (drives the optional cumulative line). */
export interface FunnelPoint {
  week: string | null;
  value: number | null;
  count: number;
}

/** Per-week WEEKLY FLOW of a funnel stage (drives the default bars): the
 * week-on-week change in the stage level (new origination that week). */
export interface FunnelFlowPoint {
  week: string | null;
  flowValue: number | null;
  flowCount: number | null;
}

/**
 * Forward conversion of a stage vs KFI: the average weekly FLOW into the stage
 * (last 5 weeks) over the KFI STOCK as it stood `lagWeeks` earlier — the KFI
 * book those completions actually came from. A weekly rate; `lagWeeks` is null
 * (and `lagApplied` false) when the KFI→completion lag is unknown.
 */
export interface FunnelConversion {
  basis: string;
  lagWeeks: number | null;
  lagApplied: boolean;
  denominatorWeek: string | null;
  avgWeeklyFlowCount: number | null;
  avgWeeklyFlowValue: number | null;
  kfiStockCount: number | null;
  kfiStockValue: number | null;
  weeklyRateCount: number | null;
  weeklyRateValue: number | null;
  /** Weeks feeding the trailing average, the minimum needed, and whether the
   * rate is reliable enough to publish/forecast off (not built on 1-2 weeks). */
  weeksInWindow: number;
  minWeeks: number;
  sufficient: boolean;
}

export interface FunnelStageSummary {
  label: string;
  // Weekly FLOW (default basis for the origination funnel).
  latestFlowValue: number | null;
  latestFlowCount: number | null;
  priorFlowValue: number | null;
  priorFlowCount: number | null;
  fiveWeekAvgFlowValue: number | null;
  fiveWeekAvgFlowCount: number | null;
  deltaFlowValue: number | null;
  deltaFlowCount: number | null;
  // STOCK level (for the optional cumulative line).
  latestStockValue: number | null;
  latestStockCount: number;
  fiveWeekAvgStockValue: number | null;
  fiveWeekAvgStockCount: number | null;
  trend: "up" | "down" | "flat";
  weeksObserved: number;
  conversion: FunnelConversion | null;
}

export interface PipelineFunnelEvolution {
  dataset: "pipeline_funnel";
  portfolioId: string;
  toRunId: string | null;
  stages: string[];
  stageLabels: Record<string, string>;
  weeks: (string | null)[];
  sourceFiles: string[];
  uniqueWeeklyExtractsUsed?: number | null;
  series: Record<string, FunnelPoint[]>;
  flowSeries: Record<string, FunnelFlowPoint[]>;
  summary: Record<string, FunnelStageSummary>;
  /** Median KFI→completion lag (weeks) applied to the velocity denominator; null when unlagged. */
  conversionLagWeeks?: number | null;
  /** Canonical conversion: cumulative % of the original KFI cohort reaching each
   * milestone by each week (a true cohort funnel, not a stock ratio). */
  cohortProgression?: KfiCohortProgression | null;
  /** Headline KPI: % of the KFI cohort funded to date (latest Funded point). */
  cumulativeCohortConversion?: number | null;
  lineage?: Record<string, unknown>;
  singlePeriod: boolean;
  error?: string;
}

/** Cumulative KFI-cohort funnel: for each week, the % of the original KFI cohort
 * that has reached each milestone (KFI → Application → Offer → Funded). Distinct
 * from the vintage static-pool `CohortProgression` in domain/cohorts. */
export interface KfiCohortProgression {
  weeks: string[];
  stages: string[];
  series: Record<string, number[]>;
  cohortSize: number;
}
