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
  /** Median KFI→completion lag (weeks) applied to the conversion denominator; null when unlagged. */
  conversionLagWeeks?: number | null;
  lineage?: Record<string, unknown>;
  singlePeriod: boolean;
  error?: string;
}
