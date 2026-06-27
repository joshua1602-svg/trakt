/**
 * Evolution (time-series) shapes — mirror `mi_agent_api.evolution`.
 * Funded / pipeline / forecast metrics across governed monthly runs and weekly
 * pipeline extracts, each period carrying its own reconciliation + lineage.
 */

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
  value: number;
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
