/**
 * Securitisation scale-up forecast shapes — mirror
 * `mi_agent_api.forecast_extrapolation`. Completion run-rate (Model A) and
 * KFI-conversion (Model B) extrapolations with downside/base/upside scenario
 * bands and milestone dates, plus the point-in-time weighted pipeline (Model C).
 */

export interface ScenarioRates {
  downside: number;
  base: number;
  upside: number;
}

export interface ProjectedBalance {
  month: string;
  offset: number;
  downside: number;
  base: number;
  upside: number;
}

export interface MilestoneRow {
  threshold: number;
  thresholdLabel: string;
  reached: boolean;
  downsideDate?: string | null;
  baseDate?: string | null;
  upsideDate?: string | null;
  downsideMonths?: number;
  baseMonths?: number;
  upsideMonths?: number;
}

export interface RunRateForecast {
  model: "completion_run_rate";
  available: boolean;
  status: string;
  observedMonths?: number;
  lookbackAverages?: Record<string, number>;
  baseMonthlyRunRate?: number;
  annualisedRunRate?: number;
  scenarioMonthlyRunRate?: ScenarioRates;
  scenarioBasis?: string;
  projectedBalances?: ProjectedBalance[];
  milestones?: MilestoneRow[];
  assumptions?: Record<string, unknown>;
  caveats?: string[];
  caveat?: string;
  completionHistory?: { period: string | null; completion_amount: number }[];
}

export interface KfiConversionForecast {
  model: "kfi_conversion";
  available: boolean;
  status: string;
  observedWeeks?: number;
  avgWeeklyKfiInflow?: number;
  conversionRate?: number;
  lagMonths?: number | null;
  expectedMonthlyCompletion?: number;
  scenarioMonthlyRunRate?: ScenarioRates;
  projectedBalances?: ProjectedBalance[];
  milestones?: MilestoneRow[];
  assumptions?: Record<string, unknown>;
  caveats?: string[];
  caveat?: string;
}

export interface CurrentWeightedPipeline {
  model: "current_weighted_pipeline";
  label: string;
  available: boolean;
  fundedBalance: number;
  weightedExpectedPipeline: number | null;
  forecastFundedBalance: number | null;
  note: string;
}

export interface ForecastExtrapolation {
  portfolioId: string;
  toRunId: string | null;
  reportingPeriod: string | null;
  currentFundedBalance: number;
  currentWeightedPipelineForecast: CurrentWeightedPipeline;
  completionRunRateForecast: RunRateForecast;
  kfiConversionForecast: KfiConversionForecast;
  thresholds: number[];
  dataSufficiency: string;
  sourcePeriods: (string | null)[];
  sourceFiles: (string | null)[];
  lineage?: Record<string, unknown>;
  error?: string;
}
