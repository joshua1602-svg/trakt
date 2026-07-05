/**
 * Origination-vintage (static-pool) cohort analysis — mirrors
 * `mi_agent_api.cohorts.cohort_analysis`. Point-in-time vintage aggregates only;
 * redemption/completion/performance curves are NOT computed in the MI path and
 * are never fabricated here.
 */

/** One static-pool cohort row. Metrics present only when computed. */
export interface CohortRow {
  /** The cohort label (vintage year, age band, LTV band or channel). */
  cohort: string;
  /** Alias of `cohort`, kept for the vintage-only contract. */
  vintage?: string;
  loanCount: number;
  balance?: number;
  sharePct?: number | null;
  waLtv?: number | null;
  waRate?: number | null;
  waMonthsOnBook?: number | null;
}

/** The dimension a static pool is grouped by (asset-class-agnostic). */
export type CohortDimension = "vintage" | "age" | "ltv" | "channel";

export interface CohortAnalysis {
  dataset: "cohorts";
  portfolioId: string;
  cohortBasis?: string;
  period?: string;
  reportingDate?: string | null;
  available: boolean;
  /** Why cohort analysis is unavailable (no source for the dimension, etc.). */
  reason?: string;
  /** The dimension these cohorts are grouped by + its column header. */
  dimension?: CohortDimension;
  dimensionLabel?: string;
  /** Which dimensions the tape can support (drives the selector). */
  availableDimensions?: CohortDimension[];
  totalBalance?: number | null;
  totalLoanCount?: number;
  /** Which per-cohort metrics were actually computed (drives the columns). */
  metricsAvailable?: string[];
  cohorts: CohortRow[];
  lineage?: Record<string, unknown>;
}

/** Vintage grain for cohort views. */
export type CohortGrain = "Y" | "Q" | "M";

/** One reporting period's funded metrics for a static-pool cohort. */
export interface CohortProgressionPeriod {
  period: string;
  reporting_date?: string | null;
  loanCount: number;
  metrics: Record<string, number | null>;
}

/**
 * Static-pool cohort PROGRESSION — mirrors
 * `mi_agent_api.evolution.funded_cohort_progression`. How a cohort (a source
 * portfolio ± origination vintage) seasons across reporting periods.
 */
export interface CohortProgression {
  dataset: "cohort_progression";
  portfolioId: string;
  available: boolean;
  reason?: string | null;
  lens: string;
  vintage?: string | null;
  grain?: CohortGrain;
  metricsAvailable: string[];
  periods: CohortProgressionPeriod[];
  singlePeriod?: boolean;
  lineage?: Record<string, unknown>;
}

/** Options for a cohort-progression request. */
export interface CohortProgressionQuery {
  lens?: string;
  vintage?: string;
  grain?: CohortGrain;
}
