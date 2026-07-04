/**
 * Origination-vintage (static-pool) cohort analysis — mirrors
 * `mi_agent_api.cohorts.cohort_analysis`. Point-in-time vintage aggregates only;
 * redemption/completion/performance curves are NOT computed in the MI path and
 * are never fabricated here.
 */

/** One origination-year cohort row. Metrics present only when computed. */
export interface CohortRow {
  vintage: string;
  loanCount: number;
  balance?: number;
  sharePct?: number | null;
  waLtv?: number | null;
  waRate?: number | null;
  waMonthsOnBook?: number | null;
}

export interface CohortAnalysis {
  dataset: "cohorts";
  portfolioId: string;
  cohortBasis?: string;
  period?: string;
  reportingDate?: string | null;
  available: boolean;
  /** Why cohort analysis is unavailable (no vintage on the tape, etc.). */
  reason?: string;
  totalBalance?: number | null;
  totalLoanCount?: number;
  /** Which per-cohort metrics were actually computed (drives the columns). */
  metricsAvailable?: string[];
  cohorts: CohortRow[];
  lineage?: Record<string, unknown>;
}
