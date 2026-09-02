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

/**
 * Vintage FORMATION — what entered the book in each origination period, each
 * loan counted once in its own vintage.
 *
 * Deliberately not the book outstanding at a reporting date: that is portfolio
 * evolution (`domain/evolution`), and a book of 33 October plus 40 November
 * loans reads 33 and 40 here, 33 then 73 there.
 */
export interface CohortVintage {
  vintage: string;
  originalLoanCount: number;
  originalBalance: number;
  firstSeen?: string | null;
  /** Balance-weighted, as a FRACTION (0.395 == 39.5%) per the unit contract. */
  waOriginalLtv?: number | null;
  waEntryLtv?: number | null;
  waRate?: number | null;
}

export interface CohortFormation {
  dataset: "cohort_formation";
  portfolioId: string;
  cohortBasis: string;
  grain: string;
  available: boolean;
  reason?: string | null;
  vintages: CohortVintage[];
  totalLoanCount?: number;
  /** Loans a later snapshot restated into a different vintage. The first
   *  assignment stands; these are surfaced rather than silently applied. */
  lateCorrections?: { loanId: string; assigned: string; restatedTo: string;
                      seenIn?: string | null }[];
  lineage?: Record<string, unknown>;
}

/** One reporting period of a fixed static pool. The count can fall through
 *  exits; it can never rise. */
export interface StaticPoolPeriod {
  period: string;
  reportingDate?: string | null;
  monthsSinceEntry?: number | null;
  survivingLoanCount: number;
  /** The vintage was still originating at this date, so the pool was not yet
   *  fixed and retention / exits are not reported for it. */
  forming?: boolean;
  currentBalance: number;
  loanRetention?: number | null;
  balanceRetention?: number | null;
  exitsInPeriod: number;
  cumulativeExits: number;
  waLtv?: number | null;
  waRate?: number | null;
}

export interface CohortStaticPool {
  /** The date the vintage stopped originating; the pool is fixed from here. */
  formationEnd?: string | null;
  /** True once a fixed pool exists (the vintage has finished forming). */
  poolAnchored?: boolean;
  /** How many reported periods pre-date formation completing. */
  formingPeriods?: number;
  dataset: "cohort_static_pool";
  portfolioId: string;
  cohortBasis: string;
  vintage: string;
  grain: string;
  available: boolean;
  reason?: string | null;
  originalLoanCount?: number | null;
  originalBalance?: number | null;
  periods: StaticPoolPeriod[];
  singlePeriod?: boolean;
  lineage?: Record<string, unknown>;
}

export interface CohortVintageQuery {
  portfolioContext?: string;
  /** Omit for formation; supply to follow one vintage's static pool. */
  vintage?: string;
  grain?: "M" | "Q" | "Y";
}
