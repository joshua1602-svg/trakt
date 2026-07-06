/**
 * Funded-portfolio snapshot + reporting-run discovery — mirrors
 * `mi_agent_api/snapshots.py`. These shapes drive the data-driven portfolio /
 * reporting-date dropdowns and the deterministic landing-page snapshot that
 * renders BEFORE any natural-language query is asked.
 */

/** One reporting run discovered from local onboarding output. */
export interface SnapshotRun {
  run_id: string;
  reporting_date: string | null;
  loan_count: number;
  current_outstanding_balance: number;
}

/** A funded portfolio (client) and its available reporting runs. */
export interface SnapshotPortfolio {
  client_id: string;
  label: string;
  /** Ordered oldest → newest by reporting date. */
  runs: SnapshotRun[];
}

/** The /mi/snapshots discovery index. */
export interface SnapshotIndex {
  portfolios: SnapshotPortfolio[];
  source?: string;
}

/** A deterministic KPI tile (no parser involved). */
export interface SnapshotKPI {
  id: string;
  label: string;
  value: string;
  format?: string;
  raw?: number | null;
  available?: boolean;
  delta?: string | null;
  deltaIntent?: "positive" | "negative" | "neutral" | null;
  hint?: string | null;
}

/** Month-on-month change vs the prior available run. */
export interface MonthlyChange {
  prior_run_id?: string | null;
  prior_reporting_date?: string | null;
  loan_count_change: number;
  balance_change: number;
  balance_change_pct: number | null;
  new_loans: number | null;
  exited_loans: number | null;
  loans_identifiable: boolean;
}

/** The full funded-book snapshot for a selected portfolio/run. */
export interface FundedSnapshot {
  ok: boolean;
  error?: string;
  portfolio: {
    client_id: string;
    label: string;
    run_id: string;
    reporting_date: string | null;
  };
  prior: { run_id: string; reporting_date: string | null } | null;
  loan_count: number;
  current_outstanding_balance: number;
  kpis: SnapshotKPI[];
  /** Point-in-time balance/share breakdowns by dimension (may be empty). */
  stratifications?: FundedStratification[];
  monthly_change: MonthlyChange | null;
  warnings: string[];
  diagnostics: string[];
}

/** One band/category of a point-in-time funded stratification. */
export interface FundedStratBar {
  label: string;
  balance: number;
  count: number;
  sharePct: number;
  /** Balance-weighted LTV for the band (fraction), when derivable. */
  waLtv?: number;
}

/** A funded stratification by one dimension (LTV band / age / region / …). */
export interface FundedStratification {
  key: string;
  label: string;
  bars: FundedStratBar[];
}
