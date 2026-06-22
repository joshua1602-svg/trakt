/**
 * Deterministic mock funded-portfolio snapshots for offline / demo mode.
 *
 * Mirrors the real /mi/snapshots + /mi/snapshot envelopes for `client_001`
 * (mi_2025_10 → 33 loans / £4.208MM, mi_2025_11 → 73 loans / £8.903MM) so the
 * data-driven dropdowns and the landing-page funded snapshot render with no
 * backend. These are NOT hardcoded prototype options — they reflect the real
 * funded spine and are clearly served via the mock client.
 */

import type { FundedSnapshot, SnapshotIndex, SnapshotKPI } from "@/domain";

const OCT = { run_id: "mi_2025_10", reporting_date: "2025-10-31", loan_count: 33, current_outstanding_balance: 4_207_999.95 };
const NOV = { run_id: "mi_2025_11", reporting_date: "2025-11-30", loan_count: 73, current_outstanding_balance: 8_902_999.7 };

export const MOCK_SNAPSHOT_INDEX: SnapshotIndex = {
  portfolios: [{ client_id: "client_001", label: "CLIENT_001", runs: [OCT, NOV] }],
  source: "mock",
};

function kpi(p: Partial<SnapshotKPI> & Pick<SnapshotKPI, "id" | "label" | "value">): SnapshotKPI {
  return { available: true, format: "number", raw: null, delta: null, deltaIntent: null, hint: null, ...p };
}

const OCT_SNAPSHOT: FundedSnapshot = {
  ok: true,
  portfolio: { client_id: "client_001", label: "CLIENT_001", run_id: "mi_2025_10", reporting_date: "2025-10-31" },
  prior: null,
  loan_count: 33,
  current_outstanding_balance: 4_207_999.95,
  kpis: [
    kpi({ id: "balance", label: "Current funded balance", value: "£4.2MM", format: "gbp", raw: 4_207_999.95 }),
    kpi({ id: "loans", label: "Loans funded", value: "33", raw: 33 }),
    kpi({ id: "wa_current_ltv", label: "Weighted avg current LTV", value: "50.0%", format: "pct", raw: 50 }),
    kpi({ id: "avg_balance", label: "Average loan balance", value: "£127K", format: "gbp", raw: 127_515.15 }),
    kpi({ id: "wa_rate", label: "Weighted avg interest rate", value: "3.2%", format: "pct", raw: 3.2 }),
  ],
  monthly_change: null,
  warnings: [],
  diagnostics: ["No prior reporting date available for this portfolio."],
};

const NOV_SNAPSHOT: FundedSnapshot = {
  ok: true,
  portfolio: { client_id: "client_001", label: "CLIENT_001", run_id: "mi_2025_11", reporting_date: "2025-11-30" },
  prior: { run_id: "mi_2025_10", reporting_date: "2025-10-31" },
  loan_count: 73,
  current_outstanding_balance: 8_902_999.7,
  kpis: [
    kpi({ id: "balance", label: "Current funded balance", value: "£8.9MM", format: "gbp", raw: 8_902_999.7, delta: "+£4.7MM", deltaIntent: "positive", hint: "+111.5% vs prior run" }),
    kpi({ id: "loans", label: "Loans funded", value: "73", raw: 73, delta: "+40", deltaIntent: "positive", hint: "vs 2025-10-31" }),
    kpi({ id: "wa_current_ltv", label: "Weighted avg current LTV", value: "50.0%", format: "pct", raw: 50 }),
    kpi({ id: "avg_balance", label: "Average loan balance", value: "£122K", format: "gbp", raw: 121_958.9 }),
    kpi({ id: "wa_rate", label: "Weighted avg interest rate", value: "3.2%", format: "pct", raw: 3.2 }),
    kpi({ id: "mom_loans", label: "Monthly change · loans", value: "+40", raw: 40, deltaIntent: "positive", hint: "vs 2025-10-31" }),
    kpi({ id: "mom_balance", label: "Monthly change · balance", value: "+£4.7MM", format: "gbp", raw: 4_695_000, deltaIntent: "positive", hint: "+111.5%" }),
    kpi({ id: "new_loans", label: "New loans since prior run", value: "40", raw: 40, deltaIntent: "positive" }),
    kpi({ id: "exited_loans", label: "Exited / redeemed loans", value: "0", raw: 0, deltaIntent: "neutral" }),
  ],
  monthly_change: {
    prior_run_id: "mi_2025_10",
    prior_reporting_date: "2025-10-31",
    loan_count_change: 40,
    balance_change: 4_695_000,
    balance_change_pct: 111.5,
    new_loans: 40,
    exited_loans: 0,
    loans_identifiable: true,
  },
  warnings: [],
  diagnostics: [],
};

const BY_RUN: Record<string, FundedSnapshot> = {
  "client_001/mi_2025_10": OCT_SNAPSHOT,
  "client_001/mi_2025_11": NOV_SNAPSHOT,
};

export function mockSnapshot(portfolioId: string): FundedSnapshot {
  return BY_RUN[portfolioId] ?? NOV_SNAPSHOT;
}
