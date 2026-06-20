/**
 * Mock MI data for the first-pass UI.
 *
 * Figures are representative of a UK Equity Release Mortgage (ERM) portfolio,
 * matching the analytical concepts in the Python MI stack (stratifications,
 * pipeline / forward exposure, static pools, validation). Numbers are
 * illustrative only.
 */

import type {
  Artifact,
  KPI,
  ValidationIssue,
} from "@/types";

export const PORTFOLIOS = [
  { id: "erm-uk-master", name: "ERM UK — Master", entity: "Trakt SPV I" },
  { id: "erm-uk-warehouse", name: "ERM UK — Warehouse", entity: "Warehouse Co" },
  { id: "erm-uk-fwd", name: "ERM UK — Forward Flow", entity: "Origination" },
];

export const REPORTING_DATES = [
  "2026-05-31",
  "2026-04-30",
  "2026-03-31",
  "2026-02-28",
];

/* --------------------------------------------------------------------- *
 * Executive KPIs (landing strip)
 * --------------------------------------------------------------------- */
export const EXEC_KPIS: KPI[] = [
  {
    id: "kpi-aum",
    label: "Portfolio Balance",
    value: "£842.6MM",
    delta: "+2.4%",
    trend: "up",
    deltaIntent: "positive",
    hint: "vs. prior period",
  },
  {
    id: "kpi-loans",
    label: "Active Loans",
    value: "4,318",
    delta: "+86",
    trend: "up",
    deltaIntent: "positive",
    hint: "net of redemptions",
  },
  {
    id: "kpi-ltv",
    label: "Wtd. Avg. LTV",
    value: "31.4%",
    delta: "+0.6pp",
    trend: "up",
    deltaIntent: "negative",
    hint: "balance-weighted",
  },
  {
    id: "kpi-age",
    label: "Wtd. Avg. Borrower Age",
    value: "71.2",
    delta: "-0.3",
    trend: "down",
    deltaIntent: "neutral",
    hint: "years",
  },
  {
    id: "kpi-coupon",
    label: "Wtd. Avg. Coupon",
    value: "6.18%",
    delta: "+0.11pp",
    trend: "up",
    deltaIntent: "neutral",
    hint: "fixed for life",
  },
  {
    id: "kpi-dq",
    label: "Data Quality",
    value: "98.1%",
    delta: "+0.4pp",
    trend: "up",
    deltaIntent: "positive",
    hint: "fields validated",
  },
];

/* --------------------------------------------------------------------- *
 * Regional concentration (stratification)
 * --------------------------------------------------------------------- */
export const REGION_ROWS = [
  { region: "London", balance: 184.2, share: 21.9, loans: 712, ltv: 33.8 },
  { region: "South East", balance: 162.7, share: 19.3, loans: 884, ltv: 31.1 },
  { region: "South West", balance: 98.4, share: 11.7, loans: 561, ltv: 30.4 },
  { region: "East of England", balance: 86.1, share: 10.2, loans: 498, ltv: 30.9 },
  { region: "North West", balance: 74.9, share: 8.9, loans: 471, ltv: 29.7 },
  { region: "West Midlands", balance: 61.3, share: 7.3, loans: 388, ltv: 30.2 },
  { region: "Yorkshire", balance: 52.8, share: 6.3, loans: 341, ltv: 29.1 },
  { region: "Scotland", balance: 48.6, share: 5.8, loans: 318, ltv: 28.4 },
  { region: "East Midlands", balance: 41.2, share: 4.9, loans: 274, ltv: 29.8 },
  { region: "Other", balance: 32.4, share: 3.8, loans: 271, ltv: 28.0 },
];

/* --------------------------------------------------------------------- *
 * Pipeline / forward exposure bridge to a target securitisation size
 * --------------------------------------------------------------------- */
export const PIPELINE_BRIDGE = [
  { stage: "Current Funded", value: 84.2, type: "base" },
  { stage: "Offer Issued", value: 6.8, type: "add" },
  { stage: "Legals", value: 5.1, type: "add" },
  { stage: "Valuation", value: 3.4, type: "add" },
  { stage: "Application", value: 2.9, type: "add" },
  { stage: "Expected Fallout", value: -2.4, type: "sub" },
  { stage: "Forecast Size", value: 100.0, type: "total" },
];

export const PIPELINE_FLOW = [
  { month: "Dec", funded: 11.2, pipeline: 14.1 },
  { month: "Jan", funded: 12.8, pipeline: 15.6 },
  { month: "Feb", funded: 10.4, pipeline: 13.9 },
  { month: "Mar", funded: 13.9, pipeline: 17.2 },
  { month: "Apr", funded: 15.1, pipeline: 18.4 },
  { month: "May", funded: 16.7, pipeline: 19.8 },
];

/* --------------------------------------------------------------------- *
 * Static pools — cumulative redemption / prepayment by vintage
 * --------------------------------------------------------------------- */
export const STATIC_POOL_VINTAGE: Array<Record<string, number>> = [
  { month: 0, "2021": 0, "2022": 0, "2023": 0, "2024": 0 },
  { month: 6, "2021": 1.1, "2022": 0.9, "2023": 0.8, "2024": 0.6 },
  { month: 12, "2021": 3.4, "2022": 2.9, "2023": 2.5, "2024": 1.9 },
  { month: 18, "2021": 6.2, "2022": 5.4, "2023": 4.6, "2024": 3.4 },
  { month: 24, "2021": 9.8, "2022": 8.3, "2023": 7.1 },
  { month: 30, "2021": 13.1, "2022": 11.0 },
  { month: 36, "2021": 16.4, "2022": 13.8 },
  { month: 42, "2021": 19.2 },
  { month: 48, "2021": 21.6 },
];

/* --------------------------------------------------------------------- *
 * Validation / governance
 * --------------------------------------------------------------------- */
export const VALIDATION_ISSUES: ValidationIssue[] = [
  {
    id: "v1",
    code: "ESMA.RREL.41",
    title: "Missing valuation date on 12 exposures",
    severity: "blocker",
    scope: "Annex 2 · Underlying Exposures",
    detail:
      "Field RREL41 (Valuation Date) is null for 12 loans onboarded in May. ESMA submission will reject until populated.",
    affected: 12,
  },
  {
    id: "v2",
    code: "ESMA.RREL.18",
    title: "Geographic region outside NUTS3 lookup",
    severity: "blocker",
    scope: "Annex 2 · Collateral",
    detail:
      "3 postcodes did not resolve against the NUTS 2024 lookup. Manual override required before reporting.",
    affected: 3,
  },
  {
    id: "v3",
    code: "MI.LTV.RANGE",
    title: "Current LTV above expected ceiling",
    severity: "warning",
    scope: "MI · Stratification",
    detail:
      "8 loans report current LTV > 75%, above the product ceiling. Likely roll-up interest; review for data quality.",
    affected: 8,
  },
  {
    id: "v4",
    code: "MI.COUPON.OUTLIER",
    title: "Coupon outlier vs. product profile",
    severity: "warning",
    scope: "MI · Product",
    detail:
      "5 loans carry a coupon >150bps from the product mean. Confirm pricing exceptions are documented.",
    affected: 5,
  },
  {
    id: "v5",
    code: "GOV.RECON.BAL",
    title: "Balance reconciliation within tolerance",
    severity: "pass",
    scope: "Governance · Reconciliation",
    detail:
      "Canonical ledger balance reconciles to servicer tape within £1,240 (<0.001%).",
  },
  {
    id: "v6",
    code: "GOV.COVERAGE",
    title: "Field coverage above threshold",
    severity: "pass",
    scope: "Governance · Completeness",
    detail: "98.1% of mandatory MI fields populated, above the 97% gate.",
  },
];

/* --------------------------------------------------------------------- *
 * Pre-seeded artifacts shown on the landing canvas
 * --------------------------------------------------------------------- */
const now = "2026-05-31T08:14:00Z";

export const LANDING_ARTIFACTS: Artifact[] = [
  {
    id: "art-kpi-exec",
    kind: "kpi",
    title: "Executive Summary",
    description: "Headline portfolio metrics for the selected reporting date.",
    source: "ERM UK — Master · 31 May 2026",
    createdAt: now,
    pinned: true,
    data: { kind: "kpi", kpis: EXEC_KPIS },
  },
  {
    id: "art-region",
    kind: "chart",
    title: "Balance by Region",
    description: "Geographic concentration of current balance (£MM).",
    source: "Stratification · NUTS1",
    createdAt: now,
    data: {
      kind: "chart",
      chartType: "bar",
      xKey: "region",
      rows: REGION_ROWS,
      valueFormat: "gbp",
      unit: "MM",
      series: [{ key: "balance", label: "Balance (£MM)", color: "#919dd1" }],
    },
  },
  {
    id: "art-validation",
    kind: "validation",
    title: "Data Quality & Governance",
    description: "Validation issues affecting reporting readiness.",
    source: "Validation engine · Annex 2 + MI",
    createdAt: now,
    data: {
      kind: "validation",
      summary: { blockers: 2, warnings: 2, passed: 2, coverage: 98.1 },
      issues: VALIDATION_ISSUES,
    },
  },
];
