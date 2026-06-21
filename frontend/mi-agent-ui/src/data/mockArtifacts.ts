/**
 * Mock artifact builders.
 *
 * Each builder returns a fully-formed, lineage-carrying `Artifact` so the
 * renderer is backend-shaped from day one. Figures are illustrative of a UK
 * Equity Release Mortgage (ERM) portfolio. When the MI Agent API lands, these
 * builders are replaced by deserialisation of the engine's artifact payloads —
 * the shapes are identical.
 */

import type {
  Artifact,
  ArtifactSource,
  ChartArtifact,
  KPI,
  KPIArtifact,
  MigrationCell,
  RiskArtifact,
  RiskGroup,
  ScenarioArtifact,
  TableArtifact,
  ValidationArtifact,
} from "@/domain";
import type { ReportingContext } from "@/domain";
import { THEME } from "@/lib/theme";
import { uid } from "@/lib/utils";

interface Ctx {
  asOf: string;
  portfolio: string;
}

function source(partial: Omit<ArtifactSource, "asOf" | "portfolio">, ctx: Ctx): ArtifactSource {
  return { ...partial, asOf: ctx.asOf, portfolio: ctx.portfolio };
}

const stamp = () => new Date().toISOString();

/* ------------------------------- KPIs -------------------------------- */

export const EXEC_KPIS: KPI[] = [
  { id: "kpi-aum", label: "Portfolio Balance", value: "£842.6MM", delta: "+2.4%", trend: "up", deltaIntent: "positive", hint: "vs. prior period" },
  { id: "kpi-loans", label: "Active Loans", value: "4,318", delta: "+86", trend: "up", deltaIntent: "positive", hint: "net of redemptions" },
  { id: "kpi-ltv", label: "Wtd. Avg. LTV", value: "31.4%", delta: "+0.6pp", trend: "up", deltaIntent: "negative", hint: "balance-weighted" },
  { id: "kpi-age", label: "Wtd. Avg. Borrower Age", value: "71.2", delta: "-0.3", trend: "down", deltaIntent: "neutral", hint: "years" },
  { id: "kpi-coupon", label: "Wtd. Avg. Coupon", value: "6.18%", delta: "+0.11pp", trend: "up", deltaIntent: "neutral", hint: "fixed for life" },
  { id: "kpi-dq", label: "Data Quality", value: "98.1%", delta: "+0.4pp", trend: "up", deltaIntent: "positive", hint: "fields validated" },
];

export function kpiArtifact(ctx: Ctx): KPIArtifact {
  return {
    id: uid("art"),
    type: "kpi",
    title: "Executive Summary",
    description: "Headline portfolio metrics for the selected reporting date.",
    source: source({ engine: "mi_agent.workflow", state: "total_funded", label: "Portfolio overview" }, ctx),
    createdAt: stamp(),
    mock: true,
    kpis: EXEC_KPIS,
  };
}

/* ---------------------------- Concentration -------------------------- */

const REGION_ROWS = [
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

export function regionChartArtifact(ctx: Ctx): ChartArtifact {
  return {
    id: uid("art"),
    type: "chart",
    title: "Balance by Region",
    description: "Geographic concentration of current balance (£MM).",
    source: source({ engine: "stratify", state: "total_funded", label: "Stratification · NUTS1", spec: { intent: "chart", chartType: "bar", dimension: "geographic_region_obligor", metric: "current_outstanding_balance", aggregation: "sum" } }, ctx),
    createdAt: stamp(),
    mock: true,
    chartType: "bar",
    xKey: "region",
    rows: REGION_ROWS,
    valueFormat: "gbp",
    unit: "MM",
    series: [{ key: "balance", label: "Balance (£MM)", color: THEME.peri }],
  };
}

export function concentrationTableArtifact(ctx: Ctx): TableArtifact {
  return {
    id: uid("art"),
    type: "table",
    title: "Concentration Detail",
    description: "Balance, share and weighted LTV by region.",
    source: source({ engine: "stratify", state: "total_funded", label: "Stratification · NUTS1" }, ctx),
    createdAt: stamp(),
    mock: true,
    columns: [
      { key: "region", label: "Region", align: "left", format: "text" },
      { key: "balance", label: "Balance (£MM)", align: "right", format: "number", bar: true },
      { key: "share", label: "Share %", align: "right", format: "pct" },
      { key: "loans", label: "Loans", align: "right", format: "number" },
      { key: "ltv", label: "Wtd. LTV %", align: "right", format: "pct" },
    ],
    rows: REGION_ROWS,
  };
}

/* ------------------------------ Pipeline ----------------------------- */

const PIPELINE_BRIDGE = [
  { stage: "Current Funded", value: 84.2, type: "base" },
  { stage: "Offer Issued", value: 6.8, type: "add" },
  { stage: "Legals", value: 5.1, type: "add" },
  { stage: "Valuation", value: 3.4, type: "add" },
  { stage: "Application", value: 2.9, type: "add" },
  { stage: "Expected Fallout", value: -2.4, type: "sub" },
  { stage: "Forecast Size", value: 100.0, type: "total" },
];

const PIPELINE_FLOW = [
  { month: "Dec", funded: 11.2, pipeline: 14.1 },
  { month: "Jan", funded: 12.8, pipeline: 15.6 },
  { month: "Feb", funded: 10.4, pipeline: 13.9 },
  { month: "Mar", funded: 13.9, pipeline: 17.2 },
  { month: "Apr", funded: 15.1, pipeline: 18.4 },
  { month: "May", funded: 16.7, pipeline: 19.8 },
];

export function pipelineBridgeArtifact(ctx: Ctx): ChartArtifact {
  return {
    id: uid("art"),
    type: "chart",
    title: "Pipeline Bridge to £100MM",
    description: "Forward exposure build from current funded to target size.",
    source: source({ engine: "pipeline_forward_risk", state: "total_forecast_funded", label: "Forward exposure · expected funding model", spec: { intent: "chart", chartType: "waterfall", state: "total_forecast_funded" } }, ctx),
    createdAt: stamp(),
    mock: true,
    chartType: "waterfall",
    xKey: "stage",
    rows: PIPELINE_BRIDGE,
    valueFormat: "gbp",
    unit: "MM",
    series: [{ key: "value", label: "£MM", color: THEME.peri }],
  };
}

export function pipelineFlowArtifact(ctx: Ctx, chartType: "area" | "line" = "line"): ChartArtifact {
  return {
    id: uid("art"),
    type: "chart",
    title: "Funded vs. Pipeline Volume",
    description: "Monthly funded completions against live pipeline (£MM).",
    source: source({ engine: "tab_pipeline", state: "total_pipeline", label: "Pipeline · trailing 6 months" }, ctx),
    createdAt: stamp(),
    mock: true,
    chartType,
    xKey: "month",
    rows: PIPELINE_FLOW,
    valueFormat: "gbp",
    unit: "MM",
    series: [
      { key: "funded", label: "Funded", color: THEME.positive },
      { key: "pipeline", label: "Pipeline", color: THEME.peri },
    ],
  };
}

/* ---------------------------- Static pools --------------------------- */

const STATIC_POOL_VINTAGE: Array<Record<string, number>> = [
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

export function staticPoolArtifact(ctx: Ctx): ChartArtifact {
  return {
    id: uid("art"),
    type: "chart",
    title: "Static Pool — Cumulative Redemption",
    description: "Cumulative redemption rate by origination vintage vs. seasoning.",
    source: source({ engine: "static_pools_core", state: "cohort_by_date", label: "Static pools · monthly seasoning" }, ctx),
    createdAt: stamp(),
    mock: true,
    chartType: "line",
    xKey: "month",
    rows: STATIC_POOL_VINTAGE,
    valueFormat: "pct",
    series: [
      { key: "2021", label: "2021 vintage", color: THEME.categorical[0] },
      { key: "2022", label: "2022 vintage", color: THEME.categorical[1] },
      { key: "2023", label: "2023 vintage", color: THEME.categorical[3] },
      { key: "2024", label: "2024 vintage", color: THEME.categorical[4] },
    ],
  };
}

/* ----------------------------- Validation ---------------------------- */

export function validationArtifact(ctx: Ctx): ValidationArtifact {
  return {
    id: uid("art"),
    type: "validation",
    title: "Data Quality & Governance",
    description: "Validation issues affecting reporting readiness.",
    source: source({ engine: "mi_prep.assert_trusted_canonical", label: "Validation engine · Annex 2 + MI" }, ctx),
    createdAt: stamp(),
    mock: true,
    summary: { blockers: 2, warnings: 2, passed: 2, coverage: 98.1 },
    issues: [
      { id: "v1", code: "ESMA.RREL.41", title: "Missing valuation date on 12 exposures", severity: "blocker", scope: "Annex 2 · Underlying Exposures", detail: "Field RREL41 (Valuation Date) is null for 12 loans onboarded in May. ESMA submission will reject until populated.", affected: 12 },
      { id: "v2", code: "ESMA.RREL.18", title: "Geographic region outside NUTS3 lookup", severity: "blocker", scope: "Annex 2 · Collateral", detail: "3 postcodes did not resolve against the NUTS 2024 lookup. Manual override required before reporting.", affected: 3 },
      { id: "v3", code: "MI.LTV.RANGE", title: "Current LTV above expected ceiling", severity: "warning", scope: "MI · Stratification", detail: "8 loans report current LTV > 75%, above the product ceiling. Likely roll-up interest; review for data quality.", affected: 8 },
      { id: "v4", code: "MI.COUPON.OUTLIER", title: "Coupon outlier vs. product profile", severity: "warning", scope: "MI · Product", detail: "5 loans carry a coupon >150bps from the product mean. Confirm pricing exceptions are documented.", affected: 5 },
      { id: "v5", code: "GOV.RECON.BAL", title: "Balance reconciliation within tolerance", severity: "pass", scope: "Governance · Reconciliation", detail: "Canonical ledger balance reconciles to servicer tape within £1,240 (<0.001%)." },
      { id: "v6", code: "GOV.COVERAGE", title: "Field coverage above threshold", severity: "pass", scope: "Governance · Completeness", detail: "98.1% of mandatory MI fields populated, above the 97% gate." },
    ],
  };
}

/* ------------------------------- Risk -------------------------------- */

const REGION_LIMITS: RiskGroup[] = [
  { name: "London (UKI)", balance: 184.2, share: 0.219, status: "amber", limit: 0.3, approaching: false },
  { name: "South East (UKJ)", balance: 162.7, share: 0.193, status: "green", limit: 0.3 },
  { name: "South West (UKK)", balance: 98.4, share: 0.117, status: "red", limit: 0.1, approaching: false },
  { name: "East of England (UKH)", balance: 86.1, share: 0.102, status: "red", limit: 0.1 },
  { name: "North West (UKD)", balance: 74.9, share: 0.089, status: "green", limit: 0.15 },
  { name: "Yorkshire (UKE)", balance: 52.8, share: 0.063, status: "green", limit: 0.1 },
  { name: "Scotland (UKM)", balance: 48.6, share: 0.058, status: "green", limit: 0.1 },
];

export function riskConcentrationArtifact(ctx: Ctx): RiskArtifact {
  return {
    id: uid("art"),
    type: "risk",
    title: "Regional Concentration vs. Limits",
    description: "Single-region exposure against the concentration limit framework.",
    source: source({ engine: "risk_monitor", state: "total_funded", label: "Risk monitor · concentration", spec: { intent: "summary", riskMode: "concentration", dimension: "geographic_region_obligor" } }, ctx),
    createdAt: stamp(),
    mock: true,
    warnings: ["2 regional limits breached (UKK, UKH); review before securitisation."],
    mode: "limits",
    dimension: "geographic_region_obligor",
    groups: REGION_LIMITS,
  };
}

const RISK_AXIS = ["A", "B", "C", "D", "E"];
const MIGRATION: MigrationCell[] = [
  { from: "A", to: "A", balance: 312.4, share: 0.371, movement: "unchanged" },
  { from: "A", to: "B", balance: 18.2, share: 0.022, movement: "deteriorated" },
  { from: "B", to: "A", balance: 9.1, share: 0.011, movement: "improved" },
  { from: "B", to: "B", balance: 204.6, share: 0.243, movement: "unchanged" },
  { from: "B", to: "C", balance: 14.7, share: 0.017, movement: "deteriorated" },
  { from: "C", to: "B", balance: 6.3, share: 0.007, movement: "improved" },
  { from: "C", to: "C", balance: 142.8, share: 0.169, movement: "unchanged" },
  { from: "C", to: "D", balance: 8.9, share: 0.011, movement: "deteriorated" },
  { from: "D", to: "D", balance: 61.2, share: 0.073, movement: "unchanged" },
  { from: "D", to: "E", balance: 4.1, share: 0.005, movement: "deteriorated" },
  { from: "E", to: "E", balance: 26.4, share: 0.031, movement: "unchanged" },
];

export function riskMigrationArtifact(ctx: Ctx): RiskArtifact {
  return {
    id: uid("art"),
    type: "risk",
    title: "Risk Grade Migration",
    description: "Period-over-period internal risk-grade transitions by balance share.",
    source: source({ engine: "risk_monitor", state: "total_funded", label: "Risk monitor · migration", spec: { intent: "summary", riskMode: "migration", dimension: "internal_risk_grade" } }, ctx),
    createdAt: stamp(),
    mock: true,
    mode: "migration",
    dimension: "internal_risk_grade",
    axis: RISK_AXIS,
    matrix: MIGRATION,
  };
}

/* ----------------------------- Scenario ------------------------------ */

export function scenarioArtifact(ctx: Ctx): ScenarioArtifact {
  const projection = Array.from({ length: 11 }, (_, i) => {
    const year = i;
    const balance = 842.6 * Math.pow(1.052, year) * Math.pow(0.96, year);
    const propertyValue = 2680 * Math.pow(1.02, year);
    const ltv = (balance / propertyValue) * 100;
    const nnegLoss = Math.max(0, (ltv - 70) * 0.18) * (year > 0 ? 1 : 0);
    return {
      year,
      balance: Math.round(balance * 10) / 10,
      propertyValue: Math.round(propertyValue),
      ltv: Math.round(ltv * 10) / 10,
      nnegLoss: Math.round(nnegLoss * 100) / 100,
      cumulativeNneg: 0,
    };
  });
  let cum = 0;
  for (const p of projection) {
    cum += p.nnegLoss;
    p.cumulativeNneg = Math.round(cum * 100) / 100;
  }
  return {
    id: uid("art"),
    type: "scenario",
    title: "Scenario — Base Case Projection",
    description: "10-year balance run-off, LTV path and expected NNEG under base assumptions.",
    source: source({ engine: "scenario_engine", label: "Scenario engine · base case" }, ctx),
    createdAt: stamp(),
    mock: true,
    assumptions: {
      "HPI growth": "+2.0% p.a.",
      "Rate shift": "+0bps",
      "Voluntary prepay": "4.0% p.a.",
      Mortality: "base table",
      "Move-to-care": "1.5% p.a.",
      "Sale costs": "5.0%",
    },
    projection,
  };
}

/* ------------------------- Landing artifacts ------------------------- */

export function landingArtifacts(reporting: ReportingContext, portfolioId: string): Artifact[] {
  const ctx: Ctx = { asOf: reporting.asOf, portfolio: portfolioId };
  const kpi = kpiArtifact(ctx);
  kpi.pinned = true;
  return [kpi, regionChartArtifact(ctx), validationArtifact(ctx)];
}
