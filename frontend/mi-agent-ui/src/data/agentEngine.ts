/**
 * Mock MI Agent engine.
 *
 * Classifies a natural-language question into an intent and returns a narrative
 * response plus relevant artifacts. This is a stand-in for the real MI Agent
 * (see ../../../../mi_agent in the repo) — the shapes here are deliberately
 * close to a future API contract so swapping in a live `fetch` is mechanical.
 */

import type { AgentResponse, Artifact, Intent } from "@/types";
import { uid } from "@/lib/utils";
import {
  EXEC_KPIS,
  PIPELINE_BRIDGE,
  PIPELINE_FLOW,
  REGION_ROWS,
  STATIC_POOL_VINTAGE,
  VALIDATION_ISSUES,
} from "./mockData";

export const PROMPT_SUGGESTIONS: { label: string; intent: Intent }[] = [
  { label: "Show portfolio movement since last period", intent: "portfolio_overview" },
  { label: "Explain top concentration risks", intent: "concentration_risk" },
  { label: "Generate pipeline bridge to £100MM securitisation size", intent: "pipeline" },
  { label: "Show static pool performance by vintage", intent: "static_pools" },
  { label: "Summarise validation issues blocking reporting", intent: "validation" },
];

const MATCHERS: { intent: Intent; patterns: RegExp[] }[] = [
  {
    intent: "concentration_risk",
    patterns: [/concentrat/i, /\brisk\b/i, /\bregion/i, /\bgeograph/i, /exposure to/i],
  },
  {
    intent: "pipeline",
    patterns: [/pipeline/i, /forward/i, /bridge/i, /securitis/i, /funding/i, /fallout/i],
  },
  {
    intent: "static_pools",
    patterns: [/static pool/i, /vintage/i, /prepay/i, /redempt/i, /cpr/i, /seasoning/i],
  },
  {
    intent: "validation",
    patterns: [/validat/i, /governance/i, /blocking/i, /data quality/i, /esma/i, /reconcil/i],
  },
  {
    intent: "portfolio_overview",
    patterns: [/overview/i, /movement/i, /summary/i, /headline/i, /since last/i, /portfolio/i],
  },
];

export function classifyIntent(question: string): Intent {
  for (const { intent, patterns } of MATCHERS) {
    if (patterns.some((p) => p.test(question))) return intent;
  }
  return "unknown";
}

function stamp(): string {
  return new Date().toISOString();
}

/* ----------------------------- builders ------------------------------ */

function portfolioOverviewArtifacts(): Artifact[] {
  return [
    {
      id: uid("art"),
      kind: "kpi",
      title: "Portfolio Movement",
      description: "Period-over-period movement in headline metrics.",
      source: "ERM UK — Master · vs. 30 Apr 2026",
      createdAt: stamp(),
      data: { kind: "kpi", kpis: EXEC_KPIS },
    },
    {
      id: uid("art"),
      kind: "chart",
      title: "Funded vs. Pipeline Volume",
      description: "Monthly funded completions against live pipeline (£MM).",
      source: "Pipeline · trailing 6 months",
      createdAt: stamp(),
      data: {
        kind: "chart",
        chartType: "area",
        xKey: "month",
        rows: PIPELINE_FLOW,
        valueFormat: "gbp",
        unit: "MM",
        series: [
          { key: "pipeline", label: "Pipeline", color: "#919dd1" },
          { key: "funded", label: "Funded", color: "#36c2a8" },
        ],
      },
    },
  ];
}

function concentrationArtifacts(): Artifact[] {
  return [
    {
      id: uid("art"),
      kind: "chart",
      title: "Regional Concentration",
      description: "Current balance by region — top exposures highlighted.",
      source: "Stratification · NUTS1",
      createdAt: stamp(),
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
      id: uid("art"),
      kind: "table",
      title: "Concentration Detail",
      description: "Balance, share and weighted LTV by region.",
      source: "Stratification · NUTS1",
      createdAt: stamp(),
      data: {
        kind: "table",
        columns: [
          { key: "region", label: "Region", align: "left", format: "text" },
          { key: "balance", label: "Balance (£MM)", align: "right", format: "number", bar: true },
          { key: "share", label: "Share %", align: "right", format: "pct" },
          { key: "loans", label: "Loans", align: "right", format: "number" },
          { key: "ltv", label: "Wtd. LTV %", align: "right", format: "pct" },
        ],
        rows: REGION_ROWS,
      },
    },
  ];
}

function pipelineArtifacts(): Artifact[] {
  return [
    {
      id: uid("art"),
      kind: "chart",
      title: "Pipeline Bridge to £100MM",
      description: "Forward exposure build from current funded to target size.",
      source: "Forward exposure · expected funding model",
      createdAt: stamp(),
      data: {
        kind: "chart",
        chartType: "waterfall",
        xKey: "stage",
        rows: PIPELINE_BRIDGE,
        valueFormat: "gbp",
        unit: "MM",
        series: [{ key: "value", label: "£MM", color: "#919dd1" }],
      },
    },
    {
      id: uid("art"),
      kind: "chart",
      title: "Funded vs. Pipeline Volume",
      description: "Monthly funded completions against live pipeline (£MM).",
      source: "Pipeline · trailing 6 months",
      createdAt: stamp(),
      data: {
        kind: "chart",
        chartType: "line",
        xKey: "month",
        rows: PIPELINE_FLOW,
        valueFormat: "gbp",
        unit: "MM",
        series: [
          { key: "funded", label: "Funded", color: "#36c2a8" },
          { key: "pipeline", label: "Pipeline", color: "#919dd1" },
        ],
      },
    },
  ];
}

function staticPoolArtifacts(): Artifact[] {
  return [
    {
      id: uid("art"),
      kind: "chart",
      title: "Static Pool — Cumulative Redemption",
      description: "Cumulative redemption rate by origination vintage vs. seasoning.",
      source: "Static pools · monthly seasoning",
      createdAt: stamp(),
      data: {
        kind: "chart",
        chartType: "line",
        xKey: "month",
        rows: STATIC_POOL_VINTAGE,
        valueFormat: "pct",
        series: [
          { key: "2021", label: "2021 vintage", color: "#232d55" },
          { key: "2022", label: "2022 vintage", color: "#3d4a82" },
          { key: "2023", label: "2023 vintage", color: "#919dd1" },
          { key: "2024", label: "2024 vintage", color: "#36c2a8" },
        ],
      },
    },
  ];
}

function validationArtifacts(): Artifact[] {
  return [
    {
      id: uid("art"),
      kind: "validation",
      title: "Validation & Governance Summary",
      description: "Issues affecting reporting readiness, ranked by severity.",
      source: "Validation engine · Annex 2 + MI",
      createdAt: stamp(),
      data: {
        kind: "validation",
        summary: { blockers: 2, warnings: 2, passed: 2, coverage: 98.1 },
        issues: VALIDATION_ISSUES,
      },
    },
  ];
}

/* ----------------------------- responses ----------------------------- */

const NARRATIVES: Record<Intent, { narrative: string; assumptions: string[] }> = {
  portfolio_overview: {
    narrative:
      "Portfolio balance rose 2.4% to £842.6MM over the period, driven by 86 net new completions outpacing redemptions. Weighted-average LTV ticked up 0.6pp to 31.4%, consistent with interest roll-up on a seasoning book rather than new high-LTV lending.",
    assumptions: [
      "Comparison is against the prior month-end (30 Apr 2026).",
      "Balances are gross of accrued interest, balance-weighted where averaged.",
      "Redemptions recognised on servicer-confirmed settlement.",
    ],
  },
  concentration_risk: {
    narrative:
      "The book is concentrated in London and the South East, which together hold 41.2% of balance. London also carries the highest weighted LTV (33.8%), so geographic and valuation risk are correlated at the top of the distribution. No single region breaches the 25% single-region limit.",
    assumptions: [
      "Regions mapped via the NUTS 2024 lookup; 3 unresolved postcodes excluded.",
      "Share computed on current balance, not original advance.",
      "Limit framework: 25% single-region soft limit (risk_limits_config).",
    ],
  },
  pipeline: {
    narrative:
      "Current funded exposure of £84.2MM bridges to a forecast £100.0MM at the target securitisation date, after applying a 2.4MM expected fallout haircut across the offer, legals, valuation and application stages. Conversion-weighted pipeline implies the £100MM size is reachable within the current window.",
    assumptions: [
      "Fallout rates applied per stage from the expected-funding model.",
      "Pipeline values are conversion-weighted, not gross application value.",
      "Target size and date taken from pipeline_expected_funding config.",
    ],
  },
  static_pools: {
    narrative:
      "Redemption behaviour is stable across vintages, with the 2021 cohort reaching ~21.6% cumulative redemption by month 48. Newer vintages (2023–24) are tracking slightly slower at equivalent seasoning, consistent with the higher-rate environment reducing voluntary prepayment.",
    assumptions: [
      "Static pools held on origination cohort; no re-pooling.",
      "Redemption defined as full repayment (voluntary + mortality + LTC).",
      "Partial-period vintages shown only where seasoning is observed.",
    ],
  },
  validation: {
    narrative:
      "There are 2 blocking issues preventing ESMA submission: 12 exposures missing a valuation date (RREL41) and 3 postcodes unresolved against the NUTS lookup. Two warnings (LTV ceiling, coupon outliers) should be reviewed but do not block. Balance reconciliation and field coverage both pass their gates.",
    assumptions: [
      "Blockers gate the Annex 2 submission; warnings are advisory.",
      "Coverage gate is 97% of mandatory MI fields (currently 98.1%).",
      "Reconciliation tolerance is 0.01% of ledger balance.",
    ],
  },
  unknown: {
    narrative:
      "I wasn't able to map that to a specific MI query, so I've surfaced the default portfolio dashboard. Try one of the suggested questions, or ask about balances, concentration, pipeline, static pools or validation.",
    assumptions: ["Showing default landing artifacts for ERM UK — Master."],
  },
};

const BUILDERS: Record<Intent, () => Artifact[]> = {
  portfolio_overview: portfolioOverviewArtifacts,
  concentration_risk: concentrationArtifacts,
  pipeline: pipelineArtifacts,
  static_pools: staticPoolArtifacts,
  validation: validationArtifacts,
  unknown: () => [...portfolioOverviewArtifacts().slice(0, 1), ...concentrationArtifacts().slice(0, 1)],
};

/** Produce a mocked agent response for a question. */
export function runAgent(question: string): AgentResponse {
  const intent = classifyIntent(question);
  const { narrative, assumptions } = NARRATIVES[intent];
  return {
    intent,
    narrative,
    assumptions,
    artifacts: BUILDERS[intent](),
  };
}

/** Simulated latency for the streaming-style loading state (ms). */
export const MOCK_LATENCY_MS = 1100;
