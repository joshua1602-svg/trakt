/**
 * Mock response composition — pure functions mapping an AgentRequest to an
 * AgentResponse. Kept separate from the client (transport) and the artifact
 * builders (data) so each layer is independently testable.
 *
 * Mirrors the deterministic interpreter in mi_agent/interpreter/deterministic.py:
 * keyword routing → intent → narrative + assumptions + typed artifacts.
 */

import type { AgentRequest, AgentResponse, Artifact, Intent } from "@/domain";
import {
  concentrationTableArtifact,
  kpiArtifact,
  pipelineBridgeArtifact,
  pipelineFlowArtifact,
  regionChartArtifact,
  riskConcentrationArtifact,
  riskMigrationArtifact,
  scenarioArtifact,
  staticPoolArtifact,
  validationArtifact,
} from "./mockArtifacts";

export const PROMPT_SUGGESTIONS: { label: string; intent: Intent }[] = [
  { label: "Show portfolio movement since last period", intent: "portfolio_overview" },
  { label: "Explain top concentration risks", intent: "concentration_risk" },
  { label: "Generate pipeline bridge to £100MM securitisation size", intent: "pipeline" },
  { label: "Show static pool performance by vintage", intent: "static_pools" },
  { label: "Check risk-grade migration and limit breaches", intent: "risk_monitoring" },
  { label: "Project the book under a base-case scenario", intent: "scenario" },
  { label: "Summarise validation issues blocking reporting", intent: "validation" },
];

const MATCHERS: { intent: Intent; patterns: RegExp[] }[] = [
  { intent: "risk_monitoring", patterns: [/migration/i, /breach/i, /\blimit/i, /risk grade/i, /rag\b/i, /monitor/i] },
  { intent: "scenario", patterns: [/scenario/i, /stress/i, /project/i, /run.?off/i, /nneg/i, /hpi/i, /forecast loss/i] },
  { intent: "concentration_risk", patterns: [/concentrat/i, /\bregion/i, /\bgeograph/i, /exposure to/i, /single.?name/i] },
  { intent: "pipeline", patterns: [/pipeline/i, /forward/i, /bridge/i, /securitis/i, /funding/i, /fallout/i] },
  { intent: "static_pools", patterns: [/static pool/i, /vintage/i, /prepay/i, /redempt/i, /cpr/i, /seasoning/i] },
  { intent: "validation", patterns: [/validat/i, /governance/i, /blocking/i, /data quality/i, /esma/i, /reconcil/i] },
  { intent: "portfolio_overview", patterns: [/overview/i, /movement/i, /summary/i, /headline/i, /since last/i, /portfolio/i] },
];

export function classifyIntent(question: string): Intent {
  for (const { intent, patterns } of MATCHERS) {
    if (patterns.some((p) => p.test(question))) return intent;
  }
  return "unknown";
}

interface IntentSpec {
  narrative: string;
  assumptions: string[];
  interpreted: string;
  build: (ctx: { asOf: string; portfolio: string }) => Artifact[];
}

const INTENT_SPECS: Record<Intent, IntentSpec> = {
  portfolio_overview: {
    interpreted: "Portfolio overview · total_funded · period-over-period movement",
    narrative:
      "Portfolio balance rose 2.4% to £842.6MM over the period, driven by 86 net new completions outpacing redemptions. Weighted-average LTV ticked up 0.6pp to 31.4%, consistent with interest roll-up on a seasoning book rather than new high-LTV lending.",
    assumptions: [
      "Comparison is against the prior month-end.",
      "Balances are gross of accrued interest, balance-weighted where averaged.",
      "Redemptions recognised on servicer-confirmed settlement.",
    ],
    build: (c) => [kpiArtifact(c), pipelineFlowArtifact(c, "area")],
  },
  concentration_risk: {
    interpreted: "Concentration · total_funded · balance by geographic_region_obligor",
    narrative:
      "The book is concentrated in London and the South East, which together hold 41.2% of balance. London also carries the highest weighted LTV (33.8%), so geographic and valuation risk are correlated at the top of the distribution. No single region breaches the 25% single-region soft limit.",
    assumptions: [
      "Regions mapped via the NUTS 2024 lookup; 3 unresolved postcodes excluded.",
      "Share computed on current balance, not original advance.",
      "Limit framework: 25% single-region soft limit (risk_limits_config).",
    ],
    build: (c) => [regionChartArtifact(c), concentrationTableArtifact(c)],
  },
  pipeline: {
    interpreted: "Forward exposure · total_forecast_funded · waterfall bridge",
    narrative:
      "Current funded exposure of £84.2MM bridges to a forecast £100.0MM at the target securitisation date, after a 2.4MM expected fallout haircut across the offer, legals, valuation and application stages. Conversion-weighted pipeline implies the £100MM size is reachable within the current window.",
    assumptions: [
      "Fallout rates applied per stage from the expected-funding model.",
      "Pipeline values are conversion-weighted, not gross application value.",
      "Target size and date from pipeline_expected_funding config.",
    ],
    build: (c) => [pipelineBridgeArtifact(c), pipelineFlowArtifact(c, "line")],
  },
  static_pools: {
    interpreted: "Static pools · cohort_by_date · cumulative redemption by vintage",
    narrative:
      "Redemption behaviour is stable across vintages, with the 2021 cohort reaching ~21.6% cumulative redemption by month 48. Newer vintages (2023–24) are tracking slightly slower at equivalent seasoning, consistent with the higher-rate environment reducing voluntary prepayment.",
    assumptions: [
      "Static pools held on origination cohort; no re-pooling.",
      "Redemption defined as full repayment (voluntary + mortality + LTC).",
      "Partial-period vintages shown only where seasoning is observed.",
    ],
    build: (c) => [staticPoolArtifact(c)],
  },
  risk_monitoring: {
    interpreted: "Risk monitor · concentration (limits) + grade migration",
    narrative:
      "Two regional limits are breached: South West (UKK) at 11.7% vs a 10% cap and East of England (UKH) at 10.2% vs 10%. Risk-grade migration is broadly stable — 3.3% of balance deteriorated one notch against 1.8% improving — with no movement into the watchlist grade E beyond run-rate.",
    assumptions: [
      "Concentration RAG thresholds: amber 20%, red 30% (default); regional caps per risk_limits_config.",
      "Migration computed baseline (prior snapshot) vs current; direction from configured grade ordering A→G.",
      "Below-minimum groups suppressed from RAG.",
    ],
    build: (c) => [riskConcentrationArtifact(c), riskMigrationArtifact(c)],
  },
  scenario: {
    interpreted: "Scenario engine · base case · 10-year projection",
    narrative:
      "Under the base case (+2% HPI, +0bps rates, 4% prepay), the book runs off to ~£0.97BN of accrued balance by year 10 while property values grow modestly, holding portfolio LTV around the low-40s and keeping expected NNEG losses immaterial. NNEG exposure only becomes sensitive if HPI turns negative for a sustained period.",
    assumptions: [
      "Balance accretes at the weighted coupon net of exits (mortality, move-to-care, voluntary prepay).",
      "NNEG loss = max(balance − property·(1 − sale costs), 0); sale costs 5%.",
      "Deterministic central projection — not a stochastic distribution.",
    ],
    build: (c) => [scenarioArtifact(c)],
  },
  validation: {
    interpreted: "Validation & governance · Annex 2 + MI gates",
    narrative:
      "There are 2 blocking issues preventing ESMA submission: 12 exposures missing a valuation date (RREL41) and 3 postcodes unresolved against the NUTS lookup. Two warnings (LTV ceiling, coupon outliers) should be reviewed but do not block. Balance reconciliation and field coverage both pass their gates.",
    assumptions: [
      "Blockers gate the Annex 2 submission; warnings are advisory.",
      "Coverage gate is 97% of mandatory MI fields (currently 98.1%).",
      "Reconciliation tolerance is 0.01% of ledger balance.",
    ],
    build: (c) => [validationArtifact(c)],
  },
  unknown: {
    interpreted: "Unrecognised query · default dashboard",
    narrative:
      "I wasn't able to map that to a specific MI query, so I've surfaced the default portfolio dashboard. Try one of the suggested questions, or ask about balances, concentration, pipeline, static pools, risk monitoring, scenarios or validation.",
    assumptions: ["Showing default landing artifacts for the selected portfolio."],
    build: (c) => [kpiArtifact(c), regionChartArtifact(c)],
  },
};

export function buildAgentResponse(request: AgentRequest): AgentResponse {
  const intent = classifyIntent(request.question);
  const specDef = INTENT_SPECS[intent];
  const ctx = { asOf: request.reporting.asOf, portfolio: request.portfolio.id };
  const artifacts = specDef.build(ctx);
  const warnings = artifacts.flatMap((a) => a.warnings ?? []);
  return {
    ok: true,
    question: request.question,
    intent,
    interpreted: specDef.interpreted,
    narrative: specDef.narrative,
    assumptions: specDef.assumptions,
    artifacts,
    warnings,
    spec: artifacts.find((a) => a.source.spec)?.source.spec,
  };
}
