/**
 * Deterministic mock forecast-snapshot envelopes for offline / demo mode.
 *
 * Mirror the real `GET /mi/forecast/snapshot` composition for `client_001`:
 * the funded balance (from the funded spine) + the Phase 1 pipeline snapshot of
 * the M2L KFI fixture pack + config stage probabilities. Numbers match the
 * committed pipeline fixtures (Nov: 10 cases, £1.755MM, weighted £1.06325MM).
 * Served only via the mock client — NOT hardcoded prototype options.
 */

import type {
  ExpectedCompletionBucket,
  ForecastSnapshot,
  PipelineSnapshot,
  PipelineStageBucket,
} from "@/domain";

const NOV_STAGES: PipelineStageBucket[] = [
  { stage: "APPLICATION", caseCount: 3, pipelineAmount: 485_000, weightedExpectedFundedAmount: 218_250 },
  { stage: "COMPLETED", caseCount: 2, pipelineAmount: 450_000, weightedExpectedFundedAmount: 450_000 },
  { stage: "KFI", caseCount: 2, pipelineAmount: 400_000, weightedExpectedFundedAmount: 80_000 },
  { stage: "OFFER", caseCount: 3, pipelineAmount: 420_000, weightedExpectedFundedAmount: 315_000 },
];

const NOV_COMPLETION: ExpectedCompletionBucket[] = [
  { month: "2025-10", caseCount: 2, expectedFundedAmount: 450_000, weightedExpectedFundedAmount: 450_000 },
  { month: "2025-12", caseCount: 6, expectedFundedAmount: 905_000, weightedExpectedFundedAmount: 533_250 },
  { month: "2026-01", caseCount: 2, expectedFundedAmount: 400_000, weightedExpectedFundedAmount: 80_000 },
];

const NOV_PIPELINE: PipelineSnapshot = {
  ok: true,
  recordType: "pipeline",
  portfolioId: "client_001/mi_2025_11",
  client_id: "client_001",
  runId: "mi_2025_11",
  reportingDate: "2025-11-30",
  pipelineRowCount: 10,
  pipelineAmount: 1_755_000,
  expectedFundedAmount: 1_755_000,
  weightedExpectedFundedAmount: 1_063_250,
  stageBreakdown: NOV_STAGES,
  expectedCompletionBreakdown: NOV_COMPLETION,
  availableMetrics: ["current_outstanding_balance", "expected_funded_amount", "weighted_expected_funded_amount"],
  availableDimensions: ["pipeline_stage", "broker_channel", "geographic_region_obligor", "ltv_bucket"],
  missingDimensions: [],
  dataQuality: [],
  fieldCorrelationToFunded: {
    collateral_geography: { funded_correlation: ["geographic_region_obligor"], available: true },
    broker_channel: { funded_correlation: ["broker_channel", "origination_channel"], available: true },
  },
  forecastReadiness: { forecast_ready: true },
};

const NOV_FORECAST: ForecastSnapshot = {
  ok: true,
  portfolioId: "client_001/mi_2025_11",
  client_id: "client_001",
  runId: "mi_2025_11",
  reportingDate: "2025-11-30",
  fundedBalance: 8_902_999.7,
  fundedLoanCount: 73,
  pipelineAvailable: true,
  pipelineSnapshot: NOV_PIPELINE,
  forecastBridge: {
    portfolioId: "client_001/mi_2025_11",
    client_id: "client_001",
    runId: "mi_2025_11",
    reportingDate: "2025-11-30",
    fundedBalance: 8_902_999.7,
    fundedLoanCount: 73,
    pipelineAvailable: true,
    pipelineAmount: 1_755_000,
    pipelineCaseCount: 10,
    weightedExpectedFundedAmount: 1_063_250,
    forecastFundedBalance: 9_966_249.7,
    forecastLoanCount: 83,
    completionProbabilityBasis: "stage_config",
    expectedCompletionBreakdown: NOV_COMPLETION,
    stageBreakdown: NOV_STAGES,
    forecastReadiness: { status: "ready", missingRequiredFields: [], warnings: [] },
    dataQuality: { blockers: [], warnings: [], info: [] },
  },
  watchlist: [],
};

const OCT_STAGES: PipelineStageBucket[] = [
  { stage: "APPLICATION", caseCount: 2, pipelineAmount: 220_000, weightedExpectedFundedAmount: 99_000 },
  { stage: "COMPLETED", caseCount: 1, pipelineAmount: 90_000, weightedExpectedFundedAmount: 90_000 },
  { stage: "KFI", caseCount: 2, pipelineAmount: 390_000, weightedExpectedFundedAmount: 78_000 },
  { stage: "OFFER", caseCount: 2, pipelineAmount: 450_000, weightedExpectedFundedAmount: 337_500 },
  { stage: "WITHDRAWN", caseCount: 1, pipelineAmount: 80_000, weightedExpectedFundedAmount: null },
];

const OCT_FORECAST: ForecastSnapshot = {
  ok: true,
  portfolioId: "client_001/mi_2025_10",
  client_id: "client_001",
  runId: "mi_2025_10",
  reportingDate: "2025-10-31",
  fundedBalance: 4_207_999.95,
  fundedLoanCount: 33,
  pipelineAvailable: true,
  pipelineSnapshot: {
    ...NOV_PIPELINE,
    portfolioId: "client_001/mi_2025_10",
    runId: "mi_2025_10",
    reportingDate: "2025-10-31",
    pipelineRowCount: 8,
    pipelineAmount: 1_230_000,
    expectedFundedAmount: 1_230_000,
    weightedExpectedFundedAmount: 604_500,
    stageBreakdown: OCT_STAGES,
    expectedCompletionBreakdown: [],
  },
  forecastBridge: {
    portfolioId: "client_001/mi_2025_10",
    client_id: "client_001",
    runId: "mi_2025_10",
    reportingDate: "2025-10-31",
    fundedBalance: 4_207_999.95,
    fundedLoanCount: 33,
    pipelineAvailable: true,
    pipelineAmount: 1_230_000,
    pipelineCaseCount: 8,
    weightedExpectedFundedAmount: 604_500,
    forecastFundedBalance: 4_812_499.95,
    forecastLoanCount: 41,
    completionProbabilityBasis: "stage_config",
    expectedCompletionBreakdown: [],
    stageBreakdown: OCT_STAGES,
    forecastReadiness: { status: "partial", missingRequiredFields: [], warnings: ["expected completion date present but empty for some rows"] },
    dataQuality: { blockers: [], warnings: [], info: [{ check: "expected_completion_date_partial", severity: "info", detail: "expected completion date populated for some rows" }] },
  },
  watchlist: [
    { category: "missing_completion_probability", severity: "warning", title: "1 case without a completion probability", detail: "1/8 rows have no row or config stage probability; excluded from weighted forecast.", count: 1 },
  ],
};

const BY_RUN: Record<string, ForecastSnapshot> = {
  "client_001/mi_2025_10": OCT_FORECAST,
  "client_001/mi_2025_11": NOV_FORECAST,
};

export function mockForecastSnapshot(portfolioId: string): ForecastSnapshot {
  return BY_RUN[portfolioId] ?? NOV_FORECAST;
}
