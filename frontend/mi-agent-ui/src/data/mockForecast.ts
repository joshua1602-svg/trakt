/**
 * Deterministic mock forecast-snapshot envelopes for offline / demo mode.
 *
 * Mirror the real `GET /mi/forecast/snapshot` composition for `client_001`:
 * the funded balance (from the funded spine) + the Phase 1 pipeline snapshot of
 * the M2L KFI fixture pack + config stage probabilities.
 *
 * Pipeline dates are weekly-operational and DISTINCT from the funded reporting
 * date: the November scope (`pipelineSourceFolderDate` 2025-11-01) selects the
 * latest weekly extract (`pipelineAsOfDate`/`pipelineExtractDate` 2025-12-01),
 * while the funded book reports at month-end (`fundedReportingDate` 2025-11-30).
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
  { month: "2025-12", caseCount: 6, expectedFundedAmount: 905_000, weightedExpectedFundedAmount: 533_250 },
  { month: "2026-01", caseCount: 4, expectedFundedAmount: 850_000, weightedExpectedFundedAmount: 530_000 },
];

const NOV_PIPELINE: PipelineSnapshot = {
  ok: true,
  recordType: "pipeline",
  portfolioId: "client_001/mi_2025_11",
  client_id: "client_001",
  runId: "mi_2025_11",
  pipelineAsOfDate: "2025-12-01",
  pipelineExtractDate: "2025-12-01",
  pipelineSourceFolderDate: "2025-11-01",
  sourceFile: "M2L_KFI_and_Pipeline_2025_12_01_115711.csv",
  pipelineRowCount: 10,
  pipelineAmount: 1_755_000,
  expectedFundedAmount: 1_755_000,
  weightedExpectedFundedAmount: 1_063_250,
  completionProbabilityBasis: "mixed_historical_and_config",
  stageBreakdown: NOV_STAGES,
  expectedCompletionBreakdown: NOV_COMPLETION,
  // Capped to top 10 (+ Other) by the backend — 14 brokers -> 9 + Other.
  brokerBreakdown: [
    { key: "Broker Alpha", caseCount: 30, pipelineAmount: 530_000, weightedExpectedFundedAmount: 320_000 },
    { key: "Broker Beta", caseCount: 28, pipelineAmount: 480_000, weightedExpectedFundedAmount: 300_000 },
    { key: "Broker Gamma", caseCount: 22, pipelineAmount: 390_000, weightedExpectedFundedAmount: 220_000 },
    { key: "Broker Delta", caseCount: 18, pipelineAmount: 355_000, weightedExpectedFundedAmount: 223_250 },
    { key: "Broker Epsilon", caseCount: 14, pipelineAmount: 300_000, weightedExpectedFundedAmount: 180_000 },
    { key: "Broker Zeta", caseCount: 12, pipelineAmount: 260_000, weightedExpectedFundedAmount: 150_000 },
    { key: "Broker Eta", caseCount: 10, pipelineAmount: 220_000, weightedExpectedFundedAmount: 120_000 },
    { key: "Broker Theta", caseCount: 8, pipelineAmount: 180_000, weightedExpectedFundedAmount: 95_000 },
    { key: "Broker Iota", caseCount: 6, pipelineAmount: 140_000, weightedExpectedFundedAmount: 70_000 },
    { key: "Other", caseCount: 20, pipelineAmount: 300_000, weightedExpectedFundedAmount: 150_000, isOther: true, categoriesIncluded: 5, sharePct: 9.0 },
  ],
  regionBreakdown: [
    { key: "South West", caseCount: 2, pipelineAmount: 420_000, weightedExpectedFundedAmount: 260_000 },
    { key: "West Midlands", caseCount: 2, pipelineAmount: 410_000, weightedExpectedFundedAmount: 200_000 },
    { key: "London", caseCount: 3, pipelineAmount: 480_000, weightedExpectedFundedAmount: 350_000 },
  ],
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
  fundedReportingDate: "2025-11-30",
  pipelineAsOfDate: "2025-12-01",
  pipelineExtractDate: "2025-12-01",
  pipelineSourceFolderDate: "2025-11-01",
  fundedBalance: 8_902_999.7,
  fundedLoanCount: 73,
  pipelineAvailable: true,
  pipelineSnapshot: NOV_PIPELINE,
  forecastBridge: {
    portfolioId: "client_001/mi_2025_11",
    client_id: "client_001",
    runId: "mi_2025_11",
    fundedReportingDate: "2025-11-30",
    pipelineAsOfDate: "2025-12-01",
    pipelineExtractDate: "2025-12-01",
    pipelineSourceFolderDate: "2025-11-01",
    fundedBalance: 8_902_999.7,
    fundedLoanCount: 73,
    pipelineAvailable: true,
    pipelineAmount: 1_755_000,
    pipelineCaseCount: 10,
    weightedExpectedFundedAmount: 1_063_250,
    forecastFundedBalance: 9_966_249.7,
    forecastLoanCount: 83,
    completionProbabilityBasis: "mixed_historical_and_config",
    grossPipelineAmount: 1_755_000,
    excludedFromWeightingAmount: 95_000,
    excludedCaseCount: 1,
    activeGrossPipelineAmount: 1_660_000,
    amountWeightedHistorical: 840_000,
    amountWeightedConfig: 820_000,
    blendedWeightedConversion: 0.6406,
    expectedCompletionBreakdown: NOV_COMPLETION,
    stageBreakdown: NOV_STAGES,
    forecastReadiness: { status: "ready", missingRequiredFields: [], warnings: [] },
    dataQuality: { blockers: [], warnings: [], info: [] },
  },
  watchlist: [
    {
      category: "withdrawn_excluded_from_weighting",
      severity: "info",
      title: "1 withdrawn/inactive case excluded from forecast probability weighting",
      detail: "1 withdrawn/inactive case(s) excluded from forecast probability weighting. By stage [WITHDRAWN:1]. Intentionally excluded from weighted forecast.",
      count: 1,
      byStage: { WITHDRAWN: 1 },
      excluded: true,
      weighted: false,
    },
  ],
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
  fundedReportingDate: "2025-10-31",
  pipelineAsOfDate: "2025-10-01",
  pipelineExtractDate: "2025-10-01",
  pipelineSourceFolderDate: "2025-10-01",
  fundedBalance: 4_207_999.95,
  fundedLoanCount: 33,
  pipelineAvailable: true,
  pipelineSnapshot: {
    ...NOV_PIPELINE,
    portfolioId: "client_001/mi_2025_10",
    runId: "mi_2025_10",
    pipelineAsOfDate: "2025-10-01",
    pipelineExtractDate: "2025-10-01",
    pipelineSourceFolderDate: "2025-10-01",
    sourceFile: "M2L_KFI_and_Pipeline_2025_10_01.csv",
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
    fundedReportingDate: "2025-10-31",
    pipelineAsOfDate: "2025-10-01",
    pipelineExtractDate: "2025-10-01",
    pipelineSourceFolderDate: "2025-10-01",
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
