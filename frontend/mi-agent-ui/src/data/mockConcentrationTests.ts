/**
 * Synthetic concentration-test fixtures for mock mode and tests.
 * Mirrors `mi_agent_api.concentration_tests_api` — an APPROVED configuration
 * with pass / warning / breach / unavailable rows, prior-period movement and
 * a reconciling drill-through.
 */

import type {
  ConcentrationDrillthrough,
  ConcentrationHistory,
  ConcentrationTest,
  ConcentrationTestsSnapshot,
} from "@/domain";

const APPROVAL = {
  approvedBy: "Demo Operator",
  approvedAt: "2026-01-10T09:00:00+00:00",
  approvalComments: "Approved against the facility schedule.",
};

function test(partial: Partial<ConcentrationTest>): ConcentrationTest {
  return {
    testId: "ct_demo",
    metricId: "geo_region_share",
    displayName: "Demo test",
    category: "geography",
    reportingDate: "2025-11-30",
    currentValue: null,
    priorValue: null,
    priorReportingDate: "2025-10-31",
    priorAvailable: true,
    absoluteChange: null,
    percentagePointChange: null,
    relativeChange: null,
    threshold: null,
    operator: "max",
    warningFraction: 0.9,
    unit: "percent",
    utilization: null,
    headroom: null,
    status: "unavailable",
    priorStatus: null,
    statusTransition: null,
    deteriorated: false,
    breachAmount: null,
    dataStatus: "ok",
    missingFields: [],
    numeratorValue: null,
    denominatorValue: null,
    denominatorBasis: "current_balance",
    loansInNumerator: 0,
    totalLoans: 1250,
    severity: "high",
    effectiveDate: "2026-01-01",
    expiryDate: null,
    notes: "",
    resolvedColumns: {},
    parameters: {},
    definition: {
      numerator: "",
      aggregation: "sum",
      description: "",
      definitionNotes: "",
    },
    provenance: {
      sourceReference: "facility_schedule_2026.txt",
      sourceText: "",
      locator: "",
      proposalId: "ctp_demo",
      ...APPROVAL,
    },
    configurationVersion: 2,
    evaluatedAt: "2025-12-01T08:00:00+00:00",
    ...partial,
  } as ConcentrationTest;
}

const TESTS: ConcentrationTest[] = [
  test({
    testId: "ct_east_anglia",
    displayName: "Regional exposure — East Anglia",
    currentValue: 18.4,
    priorValue: 17.1,
    absoluteChange: 1.3,
    percentagePointChange: 1.3,
    relativeChange: 7.6,
    threshold: 25,
    utilization: 73.6,
    headroom: 6.6,
    status: "pass",
    priorStatus: "pass",
    numeratorValue: 33_120_000,
    denominatorValue: 180_000_000,
    loansInNumerator: 231,
    parameters: { regions: ["East Anglia"] },
    definition: {
      numerator: "Sum of balance for loans whose region is in `regions`.",
      aggregation: "sum",
      description:
        "Balance-weighted share of the portfolio secured on properties in one or more named regions.",
      definitionNotes: "",
    },
    provenance: {
      sourceReference: "facility_schedule_2026.txt",
      sourceText:
        "1.1 The aggregate Current Balance of Loans secured on Properties located in East Anglia must not exceed 25% of the Portfolio.",
      locator: "clause 1.1",
      proposalId: "ctp_ea",
      ...APPROVAL,
    },
  }),
  test({
    testId: "ct_london_se",
    displayName: "Regional exposure — London + South East",
    currentValue: 47.9,
    priorValue: 44.2,
    absoluteChange: 3.7,
    percentagePointChange: 3.7,
    relativeChange: 8.4,
    threshold: 50,
    utilization: 95.8,
    headroom: 2.1,
    status: "warning",
    priorStatus: "pass",
    statusTransition: "pass -> warning",
    deteriorated: true,
    numeratorValue: 86_220_000,
    denominatorValue: 180_000_000,
    loansInNumerator: 540,
    parameters: { regions: ["London", "South East"] },
    provenance: {
      sourceReference: "facility_schedule_2026.txt",
      sourceText:
        "1.3 The aggregate Current Balance of Loans secured on Properties located in London plus the South East must not exceed 50% of the Portfolio.",
      locator: "clause 1.3",
      proposalId: "ctp_lse",
      ...APPROVAL,
    },
  }),
  test({
    testId: "ct_high_value",
    metricId: "property_value_above_share",
    category: "property_value",
    displayName: "High-value property exposure — above 1,500,000",
    currentValue: 11.2,
    priorValue: 9.8,
    absoluteChange: 1.4,
    percentagePointChange: 1.4,
    threshold: 10,
    utilization: 112,
    headroom: -1.2,
    status: "breach",
    priorStatus: "warning",
    statusTransition: "warning -> breach",
    deteriorated: true,
    breachAmount: 1.2,
    numeratorValue: 20_160_000,
    denominatorValue: 180_000_000,
    loansInNumerator: 12,
    severity: "critical",
    parameters: { amount: 1_500_000, value_basis: "original", comparison: "above" },
    provenance: {
      sourceReference: "facility_schedule_2026.txt",
      sourceText:
        "2.1 The aggregate Current Balance of Loans with an original valuation above GBP 1,500,000 must not exceed 10% of the Portfolio.",
      locator: "clause 2.1",
      proposalId: "ctp_hv",
      ...APPROVAL,
    },
  }),
  test({
    testId: "ct_net_wac",
    metricId: "rate_net_wac",
    category: "rate_product",
    displayName: "Net WAC",
    currentValue: 4.12,
    priorValue: 4.05,
    absoluteChange: 0.07,
    percentagePointChange: 0.07,
    threshold: 3.75,
    operator: "min",
    utilization: 91,
    headroom: 0.37,
    status: "pass",
    priorStatus: "pass",
    numeratorValue: 741_600,
    denominatorValue: 180_000_000,
    loansInNumerator: 1250,
    parameters: { deduction_percent: 0.5, deduction_basis: "servicing_fee_only" },
    definition: {
      numerator: "Σ((coupon − deduction) × balance).",
      aggregation: "weighted_average",
      description:
        "Balance-weighted average coupon net of a confirmed deduction.",
      definitionNotes:
        "Confirmed at onboarding: servicing fee only, 0.50pp.",
    },
  }),
  test({
    testId: "ct_hpi",
    metricId: "index_hpi_ratio",
    category: "external_index",
    displayName: "House-price-index ratio",
    threshold: 90,
    operator: "min",
    status: "unavailable",
    dataStatus: "external_reference_unconfigured",
    priorAvailable: false,
    priorReportingDate: null,
    notes:
      "No approved external index source is configured for this test. The ratio is not simulated.",
  }),
];

export function mockConcentrationTests(portfolioId: string): ConcentrationTestsSnapshot {
  return {
    portfolioId,
    toRunId: "mi_2025_11",
    reportingDate: "2025-11-30",
    priorReportingDate: "2025-10-31",
    priorAvailable: true,
    available: true,
    source: "approved_configuration",
    approvalStatus: "approved",
    configurationVersion: 2,
    configurationHash: "47a7b0e4e4236edd",
    activatedBy: "Demo Operator",
    activatedAt: "2026-01-10T09:05:00+00:00",
    libraryVersion: "1.0.0",
    evaluatedAt: "2025-12-01T08:00:00+00:00",
    fundedDataAvailable: true,
    proposalCounts: { approved: 4, pending_confirmation: 1 },
    openProposals: 1,
    unsupportedProposals: 1,
    tests: TESTS,
    summary: {
      overallStatus: "breach",
      activeTests: TESTS.length,
      breaches: 1,
      warnings: 1,
      passes: 2,
      unavailable: 1,
      pendingEffectiveDate: 0,
      expired: 0,
      reportingDate: "2025-11-30",
      priorReportingDate: "2025-10-31",
      priorAvailable: true,
      deteriorations: 2,
      closestToLimit: {
        testId: "ct_london_se",
        displayName: "Regional exposure — London + South East",
        headroom: 2.1,
        unit: "percent",
      },
    },
    lineage: {
      source: "approved_configuration",
      note: "Approved configuration v2, evaluated against the governed funded frames.",
    },
  };
}

export function mockConcentrationDrillthrough(
  _portfolioId: string,
  testId: string,
): ConcentrationDrillthrough {
  const t = TESTS.find((x) => x.testId === testId);
  if (!t || t.status === "unavailable") {
    return {
      available: false,
      reason: "This test has no loan-level contributing population.",
      columns: [],
      rows: [],
    };
  }
  const rows = Array.from({ length: Math.min(t.loansInNumerator, 25) }, (_, i) => ({
    loan_id: `L${1000 + i}`,
    collateral_geography: "East Anglia",
    current_outstanding_balance: 140_000 + i * 1000,
    original_valuation_amount: 320_000 + i * 2500,
  }));
  return {
    available: true,
    testId,
    displayName: t.displayName,
    columns: [
      "loan_id",
      "collateral_geography",
      "current_outstanding_balance",
      "original_valuation_amount",
    ],
    rows,
    rowCount: t.loansInNumerator,
    truncated: t.loansInNumerator > rows.length,
    loansInNumerator: t.loansInNumerator,
    numeratorValue: t.numeratorValue,
    denominatorValue: t.denominatorValue,
    denominatorBasis: t.denominatorBasis,
    reconciles: true,
    reportingDate: t.reportingDate,
    configurationVersion: 2,
  };
}

export function mockConcentrationHistory(
  _portfolioId: string,
  testId?: string,
): ConcentrationHistory {
  const dates = ["2025-08-31", "2025-09-30", "2025-10-31", "2025-11-30"];
  const series = TESTS.filter(
    (t) => (!testId || t.testId === testId) && t.currentValue !== null,
  ).map((t) => ({
    testId: t.testId,
    displayName: t.displayName,
    unit: t.unit,
    threshold: t.threshold,
    operator: t.operator,
    warningFraction: t.warningFraction,
    points: dates.map((d, i) => ({
      runId: `mi_2025_${8 + i}`,
      reportingDate: d,
      value:
        t.currentValue === null
          ? null
          : Math.round((t.currentValue - (dates.length - 1 - i) * (t.absoluteChange ?? 0.5)) * 100) / 100,
      status: i === dates.length - 1 ? t.status : (t.priorStatus ?? t.status),
    })),
  }));
  return {
    available: series.length > 0,
    series,
    configurationVersion: 2,
    periods: dates.map((d, i) => ({ runId: `mi_2025_${8 + i}`, reportingDate: d })),
  };
}
