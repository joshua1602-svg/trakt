import type { RiskLimitsSnapshot, RiskLimitTest } from "@/domain";

/** Deterministic mock risk-limit monitor (synthetic, not client data). */
function t(over: Partial<RiskLimitTest>): RiskLimitTest {
  return {
    limitId: "x", category: "geographic_concentration", label: "Region", region: null,
    dimensionKey: null, actualValue: null, actualBasis: "funded exposure %",
    limitValue: null, unit: "percent", direction: "max", headroom: null,
    status: "green", movementVsPrior: null, source: "Schedule 8 extracted",
    confidence: "high", notes: "", missingFields: [], sourceSnippet: null,
    sourceSection: null, ...over,
  };
}

export function mockRiskLimits(portfolioId: string): RiskLimitsSnapshot {
  const client = portfolioId.split("/")[0] || "client_001";
  const geo: RiskLimitTest[] = [
    t({ limitId: "geo_london", label: "London", region: "London", actualValue: 27.4,
      limitValue: 30, headroom: 2.6, status: "amber", movementVsPrior: 1.8,
      sourceSnippet: "London exposure must not exceed 30% of the Portfolio.",
      sourceSection: "Geographic Concentration" }),
    t({ limitId: "geo_south_east", label: "South East", region: "South East",
      actualValue: 19.1, limitValue: 30, headroom: 10.9, status: "green",
      movementVsPrior: -0.4 }),
    t({ limitId: "geo_scotland", label: "Scotland", region: "Scotland",
      actualValue: 6.2, limitValue: 10, headroom: 3.8, status: "green",
      movementVsPrior: 0.2 }),
  ];
  const broker: RiskLimitTest[] = [
    t({ limitId: "broker_single", category: "broker_concentration",
      label: "Largest single broker (Alpha)", dimensionKey: "broker_channel",
      actualBasis: "largest broker funded exposure %", actualValue: 18.3,
      limitValue: 20, headroom: 1.7, status: "amber", movementVsPrior: 0.9 }),
    t({ limitId: "broker_top3", category: "broker_concentration", label: "Top 3 brokers",
      actualBasis: "top-3 funded exposure %", actualValue: 41.0, limitValue: 45,
      headroom: 4.0, status: "green", movementVsPrior: 0.5 }),
  ];
  const other: RiskLimitTest[] = [
    t({ limitId: "large_single_loan", category: "large_loan_concentration",
      label: "Largest single loan", actualBasis: "single loan funded exposure %",
      actualValue: 1.4, limitValue: 2, headroom: 0.6, status: "green" }),
    t({ limitId: "wa_ltv", category: "ltv_limit", label: "WA current LTV",
      dimensionKey: "current_loan_to_value", actualBasis: "WA current LTV %",
      actualValue: 33.2, limitValue: 35, headroom: 1.8, status: "amber",
      movementVsPrior: 0.7 }),
    t({ limitId: "variable_rate", category: "interest_rate_limit",
      label: "Variable-rate balance", actualBasis: "variable-rate funded exposure %",
      actualValue: null, limitValue: 90, headroom: null, status: "unavailable",
      missingFields: ["variable_rate_flag"],
      notes: "No variable/fixed rate-type flag in the funded tape." }),
    t({ limitId: "joint_borrower", category: "joint_borrower_limit",
      label: "Joint borrowers", actualValue: null, limitValue: null, unit: "unknown",
      headroom: null, status: "needs_review", confidence: "low",
      notes: "Limit hedged ('subject to policy') — manual review required.",
      sourceSnippet: "Joint Borrowers should be monitored ... subject to the prevailing eligibility policy." }),
  ];
  const tests = [...geo, ...broker, ...other];
  const testsByCategory: Record<string, RiskLimitTest[]> = {
    geographic_concentration: geo,
    broker_concentration: broker,
    large_loan_concentration: [other[0]],
    ltv_limit: [other[1]],
    interest_rate_limit: [other[2]],
    joint_borrower_limit: [other[3]],
  };
  return {
    portfolioId: client,
    toRunId: "mi_2025_11",
    reportingDate: "2025-11-30",
    available: true,
    limitsStatus: "needs_review",
    limitsSource: "Schedule 8 extracted",
    fundedDataAvailable: true,
    summary: {
      testsPassed: tests.filter((x) => x.status === "green").length,
      warnings: tests.filter((x) => x.status === "amber").length,
      breaches: tests.filter((x) => x.status === "red").length,
      needsReview: tests.filter((x) => x.status === "needs_review").length,
      unavailable: tests.filter((x) => x.status === "unavailable").length,
      total: tests.length,
      closestHeadroom: { label: "Largest single loan", headroom: 0.6, limitId: "large_single_loan" },
      largestConcentration: { label: "Top 3 brokers", actualValue: 41.0, limitId: "broker_top3" },
    },
    testsByCategory,
    tests,
    observations: [
      "Nearest to limit: Largest single loan (0.60 pp headroom).",
      "Largest exposure: Top 3 brokers at 41.00%.",
      "2 test(s) worsening vs the prior run.",
      "1 test(s) unavailable (missing fields in the funded tape).",
      "1 limit(s) need manual review.",
    ],
    lineage: {
      limitSource: "Schedule 8 extracted",
      sourceDocument: "schedule_8_concentration.txt",
      dataSource: "governed central lender tape (18_central_lender_tape.csv)",
      exposureBasis: "funded",
      reportingDate: "2025-11-30",
      extractionMethod: "deterministic",
      needsReviewCount: 1,
    },
  };
}
