import type { FundedEvolution, PipelineEvolution, ForecastEvolution } from "@/domain";

/** Deterministic mock evolution series (3 funded months, weekly pipeline). */
function recon(records: number, balance: number) {
  return {
    total_records: records,
    total_balance: balance,
    records_included: records,
    balance_included: balance,
    records_excluded_missing: 0,
    balance_excluded_missing: 0,
    coverage_by_balance_pct: 100.0,
    missing_measure_fields: [],
  };
}

export function mockFundedEvolution(portfolioId: string): FundedEvolution {
  const client = portfolioId.split("/")[0] || "client_001";
  const months = [
    { period: "2025-09", run_id: "mi_2025_09", date: "2025-09-30", bal: 8_100_000, n: 70, ltv: 0.41, rate: 0.061 },
    { period: "2025-10", run_id: "mi_2025_10", date: "2025-10-31", bal: 8_500_000, n: 72, ltv: 0.42, rate: 0.062 },
    { period: "2025-11", run_id: "mi_2025_11", date: "2025-11-30", bal: 8_900_000, n: 73, ltv: 0.43, rate: 0.063 },
  ];
  return {
    dataset: "funded",
    portfolioId: client,
    toRunId: "mi_2025_11",
    availableRunIds: months.map((m) => m.run_id),
    reportingDates: months.map((m) => m.date),
    sourceFiles: months.map((m) => `${m.run_id}/output/central/18_central_lender_tape.csv`),
    periods: months.map((m) => ({
      period: m.period,
      run_id: m.run_id,
      reporting_date: m.date,
      metrics: {
        funded_balance: m.bal,
        loan_count: m.n,
        wa_ltv: m.ltv,
        wa_interest_rate: m.rate,
        avg_borrower_age: 74,
      },
      reconciliation: recon(m.n, m.bal),
      source_file: `${m.run_id}/output/central/18_central_lender_tape.csv`,
    })),
    breakdowns: {
      broker: months.flatMap((m) => [
        { period: m.period, key: "Alpha", value: m.bal * 0.5 },
        { period: m.period, key: "Beta", value: m.bal * 0.3 },
        { period: m.period, key: "Gamma", value: m.bal * 0.2 },
      ]),
      region: months.flatMap((m) => [
        { period: m.period, key: "North", value: m.bal * 0.6 },
        { period: m.period, key: "South East", value: m.bal * 0.4 },
      ]),
      ltv_bucket: [],
    },
    lineage: { source: "governed monthly central lender tapes" },
    singlePeriod: false,
  };
}

export function mockPipelineEvolution(portfolioId: string): PipelineEvolution {
  const client = portfolioId.split("/")[0] || "client_001";
  const weeks = [
    { period: "2025-10", date: "2025-10-06", amt: 290_000_000, n: 240, w: 110_000_000 },
    { period: "2025-11", date: "2025-11-03", amt: 360_000_000, n: 300, w: 150_000_000 },
    { period: "2025-12", date: "2025-12-01", amt: 434_000_000, n: 360, w: 190_000_000 },
  ];
  return {
    dataset: "pipeline",
    portfolioId: client,
    toRunId: "mi_2025_11",
    availableExtractDates: weeks.map((w) => w.date),
    uniqueWeeklyExtractsUsed: weeks.length,
    periods: weeks.map((w) => ({
      period: w.period,
      extract_date: w.date,
      week: w.date,
      metrics: {
        pipeline_amount: w.amt,
        pipeline_case_count: w.n,
        weighted_expected_funded_amount: w.w,
      },
      reconciliation: recon(w.n, w.amt),
      source_file: `M2L KFI and Pipeline ${w.date}.xlsx`,
    })),
    byStage: weeks.flatMap((w) => [
      { period: w.date, week: w.date, stage: "KFI", value: w.amt * 0.4, count: Math.round(w.n * 0.4) },
      { period: w.date, week: w.date, stage: "APPLICATION", value: w.amt * 0.3, count: Math.round(w.n * 0.28) },
      { period: w.date, week: w.date, stage: "OFFER", value: w.amt * 0.18, count: Math.round(w.n * 0.18) },
      { period: w.date, week: w.date, stage: "COMPLETED", value: w.amt * 0.09, count: Math.round(w.n * 0.1) },
      { period: w.date, week: w.date, stage: "WITHDRAWN", value: w.amt * 0.03, count: Math.round(w.n * 0.04) },
    ]),
    lineage: { source: "governed weekly pipeline extracts (deduplicated)" },
    singlePeriod: false,
  };
}

export function mockForecastEvolution(portfolioId: string): ForecastEvolution {
  const funded = mockFundedEvolution(portfolioId);
  const pipe = mockPipelineEvolution(portfolioId);
  const wByMonth = new Map(
    pipe.periods.map((p) => [p.period, (p.metrics.weighted_expected_funded_amount as number) ?? 0]),
  );
  return {
    dataset: "forecast",
    portfolioId: funded.portfolioId,
    toRunId: funded.toRunId,
    periods: funded.periods.map((p) => {
      const fb = (p.metrics.funded_balance as number) ?? 0;
      const w = wByMonth.get(p.period) ?? 0;
      return {
        period: p.period,
        run_id: p.run_id,
        reporting_date: p.reporting_date,
        metrics: {
          funded_balance: fb,
          weighted_expected_pipeline: w,
          forecast_funded_balance: fb + w,
        },
        reconciliation: p.reconciliation,
        source_file: p.source_file,
      };
    }),
    lineage: { formula: "forecast = funded balance + Σ(weighted expected pipeline)" },
    singlePeriod: false,
  };
}
