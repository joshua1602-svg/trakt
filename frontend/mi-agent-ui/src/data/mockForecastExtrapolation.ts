import type {
  ForecastExtrapolation, ProjectedBalance, MilestoneRow,
} from "@/domain";

/** Deterministic mock scale-up forecast (synthetic, not client data). */
const THRESHOLDS = [25_000_000, 50_000_000, 75_000_000, 100_000_000, 150_000_000];
const CURRENT = 12_500_000;
const BASE = 2_500_000; // monthly run-rate
const DOWN = 1_875_000;
const UP = 3_125_000;
const REPORTING = "2025-11";

function addMonths(ym: string, m: number): string {
  const y = parseInt(ym.slice(0, 4), 10);
  const mo = parseInt(ym.slice(5, 7), 10);
  const idx = y * 12 + (mo - 1) + m;
  return `${String(Math.floor(idx / 12)).padStart(4, "0")}-${String((idx % 12) + 1).padStart(2, "0")}`;
}

function projected(): ProjectedBalance[] {
  const rows: ProjectedBalance[] = [];
  for (let m = 0; m <= 18; m++) {
    rows.push({
      month: addMonths(REPORTING, m), offset: m,
      downside: CURRENT + DOWN * m, base: CURRENT + BASE * m, upside: CURRENT + UP * m,
    });
  }
  return rows;
}

function milestones(): MilestoneRow[] {
  return THRESHOLDS.map((thr) => {
    if (CURRENT >= thr) {
      return { threshold: thr, thresholdLabel: `£${thr / 1_000_000}m`, reached: true,
        downsideDate: "reached", baseDate: "reached", upsideDate: "reached" };
    }
    const months = (rate: number) => Math.ceil((thr - CURRENT) / rate);
    return {
      threshold: thr, thresholdLabel: `£${thr / 1_000_000}m`, reached: false,
      downsideDate: addMonths(REPORTING, months(DOWN)), downsideMonths: months(DOWN),
      baseDate: addMonths(REPORTING, months(BASE)), baseMonths: months(BASE),
      upsideDate: addMonths(REPORTING, months(UP)), upsideMonths: months(UP),
    };
  });
}

export function mockForecastExtrapolation(portfolioId: string): ForecastExtrapolation {
  const client = portfolioId.split("/")[0] || "client_001";
  return {
    portfolioId: client,
    toRunId: "mi_2025_11",
    reportingPeriod: REPORTING,
    currentFundedBalance: CURRENT,
    currentWeightedPipelineForecast: {
      model: "current_weighted_pipeline",
      label: "Current weighted pipeline forecast",
      available: true,
      fundedBalance: CURRENT,
      weightedExpectedPipeline: 1_350_000,
      forecastFundedBalance: CURRENT + 1_350_000,
      note: "Point-in-time bridge (funded balance + weighted expected pipeline). NOT the full scale-up forecast.",
    },
    completionRunRateForecast: {
      model: "completion_run_rate",
      available: true,
      status: "ok",
      observedMonths: 5,
      lookbackAverages: { "4w": 2_450_000, "5w": 2_500_000 },
      baseMonthlyRunRate: BASE,
      annualisedRunRate: BASE * 12,
      scenarioMonthlyRunRate: { downside: DOWN, base: BASE, upside: UP },
      scenarioBasis: "empirical 25th/75th percentile of recent monthly completions",
      projectedBalances: projected(),
      milestones: milestones(),
      assumptions: {
        lookbackWindowsMonths: [4, 5, 8, 12], observedMonths: 5, horizonMonths: 18,
        currentFundedBalance: CURRENT, completionSignal: "month-on-month funded balance growth",
      },
      caveats: [],
      completionHistory: [
        { period: "2025-08", completion_amount: 2_300_000 },
        { period: "2025-09", completion_amount: 2_500_000 },
        { period: "2025-10", completion_amount: 2_650_000 },
        { period: "2025-11", completion_amount: 2_500_000 },
      ],
    },
    kfiConversionForecast: {
      model: "kfi_conversion",
      available: true,
      status: "ok",
      observedWeeks: 5,
      avgWeeklyKfiInflow: 2_820_000,
      conversionRate: 0.28,
      lagMonths: 2,
      expectedMonthlyCompletion: 3_421_200,
      scenarioMonthlyRunRate: { downside: 2_565_900, base: 3_421_200, upside: 4_276_500 },
      projectedBalances: projected(),
      milestones: milestones(),
      assumptions: { conversionRate: 0.28, lagMonths: 2, kfiLookbackWeeks: 5, currentFundedBalance: CURRENT },
      caveats: ["KFI→completion conversion and lag are empirical estimates; scenario bands are indicative."],
    },
    thresholds: THRESHOLDS,
    dataSufficiency: "ok",
    sourcePeriods: ["2025-09", "2025-10", "2025-11"],
    sourceFiles: ["mi_2025_11/output/central/18_central_lender_tape.csv"],
    lineage: {
      source: "governed funded central tapes + weekly pipeline extracts",
      scenarioNote: "Scenario bands are indicative ranges, not statistical confidence intervals.",
    },
  };
}
