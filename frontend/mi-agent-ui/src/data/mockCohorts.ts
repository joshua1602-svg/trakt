import type { CohortAnalysis } from "@/domain";

/** Deterministic mock origination-vintage cohort analysis (synthetic, not client
 * data). Mirrors the backend view-model: point-in-time vintage aggregates only —
 * no fabricated redemption/completion curves. */
export function mockCohorts(portfolioId: string): CohortAnalysis {
  const client = portfolioId.split("/")[0] || "client_001";
  const cohorts = [
    { vintage: "2021", loanCount: 42, balance: 6_800_000, sharePct: 21.9, waLtv: 0.66, waRate: 0.041, waMonthsOnBook: 44 },
    { vintage: "2022", loanCount: 55, balance: 9_200_000, sharePct: 29.6, waLtv: 0.71, waRate: 0.048, waMonthsOnBook: 31 },
    { vintage: "2023", loanCount: 61, balance: 10_400_000, sharePct: 33.4, waLtv: 0.68, waRate: 0.053, waMonthsOnBook: 19 },
    { vintage: "2024", loanCount: 29, balance: 4_700_000, sharePct: 15.1, waLtv: 0.63, waRate: 0.055, waMonthsOnBook: 8 },
  ];
  const totalBalance = cohorts.reduce((a, c) => a + c.balance, 0);
  const totalLoanCount = cohorts.reduce((a, c) => a + c.loanCount, 0);
  return {
    dataset: "cohorts",
    portfolioId: `${client}/mi_2025_11`,
    cohortBasis: "origination_date",
    period: "Y",
    reportingDate: "2025-11-30",
    available: true,
    totalBalance,
    totalLoanCount,
    metricsAvailable: ["balance", "loanCount", "waLtv", "waRate", "waMonthsOnBook"],
    cohorts,
    lineage: {
      source: "governed funded central lender tape (origination vintage)",
      note: "Point-in-time vintage aggregates only. Redemption / completion / "
        + "performance curves are not computed in the MI path and are not shown.",
    },
  };
}

/** Deterministic mock cohort PROGRESSION (synthetic). Three reporting periods
 * of static-pool seasoning for the requested cohort — balance grows, NNEG
 * headroom shrinks — so the Cohort tab renders with the demo client. */
export function mockCohortProgression(
  portfolioId: string,
  query?: { lens?: string; vintage?: string; grain?: string },
): import("@/domain").CohortProgression {
  const client = portfolioId.split("/")[0] || "client_001";
  const lens = query?.lens && query.lens !== "total" ? query.lens : "Total";
  const scale = lens === "Total" ? 1 : 0.4;
  const vintageScale = query?.vintage ? 0.5 : 1;
  const s = scale * vintageScale;
  const mk = (period: string, reporting_date: string, growth: number, hr: number) => ({
    period,
    reporting_date,
    loanCount: Math.round(60 * s),
    metrics: {
      funded_balance: Math.round(9_000_000 * s * growth),
      loan_count: Math.round(60 * s),
      wa_ltv: 0.39,
      wa_interest_rate: 0.0955,
      avg_borrower_age: 72,
      nneg_exposure: 0,
      nneg_headroom: Math.round(6_000_000 * s * hr),
      nneg_headroom_pct: hr,
    } as Record<string, number | null>,
  });
  return {
    dataset: "cohort_progression",
    portfolioId: `${client}/mi_2025_11`,
    available: true,
    lens,
    vintage: query?.vintage ?? null,
    grain: (query?.grain as "Y" | "Q" | "M") ?? "Y",
    metricsAvailable: ["funded_balance", "loan_count", "wa_ltv", "wa_interest_rate",
      "avg_borrower_age", "nneg_exposure", "nneg_headroom", "nneg_headroom_pct"],
    periods: [
      mk("2025-09", "2025-09-30", 0.92, 0.62),
      mk("2025-10", "2025-10-31", 0.97, 0.58),
      mk("2025-11", "2025-11-30", 1.0, 0.55),
    ],
    singlePeriod: false,
    lineage: { note: "Static-pool seasoning across reporting periods (synthetic)." },
  };
}
