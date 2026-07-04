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
