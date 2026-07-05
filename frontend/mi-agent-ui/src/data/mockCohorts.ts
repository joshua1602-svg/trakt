import type { CohortAnalysis, CohortDimension } from "@/domain";

type Row = { cohort: string; loanCount: number; balance: number; waLtv: number; waRate: number; waMonthsOnBook: number };

// Deterministic synthetic static pools per dimension (not client data).
const BY_DIMENSION: Record<CohortDimension, Row[]> = {
  vintage: [
    { cohort: "2021", loanCount: 42, balance: 6_800_000, waLtv: 0.66, waRate: 0.041, waMonthsOnBook: 44 },
    { cohort: "2022", loanCount: 55, balance: 9_200_000, waLtv: 0.71, waRate: 0.048, waMonthsOnBook: 31 },
    { cohort: "2023", loanCount: 61, balance: 10_400_000, waLtv: 0.68, waRate: 0.053, waMonthsOnBook: 19 },
    { cohort: "2024", loanCount: 29, balance: 4_700_000, waLtv: 0.63, waRate: 0.055, waMonthsOnBook: 8 },
  ],
  age: [
    { cohort: "60–64", loanCount: 22, balance: 3_100_000, waLtv: 0.72, waRate: 0.052, waMonthsOnBook: 18 },
    { cohort: "65–69", loanCount: 58, balance: 9_400_000, waLtv: 0.69, waRate: 0.050, waMonthsOnBook: 24 },
    { cohort: "70–74", loanCount: 61, balance: 10_800_000, waLtv: 0.66, waRate: 0.049, waMonthsOnBook: 29 },
    { cohort: "75–79", loanCount: 34, balance: 5_200_000, waLtv: 0.61, waRate: 0.047, waMonthsOnBook: 33 },
    { cohort: "80–84", loanCount: 12, balance: 1_600_000, waLtv: 0.55, waRate: 0.045, waMonthsOnBook: 38 },
  ],
  ltv: [
    { cohort: "20–30%", loanCount: 18, balance: 2_100_000, waLtv: 0.27, waRate: 0.046, waMonthsOnBook: 34 },
    { cohort: "30–40%", loanCount: 44, balance: 6_800_000, waLtv: 0.36, waRate: 0.048, waMonthsOnBook: 30 },
    { cohort: "40–50%", loanCount: 62, balance: 11_200_000, waLtv: 0.46, waRate: 0.050, waMonthsOnBook: 26 },
    { cohort: "50–60%", loanCount: 37, balance: 6_400_000, waLtv: 0.55, waRate: 0.052, waMonthsOnBook: 22 },
    { cohort: "60–70%", loanCount: 14, balance: 2_600_000, waLtv: 0.65, waRate: 0.054, waMonthsOnBook: 17 },
  ],
  channel: [
    { cohort: "Broker A", loanCount: 71, balance: 12_100_000, waLtv: 0.67, waRate: 0.050, waMonthsOnBook: 27 },
    { cohort: "Broker B", loanCount: 48, balance: 8_300_000, waLtv: 0.70, waRate: 0.051, waMonthsOnBook: 24 },
    { cohort: "Direct", loanCount: 39, balance: 6_900_000, waLtv: 0.64, waRate: 0.047, waMonthsOnBook: 30 },
    { cohort: "Broker C", loanCount: 27, balance: 3_800_000, waLtv: 0.69, waRate: 0.053, waMonthsOnBook: 20 },
  ],
};

const DIMENSION_LABELS: Record<CohortDimension, string> = {
  vintage: "Vintage", age: "Borrower age", ltv: "LTV band", channel: "Origination channel",
};

/** Deterministic mock static-pool cohort analysis (synthetic, not client data),
 * grouped by the requested dimension. Point-in-time aggregates only — no
 * fabricated redemption/completion curves. */
export function mockCohorts(portfolioId: string, dimension: CohortDimension = "vintage"): CohortAnalysis {
  const client = portfolioId.split("/")[0] || "client_001";
  const rows = BY_DIMENSION[dimension] ?? BY_DIMENSION.vintage;
  const totalBalance = rows.reduce((a, c) => a + c.balance, 0);
  const totalLoanCount = rows.reduce((a, c) => a + c.loanCount, 0);
  const cohorts = rows.map((r) => ({
    ...r, vintage: r.cohort,
    sharePct: Math.round((r.balance / totalBalance) * 1000) / 10,
  }));
  return {
    dataset: "cohorts",
    portfolioId: `${client}/mi_2025_11`,
    cohortBasis: "origination_date",
    period: "Y",
    reportingDate: "2025-11-30",
    available: true,
    dimension,
    dimensionLabel: DIMENSION_LABELS[dimension],
    availableDimensions: ["vintage", "age", "ltv", "channel"],
    totalBalance,
    totalLoanCount,
    metricsAvailable: ["balance", "loanCount", "waLtv", "waRate", "waMonthsOnBook"],
    cohorts,
    lineage: {
      source: `governed funded central lender tape (by ${DIMENSION_LABELS[dimension].toLowerCase()})`,
      note: "Point-in-time static-pool aggregates only. Redemption / completion / "
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
