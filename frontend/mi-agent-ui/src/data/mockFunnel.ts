import type { PipelineFunnelEvolution, FunnelPoint, FunnelStageSummary } from "@/domain";

/** Deterministic mock weekly origination funnel (synthetic, not client data). */
const STAGES = ["KFI", "APPLICATION", "OFFER", "COMPLETED"] as const;
const LABELS: Record<string, string> = {
  KFI: "KFIs", APPLICATION: "Applications", OFFER: "Offers", COMPLETED: "Completions",
};
const WEEKS = ["2025-10-13", "2025-10-20", "2025-10-27", "2025-11-03", "2025-11-10"];

// Per-stage weekly (value £, count) — a tapering funnel that grows week-on-week.
const SHAPE: Record<string, { v: number; c: number }[]> = {
  KFI: [
    { v: 2_400_000, c: 18 }, { v: 2_650_000, c: 20 }, { v: 2_800_000, c: 21 },
    { v: 3_050_000, c: 23 }, { v: 3_200_000, c: 24 },
  ],
  APPLICATION: [
    { v: 1_500_000, c: 11 }, { v: 1_640_000, c: 12 }, { v: 1_720_000, c: 13 },
    { v: 1_880_000, c: 14 }, { v: 2_000_000, c: 15 },
  ],
  OFFER: [
    { v: 900_000, c: 7 }, { v: 980_000, c: 7 }, { v: 1_050_000, c: 8 },
    { v: 1_120_000, c: 8 }, { v: 1_220_000, c: 9 },
  ],
  COMPLETED: [
    { v: 600_000, c: 4 }, { v: 640_000, c: 4 }, { v: 700_000, c: 5 },
    { v: 760_000, c: 5 }, { v: 820_000, c: 6 },
  ],
};

function avg(nums: number[]): number {
  return Math.round((nums.reduce((a, b) => a + b, 0) / nums.length) * 100) / 100;
}

export function mockFunnelEvolution(portfolioId: string): PipelineFunnelEvolution {
  const client = portfolioId.split("/")[0] || "client_001";
  const series: Record<string, FunnelPoint[]> = {};
  const summary: Record<string, FunnelStageSummary> = {};
  for (const stage of STAGES) {
    const pts = SHAPE[stage].map((p, i) => ({ week: WEEKS[i], value: p.v, count: p.c }));
    series[stage] = pts;
    const values = pts.map((p) => p.value as number);
    const counts = pts.map((p) => p.count);
    const latest = pts[pts.length - 1];
    const prior = pts[pts.length - 2];
    summary[stage] = {
      label: LABELS[stage],
      latestValue: latest.value,
      latestCount: latest.count,
      fiveWeekAvgValue: avg(values),
      fiveWeekAvgCount: avg(counts),
      deltaValue: (latest.value as number) - (prior.value as number),
      deltaCount: latest.count - prior.count,
      trend: "up",
      weeksObserved: pts.length,
    };
  }
  return {
    dataset: "pipeline_funnel",
    portfolioId: client,
    toRunId: "mi_2025_11",
    stages: [...STAGES],
    stageLabels: LABELS,
    weeks: WEEKS,
    sourceFiles: WEEKS.map((w) => `M2L_KFI_and_Pipeline_${w.replace(/-/g, "_")}.csv`),
    uniqueWeeklyExtractsUsed: WEEKS.length,
    series,
    summary,
    lineage: {
      source: "governed weekly pipeline extracts (deduplicated)",
      metric: "weekly KFI / Application / Offer / Completion value and count",
      fiveWeekAverage: "trailing mean of up to the last 5 weekly extracts",
    },
    singlePeriod: false,
  };
}
