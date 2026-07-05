import type {
  PipelineFunnelEvolution,
  FunnelPoint,
  FunnelFlowPoint,
  FunnelStageSummary,
} from "@/domain";

/** Deterministic mock weekly origination funnel (synthetic, not client data).
 * Mirrors the backend flow-first shape: per-week STOCK level (series), the
 * derived WEEKLY FLOW (flowSeries), a flow-based summary and conversion vs KFI. */
const STAGES = ["KFI", "APPLICATION", "OFFER", "COMPLETED"] as const;
const LABELS: Record<string, string> = {
  KFI: "KFIs", APPLICATION: "Applications", OFFER: "Offers", COMPLETED: "Completions",
};
const WEEKS = ["2025-10-13", "2025-10-20", "2025-10-27", "2025-11-03", "2025-11-10"];

// Per-stage weekly STOCK level (value £, count) — a tapering funnel whose stock
// grows week-on-week; the weekly flow is the (smaller) week-on-week change.
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

function round2(n: number): number {
  return Math.round(n * 100) / 100;
}

/** flow[t] = level[t] − level[t-1]; first week has no prior extract → null. */
function weeklyFlow(levels: number[]): (number | null)[] {
  return levels.map((v, i) => (i === 0 ? null : round2(v - levels[i - 1])));
}

function trailingAvg(vals: (number | null)[], window = 5): number | null {
  const use = vals.filter((v): v is number => v != null).slice(-window);
  return use.length ? round2(use.reduce((a, b) => a + b, 0) / use.length) : null;
}

function conv(num: number | null, den: number | null): number | null {
  return num != null && den ? round2((num / den) * 100) : null;
}

export function mockFunnelEvolution(portfolioId: string): PipelineFunnelEvolution {
  const client = portfolioId.split("/")[0] || "client_001";
  const series: Record<string, FunnelPoint[]> = {};
  const flowSeries: Record<string, FunnelFlowPoint[]> = {};
  const summary: Record<string, FunnelStageSummary> = {};

  const kfiValues = SHAPE.KFI.map((p) => p.v);
  const kfiCounts = SHAPE.KFI.map((p) => p.c);
  // Mock KFI→completion lag: shift the KFI denominator back ~6 weeks.
  const MOCK_LAG_WEEKS = 6;
  const denomIdx = Math.max(0, kfiCounts.length - 1 - MOCK_LAG_WEEKS);
  const kfiDenomCount = kfiCounts[denomIdx];
  const kfiDenomValue = kfiValues[denomIdx];
  const denomWeek = WEEKS[denomIdx] ?? null;

  for (const stage of STAGES) {
    const pts = SHAPE[stage].map((p, i) => ({ week: WEEKS[i], value: p.v, count: p.c }));
    series[stage] = pts;
    const values = pts.map((p) => p.value);
    const counts = pts.map((p) => p.count);
    const vFlow = weeklyFlow(values);
    const cFlow = weeklyFlow(counts);
    flowSeries[stage] = pts.map((p, i) => ({
      week: p.week, flowValue: vFlow[i], flowCount: cFlow[i] == null ? null : Math.round(cFlow[i]!),
    }));

    const latestFlowV = vFlow[vFlow.length - 1];
    const latestFlowC = cFlow[cFlow.length - 1];
    const priorFlowV = vFlow[vFlow.length - 2] ?? null;
    const priorFlowC = cFlow[cFlow.length - 2] ?? null;

    summary[stage] = {
      label: LABELS[stage],
      latestFlowValue: latestFlowV,
      latestFlowCount: latestFlowC == null ? null : Math.round(latestFlowC),
      priorFlowValue: priorFlowV,
      priorFlowCount: priorFlowC == null ? null : Math.round(priorFlowC),
      fiveWeekAvgFlowValue: trailingAvg(vFlow),
      fiveWeekAvgFlowCount: trailingAvg(cFlow),
      deltaFlowValue: latestFlowV != null && priorFlowV != null ? round2(latestFlowV - priorFlowV) : null,
      deltaFlowCount:
        latestFlowC != null && priorFlowC != null ? Math.round(latestFlowC - priorFlowC) : null,
      latestStockValue: values[values.length - 1],
      latestStockCount: counts[counts.length - 1],
      fiveWeekAvgStockValue: trailingAvg(values),
      fiveWeekAvgStockCount: trailingAvg(counts),
      trend: "up",
      weeksObserved: pts.length,
      conversion:
        stage === "KFI"
          ? null
          : {
              basis: "avg_weekly_flow_over_lagged_kfi_stock",
              lagWeeks: MOCK_LAG_WEEKS,
              lagApplied: true,
              denominatorWeek: denomWeek,
              avgWeeklyFlowCount: trailingAvg(cFlow),
              avgWeeklyFlowValue: trailingAvg(vFlow),
              kfiStockCount: kfiDenomCount,
              kfiStockValue: kfiDenomValue,
              weeklyRateCount: conv(trailingAvg(cFlow), kfiDenomCount),
              weeklyRateValue: conv(trailingAvg(vFlow), kfiDenomValue),
            },
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
    flowSeries,
    summary,
    conversionLagWeeks: MOCK_LAG_WEEKS,
    lineage: {
      source: "governed weekly pipeline extracts (deduplicated)",
      metric: "weekly KFI / Application / Offer / Completion — weekly flow (default) and stock level",
      fiveWeekAverage: "trailing mean of the last 5 weeks of WEEKLY FLOW, not the average stock level",
      conversion: "average weekly flow into a stage over the KFI stock lagWeeks earlier (KFI→completion lag)",
    },
    singlePeriod: false,
  };
}
