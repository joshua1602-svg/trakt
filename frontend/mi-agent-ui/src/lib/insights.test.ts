import { describe, expect, it } from "vitest";
import type { ChartArtifact, TableArtifact } from "@/domain";
import { buildInvestigations, computeInsights } from "./insights";

function chart(
  rows: Array<Record<string, string | number>>,
  opts: { withLtv?: boolean } = {},
): ChartArtifact {
  const series = [{ key: "balance", label: "Balance", color: "#000" }];
  const displayHints: ChartArtifact["displayHints"] = { balance: { format: "gbp", scale: null } };
  if (opts.withLtv) {
    series.push({ key: "wa_ltv", label: "WA LTV", color: "#111" });
    displayHints!.wa_ltv = { format: "pct", scale: "percent_fraction" };
  }
  return {
    id: "art",
    type: "chart",
    title: "Balance by Region",
    source: { engine: "mi_agent.workflow", label: "MI Agent · bar", spec: { metric: "balance", dimension: "region" } },
    createdAt: "2026-06-26T08:00:00Z",
    mock: false,
    chartType: "bar",
    xKey: "region",
    series,
    valueFormat: "gbp",
    displayHints,
    rows,
  };
}

describe("computeInsights — concentration", () => {
  it("flags the largest bucket and its share, with severity by threshold", () => {
    const s = computeInsights(
      chart([
        { region: "London", balance: 700 },
        { region: "South East", balance: 200 },
        { region: "South West", balance: 100 },
      ]),
    )!;
    const conc = s.observations.find((o) => o.kind === "concentration")!;
    expect(conc.text).toContain("London");
    expect(conc.text).toContain("70%");
    expect(conc.severity).toBe("significant"); // > 60%
    expect(s.statistics.topShare).toBeCloseTo(0.7);
    expect(s.statistics.total).toBe(1000);
  });

  it("uses WATCH between 40% and 60%", () => {
    const s = computeInsights(
      chart([
        { region: "London", balance: 50 },
        { region: "South East", balance: 30 },
        { region: "South West", balance: 20 },
      ]),
    )!;
    expect(s.observations.find((o) => o.kind === "concentration")!.severity).toBe("watch");
  });
});

describe("computeInsights — ranking / spread", () => {
  it("reports the top-3 share for 4+ buckets and the value range", () => {
    const s = computeInsights(
      chart([
        { region: "London", balance: 400 },
        { region: "South East", balance: 300 },
        { region: "South West", balance: 200 },
        { region: "Wales", balance: 100 },
      ]),
    )!;
    expect(s.observations.find((o) => o.kind === "ranking")!.text).toContain("90%");
    const spread = s.observations.find((o) => o.kind === "spread")!;
    expect(spread.text).toContain("Wales");
    expect(spread.text).toContain("London");
    expect(s.statistics.spread).toBe(300);
  });
});

describe("computeInsights — outlier", () => {
  it("detects a secondary-measure outlier and states the pp gap", () => {
    const s = computeInsights(
      chart(
        [
          { region: "London", balance: 400, wa_ltv: 0.3 },
          { region: "South East", balance: 300, wa_ltv: 0.31 },
          { region: "South West", balance: 300, wa_ltv: 0.62 }, // far above
        ],
        { withLtv: true },
      ),
    )!;
    const outlier = s.observations.find((o) => o.kind === "outlier");
    expect(outlier).toBeDefined();
    expect(outlier!.text).toContain("South West");
    expect(outlier!.text).toMatch(/pp above/);
  });
});

describe("computeInsights — movement", () => {
  it("reports signed movement only when a prior total is supplied", () => {
    const rows = [
      { region: "London", balance: 600 },
      { region: "South East", balance: 540 },
    ];
    expect(computeInsights(chart(rows))!.observations.find((o) => o.kind === "movement")).toBeUndefined();
    const s = computeInsights(chart(rows), undefined, { previousTotal: 1000, previousLabel: "prior week" })!;
    const mv = s.observations.find((o) => o.kind === "movement")!;
    expect(mv.text).toContain("+14.0%");
    expect(mv.text).toContain("prior week");
  });
});

describe("computeInsights — edge cases", () => {
  it("returns null for an empty artifact", () => {
    expect(computeInsights(chart([]))).toBeNull();
  });
  it("returns null for a single row", () => {
    expect(computeInsights(chart([{ region: "London", balance: 100 }]))).toBeNull();
  });
  it("handles tied values without a spread observation or crash", () => {
    const s = computeInsights(
      chart([
        { region: "A", balance: 100 },
        { region: "B", balance: 100 },
        { region: "C", balance: 100 },
        { region: "D", balance: 100 },
      ]),
    )!;
    expect(s.statistics.spread).toBe(0);
    expect(s.observations.find((o) => o.kind === "spread")).toBeUndefined();
    expect(s.observations.length).toBeGreaterThan(0);
  });
  it("returns null for a table with no numeric columns", () => {
    const t: TableArtifact = {
      id: "t",
      type: "table",
      title: "Regions",
      source: { engine: "mi_agent.workflow", label: "MI Agent · table" },
      createdAt: "2026-06-26T08:00:00Z",
      mock: false,
      columns: [{ key: "region", label: "Region", format: "text" }],
      rows: [{ region: "London" }, { region: "South East" }],
    };
    expect(computeInsights(t)).toBeNull();
  });
  it("caps observations at 5", () => {
    const s = computeInsights(
      chart(
        [
          { region: "London", balance: 700, wa_ltv: 0.6 },
          { region: "South East", balance: 200, wa_ltv: 0.31 },
          { region: "South West", balance: 60, wa_ltv: 0.3 },
          { region: "Wales", balance: 40, wa_ltv: 0.29 },
        ],
        { withLtv: true },
      ),
      undefined,
      { previousTotal: 800 },
    )!;
    expect(s.observations.length).toBeLessThanOrEqual(5);
  });
});

describe("buildInvestigations", () => {
  it("offers an executable drill into the largest bucket above the threshold", () => {
    const sugg = buildInvestigations({
      count: 3,
      total: 1000,
      max: 700,
      min: 100,
      mean: 333,
      median: 200,
      stdev: 250,
      spread: 600,
      topLabel: "London",
      topShare: 0.7,
      bottomLabel: "South West",
    });
    expect(sugg[0]).toEqual({ label: "Investigate London", question: "only London", kind: "drill" });
  });
  it("returns nothing when concentration is low and variance is small", () => {
    expect(
      buildInvestigations({ count: 3, total: 300, max: 105, min: 95, mean: 100, median: 100, stdev: 8, spread: 10, topLabel: "A", topShare: 0.37, bottomLabel: "C" }),
    ).toEqual([]);
  });
});

describe("performance", () => {
  it("analyses a 200-bucket artifact quickly", () => {
    const rows = Array.from({ length: 200 }, (_, i) => ({ region: `R${i}`, balance: 100 + i }));
    const t0 = performance.now();
    const s = computeInsights(chart(rows));
    const elapsed = performance.now() - t0;
    expect(s).not.toBeNull();
    expect(elapsed).toBeLessThan(50); // target <5ms; generous bound for CI noise
  });
});
