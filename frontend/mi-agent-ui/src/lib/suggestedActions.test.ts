import { describe, expect, it } from "vitest";
import type { ChartArtifact, KPIArtifact, MIQuerySpec } from "@/domain";
import { buildSuggestedActions } from "./suggestedActions";

function regionChart(): ChartArtifact {
  return {
    id: "art_region",
    type: "chart",
    title: "Balance by Region",
    source: { engine: "mi_agent.workflow", label: "MI Agent · bar" },
    createdAt: "2026-06-26T08:00:00Z",
    mock: false,
    chartType: "bar",
    xKey: "geographic_region_obligor",
    series: [{ key: "current_outstanding_balance", label: "Balance", color: "#919dd1" }],
    rows: [
      { geographic_region_obligor: "London", current_outstanding_balance: 400 },
      { geographic_region_obligor: "South East", current_outstanding_balance: 100 },
    ],
    valueFormat: "gbp",
  };
}

const spec: Partial<MIQuerySpec> = {
  metric: "current_outstanding_balance",
  dimension: "geographic_region_obligor",
};

describe("buildSuggestedActions", () => {
  it("returns grounded, capped follow-ups for a region balance chart", () => {
    const actions = buildSuggestedActions(spec, regionChart());
    expect(actions.length).toBeGreaterThanOrEqual(3);
    expect(actions.length).toBeLessThanOrEqual(5);
    const labels = actions.map((a) => a.label);
    expect(labels).toContain("Split by Broker");
    // The current dimension is never suggested as a split.
    expect(labels.some((l) => /Split by Region/i.test(l))).toBe(false);
    // A measure change and a drill into the largest value are offered.
    expect(actions.some((a) => a.kind === "change_measure")).toBe(true);
    expect(actions.find((a) => a.kind === "drill")?.question).toBe("only London");
    expect(actions.some((a) => a.kind === "refine" && /as a table/i.test(a.question))).toBe(true);
  });

  it("split questions reference real catalogue dimensions only", () => {
    const actions = buildSuggestedActions(spec, regionChart());
    for (const a of actions.filter((x) => x.kind === "change_dimension")) {
      expect(a.question.toLowerCase()).toMatch(/^balance by /);
    }
  });

  it("returns [] for a non-chart/table artifact", () => {
    const kpi: KPIArtifact = {
      id: "k1",
      type: "kpi",
      title: "KPIs",
      source: { engine: "mi_agent.workflow", label: "MI Agent" },
      createdAt: "2026-06-26T08:00:00Z",
      mock: false,
      kpis: [],
    };
    expect(buildSuggestedActions(spec, kpi)).toEqual([]);
  });

  it("returns [] when the dimension cannot be grounded", () => {
    const art = regionChart();
    expect(buildSuggestedActions({ metric: "current_outstanding_balance", dimension: "mystery_dim" }, art)).toEqual([]);
  });
});
