import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import type { ChartArtifact } from "@/domain";
import { buildDrillModel } from "@/lib/drill";
import { DrillThroughPanel } from "./DrillThroughPanel";

function regionChart(): ChartArtifact {
  return {
    id: "art_region",
    type: "chart",
    title: "Balance by Region",
    source: { engine: "mi_agent.workflow", label: "MI Agent · bar" },
    createdAt: "2026-06-26T08:00:00Z",
    mock: false,
    chartType: "bar",
    xKey: "region",
    series: [
      { key: "balance", label: "Balance", color: "#919dd1" },
      { key: "wa_ltv", label: "WA LTV", color: "#36c2a8" },
    ],
    rows: [
      { region: "London", balance: 400, wa_ltv: 0.33 },
      { region: "South East", balance: 100, wa_ltv: 0.29 },
    ],
    valueFormat: "gbp",
    displayHints: {
      balance: { format: "gbp", scale: null },
      wa_ltv: { format: "pct", scale: "percent_fraction" },
    },
  };
}

describe("buildDrillModel", () => {
  it("derives the dimension, values and measures from a grouped chart", () => {
    const model = buildDrillModel(regionChart());
    expect(model).not.toBeNull();
    expect(model!.dimensionLabel).toBe("Region");
    expect(model!.values).toEqual(["London", "South East"]);
    expect(model!.measures.map((m) => m.key)).toEqual(["balance", "wa_ltv"]);
    expect(model!.totals.balance).toBe(500);
    expect(model!.primary?.key).toBe("balance");
  });

  it("returns null for a continuous scatter axis", () => {
    const scatter = { ...regionChart(), chartType: "scatter" as const };
    expect(buildDrillModel(scatter)).toBeNull();
  });
});

describe("DrillThroughPanel", () => {
  it("shows detailed metrics and a share of total after selecting a value", () => {
    render(<DrillThroughPanel artifact={regionChart()} />);
    fireEvent.change(screen.getByLabelText(/Drill into Region/i), { target: { value: "London" } });
    // Balance for London and its share of the £500 total.
    expect(screen.getByText("£400")).toBeInTheDocument();
    expect(screen.getByText("80.0%")).toBeInTheDocument();
    // Fraction-scaled WA LTV renders as points, not 0.3%.
    expect(screen.getByText("33.0%")).toBeInTheDocument();
  });
});
