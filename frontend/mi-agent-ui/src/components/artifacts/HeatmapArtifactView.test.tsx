import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import type { ChartArtifact } from "@/domain";
import { ArtifactRenderer } from "./ArtifactRenderer";
import { HeatmapArtifactView } from "./HeatmapArtifactView";

/** A `balance by ltv by region` matrix artifact (rows=region, cols=ltv bucket). */
function heatmapArtifact(overrides: Partial<ChartArtifact> = {}): ChartArtifact {
  return {
    id: "hm_1",
    type: "chart",
    title: "Balance by LTV bucket and region",
    source: { engine: "mi_agent.workflow", label: "MI Agent · heatmap", nativeChartType: "heatmap" },
    createdAt: "2025-11-30T00:00:00Z",
    mock: false,
    chartType: "heatmap",
    xKey: "ltv_bucket",
    yKey: "geographic_region_obligor",
    valueKey: "current_outstanding_balance_sum",
    valueFormat: "gbp",
    series: [],
    rows: [
      { geographic_region_obligor: "South West", ltv_bucket: "30-40%", current_outstanding_balance_sum: 1_234_567 },
      { geographic_region_obligor: "South West", ltv_bucket: "40-50%", current_outstanding_balance_sum: 2_000_000 },
      { geographic_region_obligor: "London", ltv_bucket: "30-40%", current_outstanding_balance_sum: 500_000 },
    ],
    ...overrides,
  } as ChartArtifact;
}

describe("HeatmapArtifactView", () => {
  it("renders a matrix with region rows and LTV-bucket columns", () => {
    render(<HeatmapArtifactView artifact={heatmapArtifact()} />);
    // Column headers are the LTV buckets; row labels are the regions.
    expect(screen.getByText("30-40%")).toBeInTheDocument();
    expect(screen.getByText("40-50%")).toBeInTheDocument();
    expect(screen.getByText("South West")).toBeInTheDocument();
    expect(screen.getByText("London")).toBeInTheDocument();
  });

  it("formats cell values using the currency display hint", () => {
    render(<HeatmapArtifactView artifact={heatmapArtifact()} />);
    // £1,234,567 -> compact £1.2MM via formatValue("gbp").
    expect(screen.getAllByText("£1.2MM").length).toBeGreaterThanOrEqual(1);
    // £2.0MM appears in a cell (and the legend max).
    expect(screen.getAllByText("£2.0MM").length).toBeGreaterThanOrEqual(1);
  });

  it("shows an incomplete-data fallback when a dimension/measure is missing", () => {
    render(<HeatmapArtifactView artifact={heatmapArtifact({ valueKey: undefined })} />);
    expect(screen.getByText(/Heatmap data is incomplete/i)).toBeInTheDocument();
  });
});

describe("ArtifactRenderer heatmap routing", () => {
  it("routes a two-dimension heatmap to the native grid, NOT a bubble/scatter", () => {
    const { container } = render(<ArtifactRenderer artifact={heatmapArtifact()} />);
    // Native CSS-grid heatmap renders an HTML table; never a recharts scatter.
    expect(container.querySelector("table")).toBeTruthy();
    expect(container.querySelector(".recharts-scatter")).toBeFalsy();
    expect(container.querySelector(".recharts-responsive-container")).toBeFalsy();
  });

  it("still renders a bar chart natively (regression)", () => {
    const bar = heatmapArtifact({
      chartType: "bar",
      xKey: "ltv_bucket",
      series: [{ key: "current_outstanding_balance_sum", label: "Balance", color: "#919dd1" }],
    });
    const { container } = render(<ArtifactRenderer artifact={bar} />);
    expect(container.querySelector(".recharts-responsive-container")).toBeTruthy();
  });
});
