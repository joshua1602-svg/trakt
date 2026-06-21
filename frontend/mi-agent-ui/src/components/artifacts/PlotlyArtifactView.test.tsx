import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import type { ChartArtifact } from "@/domain";
import { ArtifactRenderer } from "./ArtifactRenderer";
import { PlotlyArtifactView } from "./PlotlyArtifactView";

// Stub the heavy lazy chunk so jsdom never loads real plotly.js.
vi.mock("./PlotlyLazy", () => ({
  default: ({ figure }: { figure: { data: unknown[] } }) => (
    <div data-testid="plotly-stub">traces:{figure.data.length}</div>
  ),
}));

const base = { id: "c1", title: "Chart", createdAt: "2026-05-31T08:00:00Z", mock: false };

function chart(partial: Partial<ChartArtifact> & { figure?: unknown }): ChartArtifact {
  const { figure, ...rest } = partial;
  return {
    ...base,
    type: "chart",
    chartType: "bar",
    xKey: "region",
    series: [{ key: "balance", label: "Balance", color: "#919dd1" }],
    rows: [{ region: "London", balance: 184 }],
    source: { engine: "mi_agent.workflow", label: "MI Agent · bar", ...(figure ? { figure } : {}) },
    ...rest,
  } as ChartArtifact;
}

const FIGURE = { data: [{ type: "heatmap" }], layout: {} };

describe("PlotlyArtifactView (fallback, themed)", () => {
  it("renders a valid figure through the lazy Plotly chunk", async () => {
    render(<PlotlyArtifactView artifact={chart({ figure: FIGURE })} />);
    expect(await screen.findByTestId("plotly-stub")).toHaveTextContent("traces:1");
  });

  it("shows a graceful error when the figure is missing", () => {
    render(<PlotlyArtifactView artifact={chart({ figure: undefined })} />);
    expect(screen.getByText(/Chart could not be rendered/)).toBeInTheDocument();
  });

  it("shows a graceful error when the figure is malformed", () => {
    render(<PlotlyArtifactView artifact={chart({ figure: { data: "nope" } })} />);
    expect(screen.getByText(/missing or malformed/)).toBeInTheDocument();
  });
});

describe("ArtifactRenderer chart routing — native first", () => {
  it("renders bar via Recharts even when a figure is present (Plotly is not default)", () => {
    const { container } = render(<ArtifactRenderer artifact={chart({ chartType: "bar", figure: FIGURE })} />);
    expect(container.querySelector(".recharts-responsive-container")).toBeTruthy();
    expect(screen.queryByTestId("plotly-stub")).toBeNull();
  });

  it("renders scatter via Recharts", () => {
    const a = chart({
      chartType: "scatter",
      xKey: "age",
      yKey: "ltv",
      xLabel: "Age",
      yLabel: "LTV",
      series: [
        { key: "age", label: "Age", color: "#919dd1" },
        { key: "ltv", label: "LTV", color: "#232d55" },
      ],
      rows: [{ age: 70, ltv: 30 }],
    });
    const { container } = render(<ArtifactRenderer artifact={a} />);
    expect(container.querySelector(".recharts-responsive-container")).toBeTruthy();
  });

  it("renders heatmap natively from grid keys (no Plotly)", () => {
    const a = chart({
      chartType: "heatmap",
      xKey: "region",
      yKey: "ltv",
      valueKey: "balance",
      series: [],
      rows: [
        { region: "London", ltv: "30-40%", balance: 50 },
        { region: "London", ltv: "40-50%", balance: 30 },
        { region: "Scotland", ltv: "30-40%", balance: 20 },
      ],
      figure: FIGURE, // present, but native should win
    });
    render(<ArtifactRenderer artifact={a} />);
    expect(screen.getByText("London")).toBeInTheDocument();
    expect(screen.getByText("30-40%")).toBeInTheDocument();
    expect(screen.queryByTestId("plotly-stub")).toBeNull();
  });

  it("renders treemap natively via Recharts (no Plotly)", () => {
    const a = chart({
      chartType: "treemap",
      xKey: "region",
      valueKey: "balance",
      series: [],
      rows: [
        { region: "London", balance: 184 },
        { region: "Scotland", balance: 48 },
      ],
      figure: FIGURE,
    });
    const { container } = render(<ArtifactRenderer artifact={a} />);
    expect(container.querySelector(".recharts-responsive-container")).toBeTruthy();
    expect(screen.queryByTestId("plotly-stub")).toBeNull();
  });

  it("falls back to themed Plotly for heatmap when native grid keys are absent", async () => {
    const a = chart({ chartType: "heatmap", xKey: undefined, yKey: undefined, valueKey: undefined, series: [], rows: [], figure: FIGURE });
    render(<ArtifactRenderer artifact={a} />);
    expect(await screen.findByTestId("plotly-stub")).toBeInTheDocument();
  });

  it("shows unsupported for a heatmap with neither native keys nor a figure", () => {
    const a = chart({ chartType: "heatmap", xKey: undefined, series: [], rows: [], figure: undefined });
    render(<ArtifactRenderer artifact={a} />);
    expect(screen.getByText(/No native renderer is available/)).toBeInTheDocument();
  });
});
