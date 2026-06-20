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

const base = {
  id: "c1",
  title: "Chart",
  createdAt: "2026-05-31T08:00:00Z",
  mock: false,
};

function chart(partial: Partial<ChartArtifact> & { figure?: unknown }): ChartArtifact {
  const { figure, ...rest } = partial;
  return {
    ...base,
    type: "chart",
    chartType: "bar",
    xKey: "region",
    series: [{ key: "balance", label: "Balance", color: "#919dd1" }],
    rows: [{ region: "London", balance: 184 }],
    source: { engine: "mi_agent.workflow", label: "MI Agent · bar", figure },
    ...rest,
  } as ChartArtifact;
}

const FIGURE = { data: [{ type: "bar", x: ["London"], y: [184] }], layout: {} };

describe("PlotlyArtifactView", () => {
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

describe("ArtifactRenderer chart routing", () => {
  it("routes a chart with a Plotly figure to the Plotly renderer", async () => {
    render(<ArtifactRenderer artifact={chart({ chartType: "bar", figure: FIGURE })} />);
    expect(await screen.findByTestId("plotly-stub")).toBeInTheDocument();
  });

  it("routes a heatmap WITH a figure to the Plotly renderer (no longer degraded)", async () => {
    const heatmap = chart({ chartType: "heatmap", series: [], figure: { data: [{ type: "heatmap" }] } });
    render(<ArtifactRenderer artifact={heatmap} />);
    expect(await screen.findByTestId("plotly-stub")).toBeInTheDocument();
  });

  it("routes a treemap WITH a figure to the Plotly renderer", async () => {
    const treemap = chart({ chartType: "treemap", series: [], figure: { data: [{ type: "treemap" }] } });
    render(<ArtifactRenderer artifact={treemap} />);
    expect(await screen.findByTestId("plotly-stub")).toBeInTheDocument();
  });

  it("falls back to the Recharts renderer when no figure is present", () => {
    const { container } = render(<ArtifactRenderer artifact={chart({ chartType: "bar", figure: undefined })} />);
    expect(container.querySelector(".recharts-responsive-container")).toBeTruthy();
    expect(screen.queryByTestId("plotly-stub")).toBeNull();
  });

  it("shows unsupported for a heatmap with no figure", () => {
    const heatmap = chart({ chartType: "heatmap", series: [], figure: undefined });
    render(<ArtifactRenderer artifact={heatmap} />);
    expect(screen.getByText(/require a Plotly figure/)).toBeInTheDocument();
  });
});
