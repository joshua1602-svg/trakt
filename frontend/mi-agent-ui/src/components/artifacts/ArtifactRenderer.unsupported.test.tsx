import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import type { Artifact, ChartArtifact, UnsupportedArtifact } from "@/domain";
import { ArtifactRenderer } from "./ArtifactRenderer";

const base = {
  id: "a1",
  title: "Test",
  createdAt: "2026-05-31T08:00:00Z",
  mock: false,
  source: { engine: "mi_agent.workflow", label: "test" },
};

describe("ArtifactRenderer — unsupported + scatter", () => {
  it("renders an explicit unsupported state with a Plotly note", () => {
    const artifact: UnsupportedArtifact = {
      ...base,
      type: "unsupported",
      reason: "heatmap is not yet rendered in the React UI.",
      source: { ...base.source, figure: { data: [] } },
    };
    render(<ArtifactRenderer artifact={artifact} />);
    expect(screen.getByText(/Not rendered in this view/)).toBeInTheDocument();
    expect(screen.getByText(/heatmap is not yet rendered/)).toBeInTheDocument();
    expect(screen.getByText(/raw Plotly figure was preserved/)).toBeInTheDocument();
  });

  it("falls back to the unsupported view for an unknown type", () => {
    const artifact = { ...base, type: "galaxy-map" } as unknown as Artifact;
    render(<ArtifactRenderer artifact={artifact} />);
    expect(screen.getByText(/No renderer is registered/)).toBeInTheDocument();
  });

  it("renders a scatter chart with x/y series from the adapter mapping", () => {
    const artifact: ChartArtifact = {
      ...base,
      type: "chart",
      chartType: "scatter",
      xKey: "youngest_borrower_age",
      series: [
        { key: "youngest_borrower_age", label: "Age", color: "#919dd1" },
        { key: "current_loan_to_value", label: "LTV", color: "#232d55" },
      ],
      rows: [
        { youngest_borrower_age: 71, current_loan_to_value: 31.4 },
        { youngest_borrower_age: 68, current_loan_to_value: 28.1 },
      ],
      valueFormat: "pct",
    };
    const { container } = render(<ArtifactRenderer artifact={artifact} />);
    expect(container.querySelector(".recharts-responsive-container")).toBeTruthy();
  });
});
