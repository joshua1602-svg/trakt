import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import type { ChartArtifact, KPIArtifact } from "@/domain";
import { InsightPanel } from "./InsightPanel";

function regionChart(): ChartArtifact {
  return {
    id: "art",
    type: "chart",
    title: "Balance by Region",
    source: { engine: "mi_agent.workflow", label: "MI Agent · bar", spec: { metric: "balance", dimension: "region" } },
    createdAt: "2026-06-26T08:00:00Z",
    mock: false,
    chartType: "bar",
    xKey: "region",
    series: [{ key: "balance", label: "Balance", color: "#000" }],
    valueFormat: "gbp",
    displayHints: { balance: { format: "gbp", scale: null } },
    rows: [
      { region: "London", balance: 700 },
      { region: "South East", balance: 200 },
      { region: "South West", balance: 100 },
    ],
  };
}

describe("InsightPanel", () => {
  it("renders Key Observations and a working investigation chip", () => {
    const onAsk = vi.fn();
    render(<InsightPanel artifact={regionChart()} onAsk={onAsk} />);
    expect(screen.getByText(/Key observations/i)).toBeInTheDocument();
    expect(screen.getByText(/London has the largest balance/i)).toBeInTheDocument();
    const chip = screen.getByRole("button", { name: "Investigate London" });
    fireEvent.click(chip);
    expect(onAsk).toHaveBeenCalledWith("only London");
  });

  it("renders nothing when there is nothing to say (single bucket)", () => {
    const art = regionChart();
    art.rows = [{ region: "London", balance: 100 }];
    const { container } = render(<InsightPanel artifact={art} />);
    expect(container).toBeEmptyDOMElement();
  });

  it("renders nothing for a non-chart/table artifact", () => {
    const kpi: KPIArtifact = {
      id: "k",
      type: "kpi",
      title: "KPIs",
      source: { engine: "mi_agent.workflow", label: "MI Agent" },
      createdAt: "2026-06-26T08:00:00Z",
      mock: false,
      kpis: [],
    };
    const { container } = render(<InsightPanel artifact={kpi} />);
    expect(container).toBeEmptyDOMElement();
  });
});
