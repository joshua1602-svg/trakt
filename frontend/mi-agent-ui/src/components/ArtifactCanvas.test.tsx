import { describe, expect, it, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import type { ChartArtifact, TableArtifact } from "@/domain";
import { ArtifactCanvas } from "./ArtifactCanvas";
import { kpiArtifact } from "@/data/mockArtifacts";

const ctx = { asOf: "2026-05-31", portfolio: "erm-uk-master" };

const TITLE = "Balance by Age by LTV";
function pairedChart(): ChartArtifact {
  return {
    id: "chart_1", type: "chart", title: TITLE,
    source: { engine: "mi_agent.workflow", label: "MI Agent · bar" },
    createdAt: "2026-05-31T08:00:00Z", mock: false, chartType: "bar", xKey: "bucket",
    series: [{ key: "balance", label: "Balance", color: "#919dd1" }],
    valueFormat: "gbp", displayHints: {},
    rows: [{ bucket: "<60", balance: 700 }, { bucket: "60-80", balance: 300 }],
  } as unknown as ChartArtifact;
}
function pairedTable(): TableArtifact {
  return {
    id: "table_1", type: "table", title: TITLE,
    source: { engine: "mi_agent.workflow", label: "MI Agent · table" },
    createdAt: "2026-05-31T08:00:00Z", mock: false,
    columns: [
      { key: "bucket", label: "Bucket", format: "text" },
      { key: "balance", label: "Balance", format: "gbp" },
    ],
    rows: [{ bucket: "<60", balance: 700 }, { bucket: "60-80", balance: 300 }],
  } as unknown as TableArtifact;
}

describe("ArtifactCanvas declutter controls (A8)", () => {
  it("clears artifacts via the Clear button (view-only)", () => {
    const onClear = vi.fn();
    render(
      <ArtifactCanvas
        artifacts={[kpiArtifact(ctx)]}
        onTogglePin={() => {}}
        isWorking={false}
        portfolioName="ERM UK"
        onClear={onClear}
      />,
    );
    fireEvent.click(screen.getByRole("button", { name: /Clear artifacts/i }));
    expect(onClear).toHaveBeenCalledTimes(1);
  });

  it("collapses and expands the workspace body", () => {
    render(
      <ArtifactCanvas
        artifacts={[kpiArtifact(ctx)]}
        onTogglePin={() => {}}
        isWorking={false}
        portfolioName="ERM UK"
        onClear={() => {}}
      />,
    );
    // Expanded: the KPI artifact title renders.
    expect(screen.getByText("Executive Summary")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: /Collapse artifact workspace/i }));
    expect(screen.queryByText("Executive Summary")).toBeNull();
    expect(screen.getByText(/Workspace collapsed/)).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: /Expand artifact workspace/i }));
    expect(screen.getByText("Executive Summary")).toBeInTheDocument();
  });
});

describe("ArtifactCanvas chart/table grouping (Task 4 — no duplicate buttons)", () => {
  it("groups a same-name chart + table into one artifact with a Chart/Table toggle", () => {
    render(
      <ArtifactCanvas
        artifacts={[pairedChart(), pairedTable()]}
        onTogglePin={() => {}}
        isWorking={false}
        portfolioName="ERM UK"
      />,
    );
    // The card exposes an internal Chart / Table view toggle (one artifact).
    const toggle = screen.getByRole("tablist", { name: "Artifact view" });
    expect(toggle).toBeInTheDocument();
    expect(screen.getByRole("tab", { name: "Chart" })).toBeInTheDocument();
    expect(screen.getByRole("tab", { name: "Table" })).toBeInTheDocument();
    // Only ONE heading with the shared title (no duplicate same-name entries).
    expect(screen.getAllByRole("heading", { name: TITLE })).toHaveLength(1);
  });

  it("shows a single dedup tab button (not two same-name buttons) in Tabs view", () => {
    render(
      <ArtifactCanvas
        artifacts={[pairedChart(), pairedTable()]}
        onTogglePin={() => {}}
        isWorking={false}
        portfolioName="ERM UK"
      />,
    );
    fireEvent.click(screen.getByRole("button", { name: "Tabs" }));
    // Exactly one navigation button carries the shared artifact name.
    expect(screen.getAllByRole("button", { name: TITLE })).toHaveLength(1);
  });

  it("switches the rendered view when the Table toggle is clicked", () => {
    const { container } = render(
      <ArtifactCanvas
        artifacts={[pairedChart(), pairedTable()]}
        onTogglePin={() => {}}
        isWorking={false}
        portfolioName="ERM UK"
      />,
    );
    // Default view is the chart.
    expect(container.querySelector(".recharts-responsive-container")).not.toBeNull();
    fireEvent.click(screen.getByRole("tab", { name: "Table" }));
    // Now the table renders (a real <table>) and the chart is gone.
    expect(container.querySelector("table")).not.toBeNull();
    expect(container.querySelector(".recharts-responsive-container")).toBeNull();
  });
});
