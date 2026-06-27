import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import type { Artifact } from "@/domain";
import { isArtifact } from "@/domain";
import { ArtifactRenderer } from "./artifacts/ArtifactRenderer";
import { ChatResult } from "./ChatResult";

// These objects mirror the EXACT artifact JSON that mi_agent_api/chat_routing.py
// emits for the routed chat intents, proving the chat/workspace render them
// through the existing artifact union (no new renderer).
const SOURCE = {
  engine: "mi_agent.workflow",
  label: "MI Agent · line",
  spec: {},
  asOf: "2025-11-30",
  portfolio: "client_001/mi_2025_11",
};

const evolutionLineChart: Artifact = {
  id: "art_evo1", type: "chart", title: "Funded balance by month",
  source: { ...SOURCE, nativeChartType: "line" }, createdAt: "2025-11-30T00:00:00Z",
  mock: false, chartType: "line", xKey: "period",
  series: [{ key: "value", label: "Funded balance", color: "#919dd1" }],
  rows: [{ period: "2025-10", value: 11_900_000 }, { period: "2025-11", value: 15_600_000 }],
  valueFormat: "gbp", displayHints: {}, warnings: [],
} as unknown as Artifact;

const compareBarChart: Artifact = {
  id: "art_cmp1", type: "chart", title: "Funded balance: 2025-10 vs 2025-11",
  source: { ...SOURCE, nativeChartType: "bar" }, createdAt: "2025-11-30T00:00:00Z",
  mock: false, chartType: "bar", xKey: "period",
  series: [{ key: "value", label: "Funded balance", color: "#919dd1" }],
  rows: [{ period: "2025-10", value: 11_900_000 }, { period: "2025-11", value: 15_600_000 }],
  valueFormat: "gbp", displayHints: {}, warnings: [],
  reconciliation: { dataset: "funded", coverage_by_balance_pct: 100.0 },
} as unknown as Artifact;

const milestoneTable: Artifact = {
  id: "art_ms1", type: "table", title: "Milestone dates to funding thresholds",
  source: { ...SOURCE, label: "MI Agent · table" }, createdAt: "2025-11-30T00:00:00Z",
  mock: false,
  columns: [
    { key: "threshold", label: "Threshold", align: "left", format: "text" },
    { key: "downside", label: "Downside", align: "right", format: "text" },
    { key: "base", label: "Base", align: "right", format: "text" },
    { key: "upside", label: "Upside", align: "right", format: "text" },
  ],
  rows: [
    { threshold: "£25m", downside: "2026-04", base: "2026-02", upside: "2026-01" },
    { threshold: "£50m", downside: "2027-08", base: "2027-01", upside: "2026-09" },
  ],
} as unknown as Artifact;

const riskLimitsArtifact: Artifact = {
  id: "art_risk1", type: "risk", title: "Concentration vs Schedule 8 limits",
  source: { ...SOURCE, engine: "risk_monitor", label: "Risk monitor · concentration", state: "total_funded" },
  createdAt: "2025-11-30T00:00:00Z", mock: false,
  mode: "limits", dimension: "concentration",
  groups: [
    { name: "London", balance: 27.4, share: 0.274, status: "amber", limit: 0.3, approaching: true },
    { name: "Top 3 brokers", balance: 41.0, share: 0.41, status: "green", limit: 0.45, approaching: false },
  ],
  warnings: [],
} as unknown as Artifact;

describe("chat routing artifacts render via the existing union", () => {
  it("all routed artifacts pass the isArtifact guard", () => {
    for (const a of [evolutionLineChart, compareBarChart, milestoneTable, riskLimitsArtifact]) {
      expect(isArtifact(a)).toBe(true);
    }
  });

  it("renders an evolution trend as a line chart (not Unsupported)", () => {
    const { container } = render(<ArtifactRenderer artifact={evolutionLineChart} />);
    expect(container.querySelector(".recharts-responsive-container")).toBeTruthy();
    expect(screen.queryByText(/No native renderer/)).toBeNull();
  });

  it("renders the forecast milestone table", () => {
    render(<ArtifactRenderer artifact={milestoneTable} />);
    expect(screen.getByText("£50m")).toBeInTheDocument();
    expect(screen.getByText("Threshold")).toBeInTheDocument();
  });

  it("renders the risk-limits concentration artifact with RAG groups", () => {
    render(<ArtifactRenderer artifact={riskLimitsArtifact} />);
    expect(screen.getByText("London")).toBeInTheDocument();
    expect(screen.getByText("Top 3 brokers")).toBeInTheDocument();
  });
});

describe("compact chat behaviour", () => {
  it("shows key numbers + an open-in-workspace link, not a full inline chart", () => {
    render(
      <ChatResult
        artifacts={[compareBarChart, milestoneTable]}
        onTogglePin={() => {}}
        onOpenArtifact={vi.fn()}
      />,
    );
    expect(screen.getByTestId("chat-result-compact")).toBeInTheDocument();
    expect(screen.getByText(/Open chart in workspace/)).toBeInTheDocument();
    expect(screen.getByText(/Open table in workspace/)).toBeInTheDocument();
    // A key-number chip is surfaced compactly (not the full chart inline).
    expect(screen.getByText("Groups:")).toBeInTheDocument();
    expect(screen.queryByText("Show here")).toBeInTheDocument();
  });

  it("surfaces reconciliation coverage as a compact key number", () => {
    // A chart-only result reads coverage from the reconciliation footer.
    render(
      <ChatResult artifacts={[compareBarChart]} onTogglePin={() => {}} onOpenArtifact={vi.fn()} />,
    );
    expect(screen.getByText("Coverage:")).toBeInTheDocument();
  });

  it("renders nothing inline for a controlled insufficient-data response (no artifacts)", () => {
    const { container } = render(
      <ChatResult artifacts={[]} onTogglePin={() => {}} onOpenArtifact={vi.fn()} />,
    );
    // The narrative answer is shown by ChatMessage; ChatResult itself is empty.
    expect(container).toBeEmptyDOMElement();
  });
});
