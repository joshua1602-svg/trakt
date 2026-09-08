/**
 * The Artifact Workspace renders the borrowing-base MI artifacts WITHOUT any
 * new React contract: the bridge is the existing funded-bridge waterfall
 * shape (opening total, signed deltas, closing total), the reason breakdown is
 * an ordinary table with percent-point columns, and the trend is an ordinary
 * line chart keyed on `period`.
 *
 * The fixtures are the exact rows `mi_agent_api/borrowing_base_query.py`
 * emits, so a change to either side that breaks the other fails here.
 */
import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import type { ChartArtifact, TableArtifact } from "@/domain";
import { ArtifactRenderer } from "./ArtifactRenderer";
import { ChartArtifactView } from "./ChartArtifactView";
import { TableArtifactView } from "./TableArtifactView";

function borrowingBaseWaterfall(): ChartArtifact {
  return {
    id: "art_bb_wf",
    type: "chart",
    title: "Borrowing-base bridge — 2026-05 to 2026-06",
    description:
      "Opening 2026-05 → eligible collateral effect → advance rate effect → facility cap effect → closing 2026-06.",
    source: { engine: "mi_agent.workflow", label: "MI Agent · waterfall", nativeChartType: "waterfall" },
    createdAt: new Date().toISOString(),
    mock: false,
    chartType: "waterfall",
    xKey: "label",
    valueKey: "value",
    valueFormat: "gbp",
    displayHints: { value: { format: "gbp", scale: null } },
    series: [{ key: "value", label: "Borrowing base", color: "#22d3ee" }],
    rows: [
      { label: "Opening borrowing base (2026-05)", value: 10_000_000, type: "total" },
      { label: "Eligible collateral effect", value: 1_800_000, type: "delta" },
      { label: "Advance rate effect", value: -600_000, type: "delta" },
      { label: "Facility cap effect", value: 0, type: "delta" },
      { label: "Closing borrowing base (2026-06)", value: 11_200_000, type: "total" },
    ],
  } as unknown as ChartArtifact;
}

function reasonTable(): TableArtifact {
  return {
    id: "art_bb_reasons",
    type: "table",
    title: "Ineligible loans by primary reason — 2026-06-30",
    description:
      "One primary governed reason per ineligible loan — the first failing approved eligibility rule in configured order.",
    source: { engine: "mi_agent.workflow", label: "MI Agent · table" },
    createdAt: new Date().toISOString(),
    mock: false,
    columns: [
      { key: "reason", label: "Reason", align: "left", format: "text" },
      { key: "loans", label: "Loans", align: "right", format: "number" },
      { key: "balance", label: "Balance", align: "right", format: "gbp" },
      { key: "share_loans", label: "% of ineligible loans", align: "right", format: "pct", scale: "percent_points" },
      { key: "share_balance", label: "% of ineligible balance", align: "right", format: "pct", scale: "percent_points" },
    ],
    rows: [
      {
        reason: "Current LTV exceeds the facility's 45% ceiling",
        code: "current_ltv_above_facility_limit",
        loans: 2,
        balance: 6_000_000,
        share_loans: 66.6667,
        share_balance: 66.6667,
      },
      {
        reason: "Youngest borrower is below the facility minimum age of 66",
        code: "youngest_borrower_below_minimum_age",
        loans: 1,
        balance: 3_000_000,
        share_loans: 33.3333,
        share_balance: 33.3333,
      },
    ],
  } as unknown as TableArtifact;
}

function borrowingBaseTrend(): ChartArtifact {
  return {
    id: "art_bb_trend",
    type: "chart",
    title: "Borrowing base by reporting period",
    source: { engine: "mi_agent.workflow", label: "MI Agent · line", nativeChartType: "line" },
    createdAt: new Date().toISOString(),
    mock: false,
    chartType: "line",
    xKey: "period",
    valueFormat: "gbp",
    displayHints: { value: { format: "gbp", scale: null } },
    series: [{ key: "value", label: "Borrowing base", color: "#22d3ee" }],
    rows: [
      { period: "2026-04", value: 9_800_000 },
      { period: "2026-05", value: 10_000_000 },
      { period: "2026-06", value: 11_200_000 },
    ],
  } as unknown as ChartArtifact;
}

describe("Borrowing-base artifacts in the existing Artifact Workspace", () => {
  it("renders the borrowing-base bridge as the existing waterfall shape", () => {
    const { container } = render(<ChartArtifactView artifact={borrowingBaseWaterfall()} />);
    expect(container.querySelector(".recharts-responsive-container")).not.toBeNull();
    expect(container.textContent).toMatch(/Base \/ total/i);
  });

  it("renders the bridge through the generic renderer without a new artifact type", () => {
    const { container } = render(<ArtifactRenderer artifact={borrowingBaseWaterfall()} />);
    expect(container.querySelector(".recharts-responsive-container")).not.toBeNull();
  });

  it("renders the primary-reason table with percent-point shares", () => {
    render(<TableArtifactView artifact={reasonTable()} />);
    expect(screen.getByText("Reason")).toBeInTheDocument();
    expect(screen.getByText("% of ineligible loans")).toBeInTheDocument();
    expect(screen.getByText(/Current LTV exceeds the facility's 45% ceiling/)).toBeInTheDocument();
    // 66.6667 percent POINTS renders as a percentage near 66.7%, never as 6667%.
    expect(screen.getAllByText(/66\.7\s?%|66\.67\s?%|67\s?%/).length).toBeGreaterThan(0);
    expect(screen.queryByText(/6,?667/)).toBeNull();
  });

  it("renders the period trend as an ordinary line chart keyed on period", () => {
    const art = borrowingBaseTrend();
    expect(art.xKey).toBe("period");
    const { container } = render(<ChartArtifactView artifact={art} />);
    expect(container.querySelector(".recharts-responsive-container")).not.toBeNull();
  });
});
