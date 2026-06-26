import { describe, expect, it } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import type { PipelineSnapshot } from "@/domain";
import { PipelineSnapshotPanel } from "./PipelineSnapshotPanel";
import { mockForecastSnapshot } from "@/data/mockForecast";

const NOV = mockForecastSnapshot("client_001/mi_2025_11").pipelineSnapshot!;

describe("PipelineSnapshotPanel", () => {
  it("renders the pipeline SSoT KPIs (separate from the funded book)", () => {
    render(<PipelineSnapshotPanel snapshot={NOV} />);
    expect(screen.getByText("Pipeline Snapshot")).toBeInTheDocument();
    expect(screen.getByText(/Origination pipeline \(pre-funded\)/)).toBeInTheDocument();
    expect(screen.getByText("Pipeline cases")).toBeInTheDocument();
    expect(screen.getByText("10")).toBeInTheDocument();
    expect(screen.getByText("Weighted expected funded")).toBeInTheDocument();
  });

  it("renders the pipeline stage breakdown (amount and count)", () => {
    render(<PipelineSnapshotPanel snapshot={NOV} />);
    expect(screen.getByText("Pipeline amount by stage")).toBeInTheDocument();
    expect(screen.getByText("Pipeline count by stage")).toBeInTheDocument();
    // Stage labels appear in the breakdown bars.
    expect(screen.getAllByText("OFFER").length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText("COMPLETED").length).toBeGreaterThanOrEqual(1);
  });

  it("renders the expected completion breakdown when months exist", () => {
    render(<PipelineSnapshotPanel snapshot={NOV} />);
    expect(screen.getByText("Weighted expected funded by completion month")).toBeInTheDocument();
    // 2025-12 appears in both the "next completions" tile and the month bar.
    expect(screen.getAllByText("2025-12").length).toBeGreaterThanOrEqual(1);
  });

  it("caps the broker/channel breakdown at top 10 with an Other row", () => {
    render(<PipelineSnapshotPanel snapshot={NOV} />);
    expect(screen.getByText("Pipeline amount by broker / channel")).toBeInTheDocument();
    // The backend caps to 9 named brokers + Other (=10 rows max).
    expect(NOV.brokerBreakdown!.length).toBeLessThanOrEqual(10);
    expect(screen.getAllByText("Other").length).toBeGreaterThanOrEqual(1);
    expect(screen.getByText("Broker Alpha")).toBeInTheDocument();
  });

  it("hides technical diagnostics until expanded", () => {
    const snap: PipelineSnapshot = {
      ...NOV,
      dataQuality: [{ check: "amount_parse_partial", severity: "warning", detail: "economic amount parsed for 9/10 rows" }],
    };
    render(<PipelineSnapshotPanel snapshot={snap} />);
    expect(screen.queryByText(/economic amount parsed for 9\/10 rows/)).not.toBeInTheDocument();
    fireEvent.click(screen.getByText(/Technical details/));
    expect(screen.getByText(/economic amount parsed for 9\/10 rows/)).toBeInTheDocument();
  });

  it("uses the first future month for 'Next expected completions', not a past month", () => {
    // As-of 2025-11: 2025-10 is overdue, 2025-12 is the next (future) month.
    const snap: PipelineSnapshot = {
      ...NOV,
      pipelineAsOfDate: "2025-11-01",
      expectedCompletionBreakdown: [
        { month: "2025-10", caseCount: 34, expectedFundedAmount: 5_000_000, weightedExpectedFundedAmount: 4_200_000 },
        { month: "2025-11", caseCount: 12, expectedFundedAmount: 1_200_000, weightedExpectedFundedAmount: 900_000 },
        { month: "2025-12", caseCount: 20, expectedFundedAmount: 2_000_000, weightedExpectedFundedAmount: 1_500_000 },
      ],
      expectedCompletionSummary: {
        asOfMonth: "2025-11",
        overdueExpectedCompletionCount: 34,
        overdueExpectedCompletionWeightedAmount: 4_200_000,
        currentMonthExpectedCompletionCount: 12,
        currentMonthExpectedCompletionWeightedAmount: 900_000,
        nextExpectedCompletionMonth: "2025-12",
        nextExpectedCompletionCount: 20,
        nextExpectedCompletionWeightedAmount: 1_500_000,
      },
      nextExpectedCompletionMonth: "2025-12",
      overdueExpectedCompletionCount: 34,
      overdueExpectedCompletionWeightedAmount: 4_200_000,
    };
    render(<PipelineSnapshotPanel snapshot={snap} />);
    const nextTile = screen.getByText("Next expected completions").parentElement!;
    // The next tile shows the FUTURE month 2025-12, never the overdue 2025-10.
    expect(nextTile.textContent).toContain("2025-12");
    expect(nextTile.textContent).not.toContain("2025-10");
    // Overdue is exposed separately.
    expect(screen.getByText("Overdue expected completions")).toBeInTheDocument();
    expect(screen.getByText(/before as-of month/)).toBeInTheDocument();
  });

  it("shows 'None' when there are only past expected completions", () => {
    const snap: PipelineSnapshot = {
      ...NOV,
      pipelineAsOfDate: "2025-11-01",
      expectedCompletionBreakdown: [
        { month: "2025-09", caseCount: 5, expectedFundedAmount: 100, weightedExpectedFundedAmount: 50 },
        { month: "2025-10", caseCount: 3, expectedFundedAmount: 80, weightedExpectedFundedAmount: 40 },
      ],
      expectedCompletionSummary: {
        asOfMonth: "2025-11",
        overdueExpectedCompletionCount: 8,
        overdueExpectedCompletionWeightedAmount: 90,
        currentMonthExpectedCompletionCount: 0,
        currentMonthExpectedCompletionWeightedAmount: 0,
        nextExpectedCompletionMonth: null,
        nextExpectedCompletionCount: 0,
        nextExpectedCompletionWeightedAmount: 0,
      },
      nextExpectedCompletionMonth: null,
      overdueExpectedCompletionCount: 8,
    };
    render(<PipelineSnapshotPanel snapshot={snap} />);
    const nextTile = screen.getByText("Next expected completions").parentElement!;
    expect(nextTile.textContent).toContain("None");
    expect(screen.getByText("Overdue expected completions")).toBeInTheDocument();
  });

  it("shows week-on-week deltas on pipeline tiles when a prior week exists", () => {
    render(<PipelineSnapshotPanel snapshot={NOV} />);
    // Pipeline cases 10 vs prior 9 → +1 vs prior week.
    expect(screen.getByText(/\+1 vs prior week/)).toBeInTheDocument();
    // Pipeline amount delta is GBP-formatted.
    expect(screen.getAllByText(/vs prior week/).length).toBeGreaterThanOrEqual(2);
  });

  it("shows 'No prior week' and never invents a delta when prior data is absent", () => {
    const snap: PipelineSnapshot = { ...NOV, priorWeek: null };
    render(<PipelineSnapshotPanel snapshot={snap} />);
    expect(screen.getAllByText("No prior week").length).toBeGreaterThanOrEqual(1);
    expect(screen.queryByText(/vs prior week/)).not.toBeInTheDocument();
  });

  it("renders an unavailable state gracefully", () => {
    const snap = { ...NOV, ok: false, error: "No pipeline data for this reporting date." };
    render(<PipelineSnapshotPanel snapshot={snap} />);
    expect(screen.getByText("Pipeline Snapshot unavailable")).toBeInTheDocument();
    expect(screen.getByText(/No pipeline data for this reporting date/)).toBeInTheDocument();
  });
});
