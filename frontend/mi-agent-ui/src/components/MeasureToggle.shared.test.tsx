/**
 * The balance / count switch on the Pipeline and Forecast breakdowns.
 *
 * One seam, three panels. Every measure a toggle offers is a field the
 * deterministic engine already returned in the SAME payload, so switching only
 * chooses which of them to draw — the browser never re-aggregates, and a
 * breakdown that carries no count is never offered one.
 */

import { describe, expect, it } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import { PipelineSnapshotPanel } from "@/components/PipelineSnapshotPanel";
import { ForecastView } from "@/components/ForecastView";
import { MeasureToggle } from "@/components/pipeline/bits";
import type { ForecastSnapshot } from "@/domain";
import { mockForecastSnapshot } from "@/data/mockForecast";

const PIPELINE = mockForecastSnapshot("client_001/mi_2025_11").pipelineSnapshot!;

const FORECAST = {
  forecastBridge: null,
  forecastBreakdowns: {
    // caseCount is a stub 0 on the REAL payload (mi_agent_api/workspace.py
    // forecast_dimension_breakdown composes a probability-weighted BALANCE —
    // funded exposure + amount x completion-probability — never a loan-level
    // count), so the fixture matches that rather than inventing a real one.
    byRegionCapped: [
      { key: "Greater London", caseCount: 0, pipelineAmount: 58_200_000,
        weightedExpectedFundedAmount: null },
    ],
    byLtvBucketCapped: [
      { key: "20-30%", caseCount: 0, pipelineAmount: 12_000_000,
        weightedExpectedFundedAmount: null },
    ],
    // Carries only a weighted amount — no case count.
    byCompletionMonth: [{ month: "2026-02", weightedExpectedFundedAmount: 11_100_000 }],
  },
  watchlist: [],
} as unknown as ForecastSnapshot;

describe("pipeline breakdown measure toggle", () => {
  const region = () => PIPELINE.regionBreakdown![0];

  it("defaults to the amounts the payload returned", () => {
    render(<PipelineSnapshotPanel snapshot={PIPELINE} />);
    expect(screen.getByText("Pipeline amount by region")).toBeInTheDocument();
    expect(screen.getByTestId("pipeline-measure-balance")).toHaveAttribute("aria-pressed", "true");
  });

  it("switches the breakdowns to the case counts in the same payload", () => {
    render(<PipelineSnapshotPanel snapshot={PIPELINE} />);
    fireEvent.click(screen.getByTestId("pipeline-measure-count"));
    expect(screen.getByText("Pipeline count by region")).toBeInTheDocument();
    // The count drawn is the one the engine returned for that region.
    expect(screen.getAllByText(String(region().caseCount)).length).toBeGreaterThan(0);
  });

  it("switching back restores the amount headings exactly", () => {
    render(<PipelineSnapshotPanel snapshot={PIPELINE} />);
    fireEvent.click(screen.getByTestId("pipeline-measure-count"));
    fireEvent.click(screen.getByTestId("pipeline-measure-balance"));
    expect(screen.getByText("Pipeline amount by region")).toBeInTheDocument();
    expect(screen.getByText("Pipeline amount by broker / channel")).toBeInTheDocument();
  });

  it("never invents a measure — the bars keep their own labels", () => {
    render(<PipelineSnapshotPanel snapshot={PIPELINE} />);
    const before = screen.getAllByTitle(/./).length;
    fireEvent.click(screen.getByTestId("pipeline-measure-count"));
    expect(screen.getAllByTitle(/./).length).toBe(before);
  });
});

describe("forecast breakdown measure toggle", () => {
  // Forecast has no case-count measure to toggle to (see the FORECAST fixture
  // comment above), so it offers no toggle at all — not a toggle stuck on
  // "Balance", and not a "0" suffix implying zero cases contributed.
  it("offers no measure toggle — the payload carries no real count", () => {
    render(<ForecastView forecast={FORECAST} />);
    expect(screen.getByText("Forecast balance by region")).toBeInTheDocument();
    expect(screen.getByText("Forecast balance by LTV bucket")).toBeInTheDocument();
    expect(screen.queryByTestId("forecast-measure-count")).toBeNull();
    expect(screen.queryByTestId("forecast-measure-balance")).toBeNull();
  });

  it("never shows the stub caseCount as a '· 0' suffix", () => {
    render(<ForecastView forecast={FORECAST} />);
    expect(screen.getByText("£58.2MM")).toBeInTheDocument();
    expect(screen.queryByText("· 0")).toBeNull();
  });
});

describe("the toggle itself", () => {
  it("renders nothing when only one measure is available", () => {
    const { container } = render(
      <MeasureToggle measures={["balance"]} active="balance"
        onChange={() => {}} testIdPrefix="solo" />,
    );
    expect(container).toBeEmptyDOMElement();
  });

  it("marks the active measure for assistive technology", () => {
    render(<MeasureToggle measures={["balance", "count"]} active="count"
      onChange={() => {}} testIdPrefix="t" />);
    expect(screen.getByTestId("t-count")).toHaveAttribute("aria-pressed", "true");
    expect(screen.getByTestId("t-balance")).toHaveAttribute("aria-pressed", "false");
  });
});
