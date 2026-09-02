/**
 * The balance / count switch on the Pipeline and Forecast breakdowns.
 *
 * One seam, three panels. Every measure a toggle offers is a field the
 * deterministic engine already returned in the SAME payload, so switching only
 * chooses which of them to draw — the browser never re-aggregates, and a
 * breakdown that carries no count is never offered one.
 */

import { describe, expect, it } from "vitest";
import { fireEvent, render, screen, within } from "@testing-library/react";
import { PipelineSnapshotPanel } from "@/components/PipelineSnapshotPanel";
import { ForecastView } from "@/components/ForecastView";
import { MeasureToggle } from "@/components/pipeline/bits";
import type { ForecastSnapshot } from "@/domain";
import { mockForecastSnapshot } from "@/data/mockForecast";

const PIPELINE = mockForecastSnapshot("client_001/mi_2025_11").pipelineSnapshot!;

const FORECAST = {
  forecastBridge: null,
  forecastBreakdowns: {
    byRegionCapped: [
      { key: "Greater London", caseCount: 125, pipelineAmount: 58_200_000,
        weightedExpectedFundedAmount: null },
    ],
    byLtvBucketCapped: [
      { key: "20-30%", caseCount: 40, pipelineAmount: 12_000_000,
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
  it("switches region and LTV onto their case counts", () => {
    render(<ForecastView forecast={FORECAST} />);
    expect(screen.getByText("£58.2MM")).toBeInTheDocument();
    fireEvent.click(screen.getByTestId("forecast-measure-count"));
    expect(screen.getByText("125")).toBeInTheDocument();
    expect(screen.getByText("40")).toBeInTheDocument();
  });

  it("leaves completion month on balance — it carries no count", () => {
    render(<ForecastView forecast={FORECAST} />);
    fireEvent.click(screen.getByTestId("forecast-measure-count"));
    const month = screen.getByText("Forecast contribution by completion month")
      .closest("div")!.parentElement!;
    expect(within(month).getByText("£11.1MM")).toBeInTheDocument();
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
