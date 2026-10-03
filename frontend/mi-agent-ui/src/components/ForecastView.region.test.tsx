/**
 * The Forecast tab's region breakdown adds the funded book and the weighted
 * pipeline up in the client's reporting regions; what it cannot place is
 * stated under the chart, as the backend composed it.
 */
import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { ForecastView } from "@/components/ForecastView";
import type { ForecastSnapshot } from "@/domain";

function snapshot(unplaced: number): ForecastSnapshot {
  return {
    forecastBridge: null,
    forecastBreakdowns: {
      byRegion: [],
      byLtvBucket: [],
      byCompletionMonth: [],
      byRegionCapped: [
        { key: "London", caseCount: 0, pipelineAmount: 5_000_000,
          weightedExpectedFundedAmount: null },
      ],
      regionBasis: { field: "canonical_region_reporting",
                     unplacedFundedAmount: unplaced, unplacedWeightedPipelineAmount: 0,
                     unplacedForecastAmount: unplaced },
    },
    watchlist: [],
  } as unknown as ForecastSnapshot;
}

describe("Forecast balance by region", () => {
  it("says how much of the forecast has no region", () => {
    render(<ForecastView forecast={snapshot(250_000)} />);
    expect(screen.getByText(/of the forecast has no region and is not shown/))
      .toBeInTheDocument();
  });

  it("adds nothing when every row is placed", () => {
    render(<ForecastView forecast={snapshot(0)} />);
    expect(screen.queryByText(/has no region/)).not.toBeInTheDocument();
  });
});
