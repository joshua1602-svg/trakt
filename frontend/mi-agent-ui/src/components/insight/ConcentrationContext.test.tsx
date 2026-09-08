/**
 * Reported defect: the "Position by state" drill-down read 763.0% / 1500.0% /
 * 5087.0% for a test the table row directly above it showed as 7.63% / 15.00%
 * / 50.9% utilisation — the same fields, off by a factor of 100.
 *
 * Cause: `currentValue` / `threshold` / `utilization` / `headroom` are ALREADY
 * percent numbers on the wire (7.63 means 7.63%) — the same convention
 * `concentrationShared.formatValue`/`formatChange` and the server's own
 * `_fmt` use. `pct()`/`pp()` here multiplied by 100 on top of that.
 */

import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { ConcentrationContext } from "./ConcentrationContext";
import { formatValue } from "@/components/risk/concentrationShared";
import { mockConcentrationTests } from "@/data/mockConcentrationTests";

const EAST_ANGLIA = mockConcentrationTests("client_001").tests.find(
  (t) => t.testId === "ct_east_anglia",
)!;

describe("ConcentrationContext — Position by state", () => {
  it("renders the same number the risk-limits table row renders, not that number times 100", () => {
    render(
      <ConcentrationContext
        test={EAST_ANGLIA}
        fundedReportingDate="2026-06-30"
        statesAvailable
      />,
    );
    // What the table row (concentrationShared.formatValue) shows for the
    // identical fields — the cross-check that pins the two surfaces together.
    expect(formatValue(EAST_ANGLIA.currentValue, EAST_ANGLIA.unit)).toBe("18.40%");
    expect(formatValue(EAST_ANGLIA.threshold, EAST_ANGLIA.unit)).toBe("25.00%");

    expect(screen.getByText("18.4%")).toBeInTheDocument();
    expect(screen.getByText("25.0%")).toBeInTheDocument();
    expect(screen.getByText("73.6%")).toBeInTheDocument();
    expect(screen.getByText("6.6pp")).toBeInTheDocument();

    // The defect in one assertion: none of the ×100 values may appear.
    expect(screen.queryByText(/1840/)).toBeNull();
    expect(screen.queryByText(/2500/)).toBeNull();
    expect(screen.queryByText(/7360/)).toBeNull();
  });

  it("scales the expected-forecast and full-pipeline blocks the same way", () => {
    render(
      <ConcentrationContext
        test={EAST_ANGLIA}
        fundedReportingDate="2026-06-30"
        statesAvailable
      />,
    );
    // expected: { value: 25.4, utilization: 101.6, headroom: -0.4 }
    expect(screen.getByText("25.4%")).toBeInTheDocument();
    expect(screen.getByText("101.6%")).toBeInTheDocument();
    // fullPipeline: { value: 27.1, utilization: 108.4, headroom: -2.1 }
    expect(screen.getByText("27.1%")).toBeInTheDocument();
    expect(screen.getByText("108.4%")).toBeInTheDocument();
  });
});
