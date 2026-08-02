/**
 * Phase 2A — the presentation-only context blocks.
 *
 * Neither of these fetches anything: every field is already on the wire. What
 * they must get right is separation and dating — a weekly pipeline number must
 * never read as a refresh of the monthly funded actuals, a stress scenario must
 * never read as a forecast, and a metric with no valid contributor split must
 * say so instead of showing one.
 */

import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { ConcentrationContext } from "@/components/insight/ConcentrationContext";
import { ConversionContext } from "@/components/insight/ConversionContext";
import { InsightDetailDrawer } from "@/components/insight/InsightDetailDrawer";
import type { ConcentrationTest, FunnelConversion, MovementDetail } from "@/domain";
import { DETAIL_PIPELINE } from "@/domain";

const TEST = {
  testId: "t1", displayName: "Largest region share", category: "geography",
  reportingDate: "2026-07-31", currentValue: 0.31, priorValue: 0.29,
  threshold: 0.35, utilization: 0.886, headroom: 0.04, status: "warning",
  expected: { value: 0.36, status: "breach", headroom: -0.01, utilization: 1.03,
              breachAmount: 0.01 },
  fullPipeline: { value: 0.41, status: "breach", headroom: -0.06,
                  utilization: 1.17, breachAmount: 0.06 },
  expectedBreachHorizon: { available: true, period: "2026-10" },
} as unknown as ConcentrationTest;

const CONVERSION: FunnelConversion = {
  basis: "avg_weekly_flow_over_lagged_kfi_stock",
  lagWeeks: 6, lagApplied: true, denominatorWeek: "2026-06-26",
  avgWeeklyFlowCount: 21, avgWeeklyFlowValue: 4_100_000,
  kfiStockCount: 512, kfiStockValue: 96_000_000,
  weeklyRateCount: 4.1, weeklyRateValue: 4.3,
  weeksInWindow: 5, minWeeks: 4, sufficient: true,
} as unknown as FunnelConversion;

describe("ConcentrationContext", () => {
  it("dates the current position to the FUNDED cut, not the weekly pipeline", () => {
    render(<ConcentrationContext test={TEST} fundedReportingDate="2026-07-31"
      statesAvailable forecast={{ available: true, methodology: "completion_trend_model",
        observationWindowEnd: "2026-08-07" }} />);
    expect(screen.getByText("Funded actuals as of 2026-07-31")).toBeTruthy();
    expect(screen.getByText(/model observed to 2026-08-07/)).toBeTruthy();
  });

  it("keeps the three states separate and labels the stress one hypothetical", () => {
    render(<ConcentrationContext test={TEST} fundedReportingDate="2026-07-31"
      statesAvailable />);
    expect(screen.getByText("Current funded position")).toBeTruthy();
    expect(screen.getByText("Expected forecast position")).toBeTruthy();
    expect(screen.getByText("All pipeline converts")).toBeTruthy();
    expect(screen.getByText("Hypothetical")).toBeTruthy();
    expect(screen.getByText(/Not a forecast/)).toBeTruthy();
  });

  it("shows the projected breach horizon", () => {
    render(<ConcentrationContext test={TEST} statesAvailable />);
    expect(screen.getByText("2026-10")).toBeTruthy();
  });

  it("says why there is no forecast rather than showing an empty state", () => {
    render(<ConcentrationContext test={TEST} statesAvailable={false}
      statesReason="No active in-scope pipeline cases are available." />);
    expect(screen.getByTestId("concentration-context-no-forecast").textContent)
      .toContain("No active in-scope pipeline");
    expect(screen.queryByText("All pipeline converts")).toBeNull();
  });

  it("still renders the funded position when the forecast is unavailable", () => {
    render(<ConcentrationContext test={TEST} statesAvailable={false} />);
    expect(screen.getByText("Current funded position")).toBeTruthy();
  });
});

describe("ConversionContext", () => {
  it("shows the numerator, denominator, window and denominator week", () => {
    render(<ConversionContext stage="COMPLETED" conversion={CONVERSION}
      cohortPct={62.4} latestWeek="2026-08-07" />);
    const el = screen.getByTestId("conversion-context-COMPLETED");
    expect(el.textContent).toContain("avg_weekly_flow_over_lagged_kfi_stock");
    expect(el.textContent).toContain("£4.1MM");      // numerator value
    expect(el.textContent).toContain("512 KFIs");    // denominator count
    expect(el.textContent).toContain("6w lag applied");
    expect(el.textContent).toContain("5 of 4+ weeks");
    expect(el.textContent).toContain("2026-06-26");  // denominator week
    expect(el.textContent).toContain("2026-08-07");  // latest extract
  });

  it("declines to invent a prior-period comparison", () => {
    render(<ConversionContext stage="COMPLETED" conversion={CONVERSION} cohortPct={62.4} />);
    expect(screen.getByTestId("conversion-context-no-prior-COMPLETED").textContent)
      .toMatch(/not shown rather than derived/);
  });

  it("states that contributor attribution is not valid for conversion", () => {
    render(<ConversionContext stage="COMPLETED" conversion={CONVERSION} cohortPct={62.4} />);
    const el = screen.getByTestId("conversion-context-no-attribution-COMPLETED");
    expect(el.textContent).toMatch(/not available for conversion/);
    expect(el.textContent).toMatch(/lagged KFI\s+stock/);
  });

  it("flags a provisional window", () => {
    render(<ConversionContext stage="OFFER" cohortPct={10}
      conversion={{ ...CONVERSION, sufficient: false, weeksInWindow: 2 }} />);
    expect(screen.getByTestId("conversion-context-provisional-OFFER")).toBeTruthy();
  });
});

describe("InsightDetailDrawer", () => {
  const detail: MovementDetail = {
    detail_type: DETAIL_PIPELINE, portfolio_id: "ERE", scope: "total",
    as_of_date: "2026-08-07", comparison_date: "2026-07-31", available: true,
    headline_metric: { label: "Pipeline balance", value: 563_400_000,
                       change: 26_400_000, change_pct: 4.9 },
    counts: { current: 1500, comparison: 1428, change: 72 },
    contributors: {
      brokers: [{ name: "Air", amount: 18_100_000, share_of_change_pct: 68.6, case_count: 22 }],
      regions: [{ name: "South East", amount: -9_400_000, share_of_change_pct: -35.6, case_count: 12 }],
    },
    components: {
      new: { amount: 30_000_000, cases: 80 }, increased: { amount: 4_000_000, cases: 40 },
      decreased: { amount: -6_000_000, cases: 30 },
      progressed_out: { amount: -1_600_000, cases: 8 },
      removed: { amount: 0, cases: 0 }, unchanged: { amount: 0, cases: 5 },
    },
    methodology: {
      metric_definition: "Total open pipeline exposure.", movement_basis: "net",
      attribution: "current_period_dimension_prior_for_removed", version: "1",
      dimension_reassignments: { broker_channel: 3, geographic_region_obligor: 0 },
    },
    sources: { current: "w_2026-08-07.csv", comparison: "w_2026-07-31.csv" },
  };

  it("shows ranked contributors with share and case count", () => {
    render(<InsightDetailDrawer detail={detail} onClose={() => {}} />);
    expect(screen.getByText("Air")).toBeTruthy();
    expect(screen.getByText("68.6%")).toBeTruthy();
    expect(screen.getByText("22")).toBeTruthy();
  });

  it("frames contributions as changes, not current balances", () => {
    render(<InsightDetailDrawer detail={detail} onClose={() => {}} />);
    expect(screen.getAllByText(/contributions are\s+changes, not current balances/).length)
      .toBeGreaterThan(0);
  });

  it("breaks the movement down without merging the components", () => {
    render(<InsightDetailDrawer detail={detail} onClose={() => {}} />);
    const el = screen.getByTestId("insight-drawer-components");
    expect(el.textContent).toContain("New cases");
    expect(el.textContent).toContain("Left active pipeline");
    expect(el.textContent).toContain("Removed cases");
  });

  it("discloses a dimension reassignment rather than hiding the convention", () => {
    render(<InsightDetailDrawer detail={detail} onClose={() => {}} />);
    expect(screen.getByTestId("insight-drawer-methodology").textContent)
      .toContain("3 case(s) changed broker_channel");
  });

  it("carries no case-level rows", () => {
    const { container } = render(
      <InsightDetailDrawer detail={detail} onClose={() => {}} />);
    expect(container.textContent).not.toMatch(/CASE\d/);
  });

  it("reports an unavailable detail plainly", () => {
    render(<InsightDetailDrawer onClose={() => {}}
      detail={{ ...detail, available: false, headline_metric: null,
                reason: "No prior week." }} />);
    expect(screen.getByTestId("insight-drawer-unavailable").textContent)
      .toContain("No prior week.");
  });

  it("is a dialog with an accessible name", () => {
    render(<InsightDetailDrawer detail={detail} onClose={() => {}} />);
    expect(screen.getByRole("dialog", { name: "Movement details" })).toBeTruthy();
  });
});
