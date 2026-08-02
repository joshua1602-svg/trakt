/**
 * Phase 2A — the hover layer must be invisible when off and honest when on.
 *
 * The tests that matter most are the containment ones: with the flag off the
 * chart renders exactly as it does in production today, and no request is ever
 * issued. After that, the presentation has to be trustworthy — negative
 * contributors shown as negative, unknown names shown rather than dropped, and
 * a failure shown as "no detail" rather than as an error.
 */

import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { useState } from "react";
import { describe, expect, it, vi } from "vitest";

import { ContributorList } from "@/components/insight/ContributorList";
import { EnhancedMetricTooltip } from "@/components/insight/EnhancedMetricTooltip";
import { InsightLineChart } from "@/components/insight/InsightLineChart";
import { useMovementDetail } from "@/hooks/useMovementDetail";
import { enhancedHoversEnabled } from "@/lib/featureFlags";
import { DETAIL_PIPELINE, type MovementDetail } from "@/domain";
import type { AgentClient } from "@/api";

const SERIES = [
  { period: "2026-07-24", pipeline_amount: 520_000_000 },
  { period: "2026-07-31", pipeline_amount: 537_000_000 },
  { period: "2026-08-07", pipeline_amount: 563_400_000 },
];

function detail(over: Partial<MovementDetail> = {}): MovementDetail {
  return {
    detail_type: DETAIL_PIPELINE,
    portfolio_id: "ERE",
    scope: "total",
    as_of_date: "2026-08-07",
    comparison_date: "2026-07-31",
    available: true,
    headline_metric: {
      label: "Pipeline balance", value: 563_400_000,
      change: 26_400_000, change_pct: 4.9,
    },
    counts: { current: 1500, comparison: 1428, change: 72 },
    contributors: {
      brokers: [
        { name: "Air", amount: 18_100_000, share_of_change_pct: 68.6, case_count: 22 },
        { name: "Fluent", amount: 5_300_000, share_of_change_pct: 20.1, case_count: 9 },
        { name: "HUB", amount: 2_100_000, share_of_change_pct: 8.0, case_count: 4 },
      ],
      regions: [
        { name: "South East", amount: 9_400_000, share_of_change_pct: 35.6, case_count: 12 },
      ],
    },
    components: {
      new: { amount: 30_000_000, cases: 80 },
      increased: { amount: 4_000_000, cases: 40 },
      decreased: { amount: -6_000_000, cases: 30 },
      progressed_out: { amount: -1_600_000, cases: 8 },
      removed: { amount: 0, cases: 0 },
      unchanged: { amount: 0, cases: 5 },
    },
    methodology: { movement_basis: "net", attribution: "current_period_dimension_prior_for_removed" },
    ...over,
  };
}

function client(impl?: AgentClient["getMovementDetail"]): AgentClient {
  return { getMovementDetail: impl } as unknown as AgentClient;
}

// --------------------------------------------------------------------------
describe("feature flag", () => {
  it("is off unless explicitly opted in", () => {
    expect(enhancedHoversEnabled({})).toBe(false);
    expect(enhancedHoversEnabled({ VITE_MI_ENHANCED_HOVERS: "" })).toBe(false);
    expect(enhancedHoversEnabled({ VITE_MI_ENHANCED_HOVERS: "0" })).toBe(false);
    expect(enhancedHoversEnabled({ VITE_MI_ENHANCED_HOVERS: "maybe" })).toBe(false);
  });

  it("accepts the usual truthy spellings", () => {
    for (const v of ["1", "true", "TRUE", "on", "yes", "enabled"]) {
      expect(enhancedHoversEnabled({ VITE_MI_ENHANCED_HOVERS: v })).toBe(true);
    }
  });
});

// --------------------------------------------------------------------------
describe("InsightLineChart containment", () => {
  it("renders the plain chart and issues NO request when disabled", async () => {
    const spy = vi.fn();
    render(
      <InsightLineChart title="Pipeline amount by week" data={SERIES}
        lines={[{ key: "pipeline_amount", label: "Pipeline amount" }]}
        enabled={false} client={client(spy)} portfolioId="ERE"
        detailType={DETAIL_PIPELINE} />,
    );
    expect(screen.getByText("Pipeline amount by week")).toBeTruthy();
    expect(screen.queryByTestId("insight-line-chart")).toBeNull();
    await new Promise((r) => setTimeout(r, 400));
    expect(spy).not.toHaveBeenCalled();
  });

  it("wraps the same chart when enabled", () => {
    render(
      <InsightLineChart title="Pipeline amount by week" data={SERIES}
        lines={[{ key: "pipeline_amount", label: "Pipeline amount" }]}
        enabled client={client(vi.fn())} portfolioId="ERE"
        detailType={DETAIL_PIPELINE} />,
    );
    expect(screen.getByTestId("insight-line-chart")).toBeTruthy();
    expect(screen.getByText("Pipeline amount by week")).toBeTruthy();
  });

  it("exposes a keyboard route to the detail, so hover is not the only way in", async () => {
    render(
      <InsightLineChart title="Pipeline amount by week" data={SERIES}
        lines={[{ key: "pipeline_amount", label: "Pipeline amount" }]}
        enabled client={client(vi.fn().mockResolvedValue(detail()))}
        portfolioId="ERE" detailType={DETAIL_PIPELINE} drillDown />,
    );
    const el = screen.getByTestId("insight-line-chart");
    expect(el.getAttribute("tabindex")).toBe("0");
    expect(el.getAttribute("role")).toBe("button");
    fireEvent.keyDown(el, { key: "Enter" });
    // Opens the SAME contained panel a click opens, on the latest week.
    await waitFor(() => expect(screen.getByTestId("insight-drawer")).toBeTruthy());
  });

  it("closes the drill panel on Escape and on the close button", async () => {
    render(
      <InsightLineChart title="Pipeline amount by week" data={SERIES}
        lines={[{ key: "pipeline_amount", label: "Pipeline amount" }]}
        enabled client={client(vi.fn().mockResolvedValue(detail()))}
        portfolioId="ERE" detailType={DETAIL_PIPELINE} drillDown />,
    );
    fireEvent.keyDown(screen.getByTestId("insight-line-chart"), { key: "Enter" });
    await waitFor(() => expect(screen.getByTestId("insight-drawer")).toBeTruthy());
    fireEvent.keyDown(document, { key: "Escape" });
    await waitFor(() => expect(screen.queryByTestId("insight-drawer")).toBeNull());

    fireEvent.keyDown(screen.getByTestId("insight-line-chart"), { key: "Enter" });
    await waitFor(() => expect(screen.getByTestId("insight-drawer")).toBeTruthy());
    fireEvent.click(screen.getByLabelText("Close details"));
    await waitFor(() => expect(screen.queryByTestId("insight-drawer")).toBeNull());
  });

  it("reuses ONE payload for the hover and the panel — no second fetch", async () => {
    const spy = vi.fn().mockResolvedValue(detail());
    render(
      <InsightLineChart title="Pipeline amount by week" data={SERIES}
        lines={[{ key: "pipeline_amount", label: "Pipeline amount" }]}
        enabled client={client(spy)} portfolioId="ERE"
        detailType={DETAIL_PIPELINE} drillDown />,
    );
    fireEvent.keyDown(screen.getByTestId("insight-line-chart"), { key: "Enter" });
    await waitFor(() => expect(screen.getByTestId("insight-drawer")).toBeTruthy());
    await new Promise((r) => setTimeout(r, 400));
    expect(spy).toHaveBeenCalledTimes(1);
  });

  it("does not remount the chart when the detail arrives", async () => {
    // A remount would reset Recharts' animation and internal state, and would
    // read as a flicker under the pointer. The chart is memo'd and its props
    // are stable, so the same DOM node must survive the state change.
    const spy = vi.fn().mockResolvedValue(detail());
    const { container } = render(
      <InsightLineChart title="Pipeline amount by week" data={SERIES}
        lines={[{ key: "pipeline_amount", label: "Pipeline amount" }]}
        enabled client={client(spy)} portfolioId="ERE"
        detailType={DETAIL_PIPELINE} />,
    );
    const before = container.querySelector(".recharts-responsive-container")
      ?? container.firstElementChild;
    await new Promise((r) => setTimeout(r, 400));
    const after = container.querySelector(".recharts-responsive-container")
      ?? container.firstElementChild;
    expect(after).toBe(before);
  });

  it("does not navigate — the detail never leaves the page", () => {
    const before = window.location.href;
    render(
      <InsightLineChart title="Pipeline amount by week" data={SERIES}
        lines={[{ key: "pipeline_amount", label: "Pipeline amount" }]}
        enabled client={client(vi.fn().mockResolvedValue(detail()))}
        portfolioId="ERE" detailType={DETAIL_PIPELINE} drillDown />,
    );
    fireEvent.keyDown(screen.getByTestId("insight-line-chart"), { key: "Enter" });
    expect(window.location.href).toBe(before);
  });

  it("is not focusable when there is nothing to open", () => {
    render(
      <InsightLineChart title="Pipeline amount by week" data={SERIES}
        lines={[{ key: "pipeline_amount", label: "Pipeline amount" }]}
        enabled client={client(vi.fn())} portfolioId="ERE"
        detailType={DETAIL_PIPELINE} />,
    );
    expect(screen.getByTestId("insight-line-chart").getAttribute("role")).toBeNull();
  });
});

// --------------------------------------------------------------------------
describe("useMovementDetail request discipline", () => {
  function Harness({ c, weeks }: { c: AgentClient; weeks: (string | null)[] }) {
    const [i, setI] = useState(0);
    const s = useMovementDetail({
      client: c, portfolioId: "ERE", detailType: DETAIL_PIPELINE,
      asOf: weeks[i] ?? null, enabled: true, debounceMs: 40,
    });
    return (
      <div>
        <button onClick={() => setI((n) => n + 1)}>next</button>
        <span data-testid="state">
          {s.loading ? "loading" : s.unavailable ? "unavailable"
            : s.detail ? "ready" : "idle"}
        </span>
      </div>
    );
  }

  it("issues ONE request for a pointer sweeping across many weeks", async () => {
    const spy = vi.fn().mockResolvedValue(detail());
    render(<Harness c={client(spy)} weeks={SERIES.map((s) => s.period)} />);
    // Sweep: change the active point twice inside the debounce window.
    fireEvent.click(screen.getByText("next"));
    fireEvent.click(screen.getByText("next"));
    await waitFor(() => expect(screen.getByTestId("state").textContent).toBe("ready"));
    expect(spy).toHaveBeenCalledTimes(1);
    expect(spy).toHaveBeenCalledWith("ERE", DETAIL_PIPELINE, "2026-08-07", undefined);
  });

  it("issues nothing at all while no point is selected", async () => {
    const spy = vi.fn().mockResolvedValue(detail());
    render(<Harness c={client(spy)} weeks={[null]} />);
    await new Promise((r) => setTimeout(r, 200));
    expect(spy).not.toHaveBeenCalled();
  });

  it("reports a rejection as 'no detail', never as an error", async () => {
    const spy = vi.fn().mockRejectedValue(new Error("404"));
    render(<Harness c={client(spy)} weeks={["2026-08-07"]} />);
    await waitFor(() =>
      expect(screen.getByTestId("state").textContent).toBe("unavailable"));
  });

  it("reports an unavailable payload as unavailable", async () => {
    const spy = vi.fn().mockResolvedValue(
      detail({ available: false, headline_metric: null }));
    render(<Harness c={client(spy)} weeks={["2026-08-07"]} />);
    await waitFor(() =>
      expect(screen.getByTestId("state").textContent).toBe("unavailable"));
  });
});

// --------------------------------------------------------------------------
describe("EnhancedMetricTooltip", () => {
  // Mirrors how Recharts calls it: props are injected into the TOP-LEVEL
  // tooltip element only, so the base must be a render function that receives
  // them. Passing a pre-built node here would hide the very bug this shape
  // exists to prevent.
  const renderBase = (p: { active?: boolean; label?: string | number }) => (
    <div data-testid="base-tooltip">
      {p.active ? `original content @ ${p.label}` : "original content"}
    </div>
  );

  it("always renders the chart's own tooltip content first", () => {
    render(<EnhancedMetricTooltip renderBase={renderBase} detail={null} />);
    expect(screen.getByTestId("base-tooltip")).toBeTruthy();
    expect(screen.queryByTestId("movement-detail-body")).toBeNull();
  });

  it("forwards Recharts' own props to the base renderer", () => {
    // The regression a browser run caught: the base was a pre-built node, so
    // Recharts' active/payload/label never reached it and the chart's own
    // tooltip content silently vanished behind the enhancement.
    render(<EnhancedMetricTooltip renderBase={renderBase} detail={detail()}
      active label="2026-08-07" payload={[{ name: "Pipeline amount", value: 1 }]} />);
    expect(screen.getByTestId("base-tooltip").textContent)
      .toBe("original content @ 2026-08-07");
  });

  it("shows a loading state without hiding the original numbers", () => {
    render(<EnhancedMetricTooltip renderBase={renderBase} detail={null} loading />);
    expect(screen.getByTestId("base-tooltip")).toBeTruthy();
    expect(screen.getByTestId("movement-detail-loading")).toBeTruthy();
  });

  it("explains why there is no detail rather than showing nothing", () => {
    render(<EnhancedMetricTooltip renderBase={renderBase}
      detail={detail({ available: false, reason: "No prior week." })} unavailable />);
    expect(screen.getByTestId("movement-detail-unavailable").textContent)
      .toContain("No prior week.");
  });

  it("shows the movement, both contributor lists and the case count", () => {
    render(<EnhancedMetricTooltip renderBase={renderBase} detail={detail()} />);
    const body = screen.getByTestId("movement-detail-body");
    expect(body.textContent).toContain("Pipeline balance");
    expect(body.textContent).toContain("+4.9%");
    expect(body.textContent).toContain("Air");
    expect(body.textContent).toContain("South East");
    expect(body.textContent).toContain("1,500 cases");
    expect(body.textContent).toContain("+72 vs prior week");
  });

  it("states both dates so a weekly movement cannot read as a funded refresh", () => {
    render(<EnhancedMetricTooltip renderBase={renderBase} detail={detail()} />);
    const body = screen.getByTestId("movement-detail-body");
    expect(body.textContent).toContain("2026-08-07");
    expect(body.textContent).toContain("2026-07-31");
    expect(body.textContent).toContain("net movement");
  });

  it("renders a negative movement as negative", () => {
    render(<EnhancedMetricTooltip renderBase={renderBase} detail={detail({
      headline_metric: { label: "Pipeline balance", value: 1, change: -9_093_043,
                         change_pct: -19.4 },
    })} />);
    const body = screen.getByTestId("movement-detail-body");
    expect(body.textContent).toContain("−£9.1MM");
    expect(body.textContent).toContain("-19.4%");
  });

  it("only hints at a drill-down when one is actually wired up", () => {
    const { rerender } = render(
      <EnhancedMetricTooltip renderBase={renderBase} detail={detail()} />);
    expect(screen.queryByText(/Click the chart/)).toBeNull();
    rerender(<EnhancedMetricTooltip renderBase={renderBase} detail={detail()} showDrillHint />);
    expect(screen.getByText(/Click the chart/)).toBeTruthy();
  });
});

// --------------------------------------------------------------------------
describe("ContributorList", () => {
  const rows = [
    { name: "Air", amount: 18_100_000, share_of_change_pct: 68.6, case_count: 22 },
    { name: "Fluent", amount: -5_300_000, share_of_change_pct: -20.1, case_count: 9 },
  ];

  it("preserves the order it was given — ranking is the backend's decision", () => {
    render(<ContributorList title="Top brokers" rows={rows} />);
    const items = screen.getAllByRole("listitem").map((li) => li.textContent);
    expect(items[0]).toContain("Air");
    expect(items[1]).toContain("Fluent");
  });

  it("shows a negative contributor with its sign", () => {
    render(<ContributorList title="Top brokers" rows={rows} />);
    expect(screen.getByText("−£5.3MM")).toBeTruthy();
  });

  it("shows an unknown contributor rather than dropping it", () => {
    render(<ContributorList title="Top brokers" rows={[
      { name: "Unknown", amount: 1_000_000, share_of_change_pct: 3.8, case_count: 2 }]} />);
    expect(screen.getByText("Unknown")).toBeTruthy();
  });

  it("keeps a long name on one line and in the title attribute", () => {
    const long = "A Very Long Introducer Name That Would Otherwise Reflow The Tooltip";
    render(<ContributorList title="Top brokers" rows={[
      { name: long, amount: 1, share_of_change_pct: 1, case_count: 1 }]} />);
    const el = screen.getByTitle(long);
    expect(el.className).toContain("truncate");
  });

  it("says so when there are no contributors", () => {
    render(<ContributorList title="Top brokers" rows={[]} />);
    expect(screen.getByText(/No contributors/)).toBeTruthy();
  });

  it("never shows more than the top three", () => {
    render(<ContributorList title="Top brokers" rows={[1, 2, 3, 4, 5].map((i) => ({
      name: `B${i}`, amount: i, share_of_change_pct: i, case_count: i }))} />);
    expect(screen.getAllByRole("listitem")).toHaveLength(3);
  });
});
