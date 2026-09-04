/**
 * Phase 3A — the Weekly Portfolio Brief panel.
 *
 * The containment tests come first: with the flag off, or against an API that
 * does not serve the brief, the panel must render NOTHING — not an empty box,
 * not a placeholder — so the dashboard is byte-identical to today.
 *
 * After that: the panel must present what the API said and nothing else. It
 * computes no numbers, invents no wording, and reorders nothing.
 */

import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { WeeklyBriefPanel } from "@/components/insight/WeeklyBriefPanel";
import { weeklyBriefEnabled } from "@/lib/featureFlags";
import type { Insight, WeeklyBrief } from "@/domain";
import type { AgentClient } from "@/api";

function insight(over: Partial<Insight> = {}): Insight {
  return {
    insight_id: `id-${over.insight_type ?? "X"}-${over.severity ?? "info"}`,
    insight_type: "PIPELINE_MOVEMENT", version: "1",
    tenant_id: "ERE", portfolio_id: "ERE", portfolio_context: "total",
    as_of_date: "2026-08-07", comparison_date: "2026-07-31",
    severity: "info", priority: 70,
    headline: "Pipeline increased 4.8%",
    summary: "Pipeline increased by £2.4m, driven mainly by Air and South East.",
    deep_link: "/mi/pipeline",
    ...over,
  };
}

function brief(over: Partial<WeeklyBrief> = {}): WeeklyBrief {
  return {
    brief_version: "1", status: "success", tenant_id: "ERE",
    portfolio_id: "ERE", portfolio_context: "total",
    as_of_date: "2026-08-07", comparison_date: "2026-07-31",
    insight_count: 1, insights: [insight()],
    omitted: [{ insight_type: "TICKET_SIZE", category: "immaterial",
                reason: "Average ticket moved +1.1%, below threshold." }],
    source_dates: { funded_as_of: "2026-07-31" },
    ...over,
  };
}

function client(impl?: AgentClient["getWeeklyBrief"]): AgentClient {
  return { getWeeklyBrief: impl } as unknown as AgentClient;
}

const PROPS = { portfolioId: "ERE/2026-07-31" };

// --------------------------------------------------------------------------
describe("feature flag", () => {
  it("is off unless explicitly opted in", () => {
    expect(weeklyBriefEnabled({})).toBe(false);
    expect(weeklyBriefEnabled({ VITE_MI_WEEKLY_BRIEF: "0" })).toBe(false);
    expect(weeklyBriefEnabled({ VITE_MI_WEEKLY_BRIEF: "later" })).toBe(false);
  });

  it("accepts the usual truthy spellings", () => {
    for (const v of ["1", "true", "on", "YES", "enabled"]) {
      expect(weeklyBriefEnabled({ VITE_MI_WEEKLY_BRIEF: v })).toBe(true);
    }
  });
});

// --------------------------------------------------------------------------
describe("containment", () => {
  it("renders nothing and issues NO request when disabled", async () => {
    const spy = vi.fn();
    const { container } = render(
      <WeeklyBriefPanel {...PROPS} client={client(spy)} enabled={false} />);
    await new Promise((r) => setTimeout(r, 100));
    expect(container.firstChild).toBeNull();
    expect(spy).not.toHaveBeenCalled();
  });

  it("renders nothing when the API does not serve the brief", async () => {
    const { container } = render(
      <WeeklyBriefPanel {...PROPS} enabled
        client={client(vi.fn().mockRejectedValue(new Error("404")))} />);
    await waitFor(() => expect(container.firstChild).toBeNull());
  });

  it("renders nothing when the brief reports itself unavailable", async () => {
    const { container } = render(
      <WeeklyBriefPanel {...PROPS} enabled client={client(
        vi.fn().mockResolvedValue(brief({ status: "unavailable",
                                          insights: [], omitted: [] })))} />);
    await waitFor(() => expect(container.firstChild).toBeNull());
  });
});

// --------------------------------------------------------------------------
describe("presentation", () => {
  it("shows a loading state once the deferred fetch starts", async () => {
    render(<WeeklyBriefPanel {...PROPS} enabled
      client={client(vi.fn().mockReturnValue(new Promise(() => {})))} />);
    await waitFor(() =>
      expect(screen.getByTestId("weekly-brief-loading")).toBeTruthy());
  });

  it("renders the insights the API returned, in the order it returned them", async () => {
    const b = brief({
      insight_count: 3,
      insights: [
        insight({ insight_type: "CONCENTRATION_PROXIMITY", severity: "concern",
                  headline: "South West is at 93% of its limit", priority: 80 }),
        insight({ insight_type: "DATA_QUALITY", severity: "attention",
                  headline: "1 data-quality threshold breached", priority: 70 }),
        insight({ insight_type: "TICKET_SIZE", severity: "info",
                  headline: "Median ticket size decreased 6.3%", priority: 60 }),
      ],
    });
    render(<WeeklyBriefPanel {...PROPS} enabled
      client={client(vi.fn().mockResolvedValue(b))} />);
    await waitFor(() => expect(screen.getByTestId("weekly-brief-list")).toBeTruthy());
    const types = screen.getAllByTestId("insight-card")
      .map((el) => el.getAttribute("data-insight-type"));
    expect(types).toEqual(["CONCENTRATION_PROXIMITY", "DATA_QUALITY", "TICKET_SIZE"]);
  });

  it("does not reorder or re-rank — ordering is the engine's decision", async () => {
    // Deliberately "wrong" order from the API: the panel must not fix it.
    const b = brief({
      insight_count: 2,
      insights: [insight({ insight_type: "TICKET_SIZE", severity: "info" }),
                 insight({ insight_type: "DATA_QUALITY", severity: "concern" })],
    });
    render(<WeeklyBriefPanel {...PROPS} enabled
      client={client(vi.fn().mockResolvedValue(b))} />);
    await waitFor(() => expect(screen.getByTestId("weekly-brief-list")).toBeTruthy());
    expect(screen.getAllByTestId("insight-card")
      .map((el) => el.getAttribute("data-insight-type")))
      .toEqual(["TICKET_SIZE", "DATA_QUALITY"]);
  });

  it("renders up to eight items without truncating what it was given", async () => {
    const b = brief({
      insight_count: 8,
      insights: Array.from({ length: 8 }, (_v, i) =>
        insight({ insight_type: "TICKET_SIZE", headline: `h${i}` })),
    });
    // Distinct keys so React renders all eight.
    b.insights = b.insights.map((x, i) => ({ ...x, insight_id: `id-${i}` }));
    render(<WeeklyBriefPanel {...PROPS} enabled
      client={client(vi.fn().mockResolvedValue(b))} />);
    await waitFor(() => expect(screen.getAllByTestId("insight-card")).toHaveLength(8));
  });

  it("labels severity distinctly", async () => {
    const b = brief({ insights: [insight({ severity: "concern" })] });
    render(<WeeklyBriefPanel {...PROPS} enabled
      client={client(vi.fn().mockResolvedValue(b))} />);
    await waitFor(() => expect(screen.getByText("Concern")).toBeTruthy());
    expect(screen.getByTestId("insight-card").getAttribute("data-severity"))
      .toBe("concern");
  });

  it("says so plainly when nothing was material", async () => {
    render(<WeeklyBriefPanel {...PROPS} enabled client={client(
      vi.fn().mockResolvedValue(brief({ insight_count: 0, insights: [] })))} />);
    await waitFor(() => expect(screen.getByTestId("weekly-brief-empty").textContent)
      .toContain("No material changes"));
  });

  it("flags a partial brief rather than presenting it as complete", async () => {
    render(<WeeklyBriefPanel {...PROPS} enabled client={client(
      vi.fn().mockResolvedValue(brief({ status: "partial",
                                        reason: "1 insight type could not be produced." })))} />);
    await waitFor(() => expect(screen.getByTestId("weekly-brief-partial").textContent)
      .toContain("could not be produced"));
  });

  it("can explain what it did not show", async () => {
    render(<WeeklyBriefPanel {...PROPS} enabled
      client={client(vi.fn().mockResolvedValue(brief()))} />);
    await waitFor(() => expect(screen.getByTestId("weekly-brief-omissions")).toBeTruthy());
    fireEvent.click(screen.getByText(/1 item not shown/));
    expect(screen.getByText(/below threshold/)).toBeTruthy();
  });

  it("states both dates so a weekly brief cannot read as a funded refresh", async () => {
    render(<WeeklyBriefPanel {...PROPS} enabled
      client={client(vi.fn().mockResolvedValue(brief()))} />);
    await waitFor(() => expect(screen.getByTestId("weekly-brief")).toBeTruthy());
    const header = screen.getByTestId("weekly-brief").textContent ?? "";
    expect(header).toContain("Pipeline 2026-08-07 vs 2026-07-31");
    expect(header).toContain("funded actuals 2026-07-31");
  });
});

// --------------------------------------------------------------------------
describe("staying off the first-paint critical path", () => {
  // The regression this guards against: the brief fired in parallel with the
  // dashboard's own primary requests, so a cold worker prepared the same data
  // twice at once while serving them. First paint went from 2.6s to 4.0s and
  // network idle from 4.1s to 9.9s.

  it("issues NO request until the primary data has resolved", async () => {
    const spy = vi.fn().mockResolvedValue(brief());
    const { rerender } = render(
      <WeeklyBriefPanel {...PROPS} enabled ready={false} client={client(spy)} />);
    await new Promise((r) => setTimeout(r, 250));
    expect(spy).not.toHaveBeenCalled();

    rerender(<WeeklyBriefPanel {...PROPS} enabled ready client={client(spy)} />);
    await waitFor(() => expect(spy).toHaveBeenCalledTimes(1));
  });

  it("renders nothing at all while it is waiting", async () => {
    const { container } = render(
      <WeeklyBriefPanel {...PROPS} enabled ready={false}
        client={client(vi.fn().mockResolvedValue(brief()))} />);
    await new Promise((r) => setTimeout(r, 250));
    expect(container.firstChild).toBeNull();
  });

  it("does not fetch when a portfolio switch makes it un-ready again", async () => {
    const spy = vi.fn().mockResolvedValue(brief());
    const c = client(spy);
    const { rerender } = render(
      <WeeklyBriefPanel portfolioId="A" enabled ready client={c} />);
    await waitFor(() => expect(spy).toHaveBeenCalledTimes(1));
    // New portfolio selected: its snapshot is loading again, so the brief must
    // wait rather than asking about a book the dashboard has not loaded.
    rerender(<WeeklyBriefPanel portfolioId="B" enabled ready={false} client={c} />);
    await new Promise((r) => setTimeout(r, 250));
    expect(spy).toHaveBeenCalledTimes(1);
    rerender(<WeeklyBriefPanel portfolioId="B" enabled ready client={c} />);
    await waitFor(() => expect(spy).toHaveBeenCalledTimes(2));
  });

  it("defaults to ready so a caller that does not care is unaffected", async () => {
    const spy = vi.fn().mockResolvedValue(brief());
    render(<WeeklyBriefPanel {...PROPS} enabled client={client(spy)} />);
    await waitFor(() => expect(spy).toHaveBeenCalledTimes(1));
  });
});

// --------------------------------------------------------------------------
describe("navigation and requests", () => {
  it("deep-links to the existing section without navigating away", async () => {
    const onNavigate = vi.fn();
    const before = window.location.href;
    render(<WeeklyBriefPanel {...PROPS} enabled onNavigate={onNavigate}
      client={client(vi.fn().mockResolvedValue(brief()))} />);
    await waitFor(() => expect(screen.getByTestId("insight-card")).toBeTruthy());
    fireEvent.click(screen.getByTestId("insight-card"));
    expect(onNavigate).toHaveBeenCalledWith("/mi/pipeline");
    expect(window.location.href).toBe(before);
  });

  it("is keyboard reachable", async () => {
    const onNavigate = vi.fn();
    render(<WeeklyBriefPanel {...PROPS} enabled onNavigate={onNavigate}
      client={client(vi.fn().mockResolvedValue(brief()))} />);
    await waitFor(() => expect(screen.getByTestId("insight-card")).toBeTruthy());
    const card = screen.getByTestId("insight-card");
    expect(card.getAttribute("tabindex")).toBe("0");
    fireEvent.keyDown(card, { key: "Enter" });
    expect(onNavigate).toHaveBeenCalled();
  });

  it("is not interactive when there is nowhere to go", async () => {
    render(<WeeklyBriefPanel {...PROPS} enabled client={client(
      vi.fn().mockResolvedValue(brief({ insights: [insight({ deep_link: null })] })))} />);
    await waitFor(() => expect(screen.getByTestId("insight-card")).toBeTruthy());
    expect(screen.getByTestId("insight-card").getAttribute("role")).toBeNull();
  });

  it("issues ONE request and does not refetch on re-render", async () => {
    const spy = vi.fn().mockResolvedValue(brief());
    const { rerender } = render(
      <WeeklyBriefPanel {...PROPS} enabled client={client(spy)} />);
    await waitFor(() => expect(screen.getByTestId("weekly-brief")).toBeTruthy());
    rerender(<WeeklyBriefPanel {...PROPS} enabled client={client(spy)} />);
    rerender(<WeeklyBriefPanel {...PROPS} enabled client={client(spy)} />);
    await new Promise((r) => setTimeout(r, 150));
    expect(spy).toHaveBeenCalledTimes(1);
  });

  it("refetches when the portfolio changes", async () => {
    const spy = vi.fn().mockResolvedValue(brief());
    const c = client(spy);
    const { rerender } = render(
      <WeeklyBriefPanel portfolioId="ERE/2026-07-31" enabled client={c} />);
    await waitFor(() => expect(spy).toHaveBeenCalledTimes(1));
    rerender(<WeeklyBriefPanel portfolioId="ERE/2026-06-30" enabled client={c} />);
    await waitFor(() => expect(spy).toHaveBeenCalledTimes(2));
    expect(spy).toHaveBeenLastCalledWith("ERE/2026-06-30", undefined, undefined);
  });

  it("refetches when the scope changes", async () => {
    const spy = vi.fn().mockResolvedValue(brief());
    const c = client(spy);
    const { rerender } = render(
      <WeeklyBriefPanel {...PROPS} enabled client={c} portfolioContext="total" />);
    await waitFor(() => expect(spy).toHaveBeenCalledTimes(1));
    rerender(<WeeklyBriefPanel {...PROPS} enabled client={c}
      portfolioContext="acquired_001" />);
    await waitFor(() => expect(spy).toHaveBeenCalledTimes(2));
  });
});

describe("marking the brief read", () => {
  beforeEach(() => {
    localStorage.clear();
  });

  it("shows a Read control once the brief has loaded", async () => {
    render(<WeeklyBriefPanel {...PROPS} enabled client={client(vi.fn().mockResolvedValue(brief()))} />);
    await screen.findByTestId("weekly-brief");
    expect(screen.getByTestId("weekly-brief-mark-read")).toBeInTheDocument();
  });

  it("disappears immediately once marked read", async () => {
    render(<WeeklyBriefPanel {...PROPS} enabled client={client(vi.fn().mockResolvedValue(brief()))} />);
    await screen.findByTestId("weekly-brief");
    fireEvent.click(screen.getByTestId("weekly-brief-mark-read"));
    expect(screen.queryByTestId("weekly-brief")).toBeNull();
  });

  it("stays hidden on a later visit to the SAME brief", async () => {
    const spy = vi.fn().mockResolvedValue(brief());
    const { unmount } = render(<WeeklyBriefPanel {...PROPS} enabled client={client(spy)} />);
    await screen.findByTestId("weekly-brief");
    fireEvent.click(screen.getByTestId("weekly-brief-mark-read"));
    unmount();

    // A fresh mount — as a page reload would produce — reads the same
    // persisted dismissal rather than showing the brief again.
    render(<WeeklyBriefPanel {...PROPS} enabled client={client(spy)} />);
    await waitFor(() => expect(spy).toHaveBeenCalled());
    expect(screen.queryByTestId("weekly-brief")).toBeNull();
  });

  it("shows again once a NEW data upload changes the brief's dates", async () => {
    const spy = vi.fn().mockResolvedValue(brief());
    const { unmount } = render(<WeeklyBriefPanel {...PROPS} enabled client={client(spy)} />);
    await screen.findByTestId("weekly-brief");
    fireEvent.click(screen.getByTestId("weekly-brief-mark-read"));
    unmount();

    // Next week's upload: new as_of / comparison / funded dates.
    const nextWeek = client(vi.fn().mockResolvedValue(brief({
      as_of_date: "2026-08-14", comparison_date: "2026-08-07",
      source_dates: { funded_as_of: "2026-08-07" },
    })));
    render(<WeeklyBriefPanel {...PROPS} enabled client={nextWeek} />);
    await screen.findByTestId("weekly-brief");
    expect(screen.getByText(/Pipeline increased 4.8%/)).toBeInTheDocument();
  });

  it("does not break the panel when storage throws", async () => {
    const original = window.localStorage.setItem;
    window.localStorage.setItem = () => {
      throw new Error("storage blocked");
    };
    try {
      render(<WeeklyBriefPanel {...PROPS} enabled client={client(vi.fn().mockResolvedValue(brief()))} />);
      await screen.findByTestId("weekly-brief");
      // The click must not throw even though persistence silently fails.
      expect(() => fireEvent.click(screen.getByTestId("weekly-brief-mark-read")))
        .not.toThrow();
    } finally {
      window.localStorage.setItem = original;
    }
  });
});
