/**
 * One home for follow-ups.
 *
 * Follow-up suggestions used to be built by two engines and rendered in two
 * places: `buildInvestigations` (insight-derived, on the artifact card) and
 * `buildSuggestedActions` (spec-derived, as a chip row in the chat rail).
 * Neither knew the other existed, so a single chart could put six offers on
 * screen in two visual languages — and the chat's copy sat in a rail that had
 * already scrolled away from the chart it referred to.
 *
 * Both engines now feed the card's one row, beside the result they are about.
 */

import { describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen, within } from "@testing-library/react";
import type { ChartArtifact } from "@/domain";
import { InsightPanel } from "./InsightPanel";

function chart(): ChartArtifact {
  return {
    id: "c1",
    type: "chart",
    title: "Balance by Region",
    source: {
      engine: "mi_agent.workflow",
      label: "MI Agent · bar",
      spec: {
        metric: "current_outstanding_balance",
        dimension: "geographic_region_obligor",
      },
      question: "balance by region",
    },
    createdAt: "2026-05-31T08:00:00Z",
    mock: false,
    chartType: "bar",
    xKey: "geographic_region_obligor",
    series: [
      { key: "current_outstanding_balance", label: "Balance", color: "#000" },
    ],
    rows: [
      { geographic_region_obligor: "London", current_outstanding_balance: 62_000_000 },
      { geographic_region_obligor: "South East", current_outstanding_balance: 41_000_000 },
      { geographic_region_obligor: "North West", current_outstanding_balance: 12_000_000 },
      { geographic_region_obligor: "Wales", current_outstanding_balance: 3_000_000 },
    ],
  } as ChartArtifact;
}

function suggestionRow() {
  return screen.getByText("Investigate next").closest("div")!.parentElement!;
}

describe("the single suggestion surface", () => {
  it("carries the spec-derived actions the chat rail used to show", () => {
    render(<InsightPanel artifact={chart()} onAsk={vi.fn()} />);
    const row = suggestionRow();
    // "Split by <dimension>" came from buildSuggestedActions, not the insights.
    expect(within(row).getAllByText(/^Split by /).length).toBeGreaterThan(0);
  });

  it("dispatches the suggestion's question on click", () => {
    const onAsk = vi.fn();
    render(<InsightPanel artifact={chart()} onAsk={onAsk} />);
    const chip = within(suggestionRow()).getAllByRole("button")[0];
    fireEvent.click(chip);
    expect(onAsk).toHaveBeenCalledTimes(1);
    expect(typeof onAsk.mock.calls[0][0]).toBe("string");
  });

  it("never offers the same question twice, however it was built", () => {
    render(<InsightPanel artifact={chart()} onAsk={vi.fn()} />);
    const titles = within(suggestionRow())
      .getAllByRole("button")
      .map((b) => b.getAttribute("title"));
    expect(new Set(titles).size).toBe(titles.length);
  });

  it("caps the row so one chart never becomes a wall of offers", () => {
    render(<InsightPanel artifact={chart()} onAsk={vi.fn()} />);
    expect(within(suggestionRow()).getAllByRole("button").length).toBeLessThanOrEqual(5);
  });

  it("offers nothing that costs a round trip to redraw the same data", () => {
    // "Show as Table" re-parsed a sentence to reach a view the card's own
    // Chart / Table toggle already holds.
    render(<InsightPanel artifact={chart()} onAsk={vi.fn()} />);
    expect(within(suggestionRow()).queryByText(/as a table/i)).toBeNull();
  });

  it("renders the result unaffected when a suggestion builder throws", () => {
    // A spec the action builder cannot ground returns nothing rather than
    // guessing; the observations must still render.
    const ungrounded = { ...chart(), source: { ...chart().source, spec: {} } } as ChartArtifact;
    render(<InsightPanel artifact={ungrounded} onAsk={vi.fn()} />);
    expect(screen.getByText("Key observations")).toBeInTheDocument();
  });
});
