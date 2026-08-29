/**
 * Commercial go-live sprint — stratification drill-downs.
 *
 * Both capabilities are PRESENTATION ONLY, and these tests pin that:
 *
 *  * the balance / % of book / loans toggle reads three measures the
 *    deterministic engine already returned in the same payload — the browser
 *    never re-aggregates to produce one;
 *  * selecting a band asks the governed MI engine for that population rather
 *    than deriving a loan list locally.
 */

import { describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen, within } from "@testing-library/react";
import { FundedSnapshotPanel } from "@/components/FundedSnapshotPanel";
import type { FundedSnapshot } from "@/domain";

const SNAPSHOT: FundedSnapshot = {
  ok: true,
  portfolio: { client_id: "ERE", label: "ERE", run_id: "2026-07-31",
               reporting_date: "2026-07-31" },
  prior: null,
  loan_count: 300,
  current_outstanding_balance: 30_000_000,
  currencyCode: "GBP",
  kpis: [],
  stratifications: [{
    key: "ltv", label: "By LTV band", availability: "available",
    bars: [
      { label: "20-30%", balance: 20_000_000, count: 200, sharePct: 66.7 },
      { label: "30-40%", balance: 10_000_000, count: 100, sharePct: 33.3 },
    ],
  }],
  warnings: [],
  diagnostics: [],
} as unknown as FundedSnapshot;

function bars() {
  return within(screen.getByTestId("strat-ltv"));
}

describe("stratification measure toggle", () => {
  it("defaults to balance and renders the payload's balances", () => {
    render(<FundedSnapshotPanel snapshot={SNAPSHOT} />);
    expect(bars().getByText("£20.0MM")).toBeInTheDocument();
    expect(bars().getByText(/£10.0MM/)).toBeInTheDocument();
  });

  it("shows the share the engine returned, not one it recomputes", () => {
    render(<FundedSnapshotPanel snapshot={SNAPSHOT} />);
    fireEvent.click(screen.getByTestId("strat-measure-share"));
    // 66.7 / 33.3 are the engine's sharePct values, carried through unchanged.
    expect(bars().getByText("66.7%")).toBeInTheDocument();
    expect(bars().getByText("33.3%")).toBeInTheDocument();
  });

  it("shows loan counts without inventing a total", () => {
    render(<FundedSnapshotPanel snapshot={SNAPSHOT} />);
    fireEvent.click(screen.getByTestId("strat-measure-count"));
    expect(bars().getByText("200")).toBeInTheDocument();
    expect(bars().getByText("100")).toBeInTheDocument();
  });

  it("switching measure never alters the bands themselves", () => {
    render(<FundedSnapshotPanel snapshot={SNAPSHOT} />);
    const bands = () => bars().getAllByTitle(/^\d+-\d+%$/).map((n) => n.textContent);
    const before = bands();
    fireEvent.click(screen.getByTestId("strat-measure-count"));
    expect(bands()).toEqual(before);
    expect(before).toEqual(["20-30%", "30-40%"]);
  });
});

describe("stratification drill-down", () => {
  it("asks the governed engine for the selected band", () => {
    const onDrill = vi.fn();
    render(<FundedSnapshotPanel snapshot={SNAPSHOT} onDrill={onDrill} />);
    fireEvent.click(bars().getByRole("button", { name: /20-30%/ }));
    expect(onDrill).toHaveBeenCalledWith("By LTV band", "20-30%");
  });

  it("renders inert bars when no drill handler is supplied", () => {
    render(<FundedSnapshotPanel snapshot={SNAPSHOT} />);
    expect(bars().queryAllByRole("button")).toHaveLength(0);
  });
});
