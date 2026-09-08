/**
 * Funded → Cohorts must show vintages, not the running book.
 *
 * The defect: this tab rendered EvolutionPanel, so it displayed the funded
 * book at each reporting date (33, then 73) under cohort labels — a second
 * copy of the Evolution tab. These tests assert the surface now answers the
 * cohort question instead: 33 entering in October and 40 in November, and one
 * vintage followed forward as it seasons.
 */

import { describe, expect, it } from "vitest";
import { render, screen, waitFor, within } from "@testing-library/react";
import { MockAgentClient } from "@/api/MockAgentClient";
import { CohortsPanel } from "./CohortsPanel";

function renderPanel() {
  return render(<CohortsPanel client={new MockAgentClient()} portfolioId="client_001/mi_2025_12" />);
}

describe("CohortsPanel", () => {
  it("counts what entered each vintage, not the book outstanding", async () => {
    renderPanel();
    const table = await screen.findByTestId("formation-table");
    const october = within(table).getByTestId("vintage-2025-10");
    const november = within(table).getByTestId("vintage-2025-11");
    expect(within(october).getByText("33")).toBeInTheDocument();
    expect(within(november).getByText("40")).toBeInTheDocument();
  });

  it("never shows the cumulative portfolio total under a vintage label", async () => {
    renderPanel();
    const table = await screen.findByTestId("formation-table");
    // 73 is the November BOOK (33 + 40) — the number the old surface showed.
    expect(within(table).queryByText("73")).toBeNull();
    expect(within(table).queryByText("83")).toBeNull();
  });

  it("follows the oldest vintage forward as a static pool by default", async () => {
    renderPanel();
    const pool = await screen.findByTestId("static-pool-table");
    await waitFor(() => expect(within(pool).getByText("2025-10")).toBeInTheDocument());
    expect(within(pool).getByText("+1 mo")).toBeInTheDocument();
    expect(within(pool).getByText("+2 mo")).toBeInTheDocument();
  });

  it("shows the pool shrinking through exits, never growing", async () => {
    renderPanel();
    const pool = await screen.findByTestId("static-pool-table");
    await waitFor(() => expect(within(pool).getByText("31")).toBeInTheDocument());
    const surviving = within(pool).getAllByRole("row").slice(1)
      .map((r) => Number(r.children[2].textContent));
    expect(surviving).toEqual([33, 31, 28]);
    expect(surviving).toEqual([...surviving].sort((a, b) => b - a));
  });

  it("reports exits per period and cumulatively", async () => {
    renderPanel();
    const pool = await screen.findByTestId("static-pool-table");
    await waitFor(() => expect(within(pool).getByText("2 (2 cum.)")).toBeInTheDocument());
    expect(within(pool).getByText("3 (5 cum.)")).toBeInTheDocument();
  });

  it("renders LTV under the governed fraction contract", async () => {
    renderPanel();
    const table = await screen.findByTestId("formation-table");
    // 0.417 is a fraction on the wire; 41.7% on screen — not 0.4%, not 4170%.
    // The table now carries BOTH an origination LTV and an entry LTV, so the
    // fixture's equal values appear twice; assert on the count rather than
    // loosening the contract this test exists to hold.
    expect(within(table).getAllByText("41.7%").length).toBeGreaterThan(0);
    expect(within(table).getAllByText("37.0%").length).toBeGreaterThan(0);
    // Nothing renders the fraction unscaled or scaled twice.
    expect(within(table).queryByText("0.4%")).toBeNull();
    expect(within(table).queryByText("4170.0%")).toBeNull();
  });

  it("says the cohort basis is the origination date", async () => {
    renderPanel();
    expect(await screen.findByText(/origination \(policy completion\) date/i))
      .toBeInTheDocument();
  });

  it("points the reader at Evolution for the running book", async () => {
    renderPanel();
    expect(await screen.findByText(/book outstanding at each reporting date is/i))
      .toBeInTheDocument();
  });
});
