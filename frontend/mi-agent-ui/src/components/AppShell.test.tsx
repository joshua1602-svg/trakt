import { describe, expect, it } from "vitest";
import { render, screen, waitFor, within, fireEvent } from "@testing-library/react";
import { AppShell } from "./AppShell";

// AppShell with no VITE_AGENT_API_URL uses the MockAgentClient, whose snapshot
// methods return the real funded spine (client_001 · mi_2025_10 / mi_2025_11).
describe("AppShell — data-driven landing", () => {
  it("renders the funded snapshot before any query is submitted", async () => {
    render(<AppShell />);
    await waitFor(() => expect(screen.getByText("Funded Book Snapshot")).toBeInTheDocument());
    // Defaults to the latest run (November · 73 funded loans, ~£8.9MM).
    expect(screen.getByText("Loans funded")).toBeInTheDocument();
    expect(screen.getByText("73")).toBeInTheDocument();
  });

  it("presents the lifecycle top-level tabs (Funded / Pipeline / Forecast / Risk Limits)", async () => {
    render(<AppShell />);
    const nav = await screen.findByRole("tablist", { name: "MI workspace view" });
    const labels = within(nav).getAllByRole("tab").map((t) => t.textContent?.trim());
    expect(labels).toEqual(["Funded", "Pipeline", "Forecast", "Risk Limits"]);
    // Evolution and Geography are no longer top-level tabs.
    expect(labels).not.toContain("Evolution");
    expect(labels).not.toContain("Geography");
  });

  it("hosts Geography as a Funded sub-view (not a top-level tab)", async () => {
    render(<AppShell />);
    await waitFor(() => expect(screen.getByText("Funded Book Snapshot")).toBeInTheDocument());
    const subtabs = screen.getByTestId("funded-subtabs");
    expect(within(subtabs).getByRole("tab", { name: "Stratifications" })).toBeInTheDocument();
    expect(within(subtabs).getByRole("tab", { name: "Geography" })).toBeInTheDocument();
    expect(within(subtabs).getByRole("tab", { name: "Cohorts" })).toBeInTheDocument();
    // Switching to Geography renders the ITL3 exposure view.
    fireEvent.click(within(subtabs).getByRole("tab", { name: "Geography" }));
    await waitFor(() => expect(screen.getAllByText(/Exposure map/i).length).toBeGreaterThan(0));
  });

  it("reporting-date dropdown only offers discovered runs (no prototype dates)", async () => {
    render(<AppShell />);
    const select = await screen.findByRole("combobox");
    const options = within(select).getAllByRole("option").map((o) => o.textContent);
    // Exactly the two discovered reporting dates — never the old 2026 prototypes.
    expect(options).toEqual(["31 Oct 2025", "30 Nov 2025"]);
    expect(options).not.toContain("31 May 2026");
  });
});
