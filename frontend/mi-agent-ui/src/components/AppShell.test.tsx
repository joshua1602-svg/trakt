import { describe, expect, it } from "vitest";
import { render, screen, waitFor, within } from "@testing-library/react";
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

  it("reporting-date dropdown only offers discovered runs (no prototype dates)", async () => {
    render(<AppShell />);
    const select = await screen.findByRole("combobox");
    const options = within(select).getAllByRole("option").map((o) => o.textContent);
    // Exactly the two discovered reporting dates — never the old 2026 prototypes.
    expect(options).toEqual(["31 Oct 2025", "30 Nov 2025"]);
    expect(options).not.toContain("31 May 2026");
  });
});
