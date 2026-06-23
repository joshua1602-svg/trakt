import { describe, expect, it } from "vitest";
import { render, screen, waitFor, fireEvent } from "@testing-library/react";
import { AppShell } from "./AppShell";

// AppShell (no VITE_AGENT_API_URL) uses the MockAgentClient, whose forecast
// snapshot mirrors the real funded spine + Phase 1 pipeline fixture pack.
describe("AppShell — pipeline + forecast landing sections", () => {
  it("renders the funded section unchanged alongside pipeline + forecast", async () => {
    render(<AppShell />);
    await waitFor(() => expect(screen.getByText("Funded Book Snapshot")).toBeInTheDocument());
    // Funded section intact.
    expect(screen.getByText("Loans funded")).toBeInTheDocument();
    expect(screen.getByText("73")).toBeInTheDocument();
    // New Phase 2 sections present.
    await waitFor(() => expect(screen.getByText("Funded + Pipeline Forecast")).toBeInTheDocument());
    expect(screen.getByText("Pipeline Snapshot")).toBeInTheDocument();
    expect(screen.getByText("Pipeline Watchlist")).toBeInTheDocument();
  });

  it("forecast bridge shows funded + weighted pipeline = forecast", async () => {
    render(<AppShell />);
    await waitFor(() => expect(screen.getByText("Forecast funded balance")).toBeInTheDocument());
    // November: £8.9MM funded + £1.1MM weighted = £10.0MM forecast.
    expect(screen.getByText("£10.0MM")).toBeInTheDocument();
  });

  it("reporting-date selector drives funded + pipeline + forecast together", async () => {
    render(<AppShell />);
    const select = await screen.findByRole("combobox");
    // Default November forecast.
    await waitFor(() => expect(screen.getByText("£10.0MM")).toBeInTheDocument());

    // Switch to October — funded, pipeline and forecast all update.
    fireEvent.change(select, { target: { value: "mi_2025_10" } });
    await waitFor(() => expect(screen.getByText("33")).toBeInTheDocument()); // funded loans
    // October forecast funded balance £4.8MM (4.2MM funded + 0.6MM weighted).
    await waitFor(() => expect(screen.getByText("£4.8MM")).toBeInTheDocument());
  });
});
