import { describe, expect, it } from "vitest";
import { render, screen, waitFor, fireEvent, within } from "@testing-library/react";
import { AppShell } from "./AppShell";

// AppShell (no VITE_AGENT_API_URL) uses the MockAgentClient, whose forecast
// snapshot mirrors the real funded spine + pipeline fixture pack. The workspace
// shows ONE active view at a time, selected via the Funded/Pipeline/Forecast toggle.
describe("AppShell — Funded / Pipeline / Forecast workspace", () => {
  it("defaults to the Funded view and does not stack pipeline/forecast sections", async () => {
    render(<AppShell />);
    await waitFor(() => expect(screen.getByText("Funded Book Snapshot")).toBeInTheDocument());
    expect(screen.getByText("Loans funded")).toBeInTheDocument();
    expect(screen.getByText("73")).toBeInTheDocument();
    // The toggle exists with all three views.
    const tablist = screen.getByRole("tablist", { name: /workspace view/i });
    expect(within(tablist).getByRole("tab", { name: /Funded/ })).toHaveAttribute("aria-selected", "true");
    // Pipeline / Forecast sections are NOT shown while Funded is active.
    expect(screen.queryByText("Pipeline Snapshot")).not.toBeInTheDocument();
    expect(screen.queryByText("Funded + Pipeline Forecast")).not.toBeInTheDocument();
  });

  it("switching to Pipeline shows the pipeline snapshot + watchlist (not the funded panel)", async () => {
    render(<AppShell />);
    await waitFor(() => expect(screen.getByText("Funded Book Snapshot")).toBeInTheDocument());
    fireEvent.click(screen.getByRole("tab", { name: /Pipeline/ }));
    await waitFor(() => expect(screen.getByText("Pipeline Snapshot")).toBeInTheDocument());
    expect(screen.getByText("Pipeline Watchlist")).toBeInTheDocument();
    expect(screen.queryByText("Funded Book Snapshot")).not.toBeInTheDocument();
  });

  it("switching to Forecast shows funded + weighted pipeline = forecast", async () => {
    render(<AppShell />);
    await waitFor(() => expect(screen.getByText("Funded Book Snapshot")).toBeInTheDocument());
    fireEvent.click(screen.getByRole("tab", { name: /Forecast/ }));
    await waitFor(() => expect(screen.getByText("Funded + Pipeline Forecast")).toBeInTheDocument());
    // November: £8.9MM funded + £1.1MM weighted = £10.0MM forecast.
    expect(screen.getByText("£10.0MM")).toBeInTheDocument();
    // Derived forecast-by-region breakdown renders.
    expect(screen.getByText("Forecast balance by region")).toBeInTheDocument();
  });

  it("reporting-date selector refreshes the active view's data", async () => {
    render(<AppShell />);
    const select = await screen.findByRole("combobox");
    fireEvent.click(screen.getByRole("tab", { name: /Forecast/ }));
    await waitFor(() => expect(screen.getByText("£10.0MM")).toBeInTheDocument());
    // Switch run to October — forecast refreshes to £4.8MM.
    fireEvent.change(select, { target: { value: "mi_2025_10" } });
    await waitFor(() => expect(screen.getByText("£4.8MM")).toBeInTheDocument());
  });
});
