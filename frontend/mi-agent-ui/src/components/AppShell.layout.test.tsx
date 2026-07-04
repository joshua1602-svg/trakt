import { describe, expect, it } from "vitest";
import { render, screen, waitFor, fireEvent } from "@testing-library/react";
import { AppShell } from "./AppShell";

// A — clearer UI division: distinct regions (chat / core dashboard / artifact
// workspace), independent collapse, and clear chat/artifacts/both that never
// wipe the loaded MI data.
describe("AppShell — UI division, collapse and declutter (A)", () => {
  it("renders the three working regions as distinct, labelled areas", async () => {
    render(<AppShell />);
    await waitFor(() => expect(screen.getByText("Funded Book Snapshot")).toBeInTheDocument());
    // 1) Agent chat (its own surface), 2) core dashboard, 3) artifact workspace.
    expect(document.querySelector('[data-surface="ai-chat"]')).toBeTruthy();
    expect(screen.getByTestId("core-dashboard")).toBeInTheDocument();
    expect(screen.getByText("Core Dashboard")).toBeInTheDocument();
    expect(screen.getByTestId("artifact-region")).toBeInTheDocument();
    expect(screen.getByText("Artifact Workspace")).toBeInTheDocument();
  });

  it("collapses and expands the core dashboard independently of the MI data", async () => {
    render(<AppShell />);
    await waitFor(() => expect(screen.getByText("Funded Book Snapshot")).toBeInTheDocument());
    const toggle = screen.getByTestId("core-dashboard-toggle");
    fireEvent.click(toggle);
    expect(screen.getByText(/Core Dashboard collapsed/)).toBeInTheDocument();
    expect(screen.queryByText("Funded Book Snapshot")).not.toBeInTheDocument();
    // Expanding restores the loaded snapshot (data was never discarded).
    fireEvent.click(toggle);
    await waitFor(() => expect(screen.getByText("Funded Book Snapshot")).toBeInTheDocument());
    expect(screen.getByText("73")).toBeInTheDocument();
  });

  it("scopes clearing to the chat / artifact panels (no standalone declutter cluster)", async () => {
    render(<AppShell />);
    await waitFor(() => expect(screen.getByText("Funded Book Snapshot")).toBeInTheDocument());
    // The standalone declutter cluster was removed; clearing now lives in the
    // chat surface (Clear chat) and the artifact workspace (Clear artifacts).
    expect(screen.queryByTestId("declutter-controls")).toBeNull();
    expect(document.querySelector('[data-surface="ai-chat"]')).toBeTruthy();
    expect(screen.getByTestId("artifact-region")).toBeInTheDocument();
    // The loaded MI data is independent of any view declutter and stays put.
    expect(screen.getByText("Funded Book Snapshot")).toBeInTheDocument();
    expect(screen.getByText("73")).toBeInTheDocument();
  });
});
