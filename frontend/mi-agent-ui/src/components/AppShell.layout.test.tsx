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
    expect(screen.getByText("Core dashboard")).toBeInTheDocument();
    expect(screen.getByTestId("artifact-region")).toBeInTheDocument();
    expect(screen.getByText("Artifact Workspace")).toBeInTheDocument();
  });

  it("collapses and expands the core dashboard independently of the MI data", async () => {
    render(<AppShell />);
    await waitFor(() => expect(screen.getByText("Funded Book Snapshot")).toBeInTheDocument());
    const toggle = screen.getByTestId("core-dashboard-toggle");
    fireEvent.click(toggle);
    expect(screen.getByText(/Core dashboard collapsed/)).toBeInTheDocument();
    expect(screen.queryByText("Funded Book Snapshot")).not.toBeInTheDocument();
    // Expanding restores the loaded snapshot (data was never discarded).
    fireEvent.click(toggle);
    await waitFor(() => expect(screen.getByText("Funded Book Snapshot")).toBeInTheDocument());
    expect(screen.getByText("73")).toBeInTheDocument();
  });

  it("exposes clear chat / artifacts / both, and clearing keeps the MI data", async () => {
    render(<AppShell />);
    await waitFor(() => expect(screen.getByText("Funded Book Snapshot")).toBeInTheDocument());
    const group = screen.getByTestId("declutter-controls");
    expect(group).toBeInTheDocument();
    // "Clear both" wipes chat + artifacts but the funded snapshot stays loaded.
    fireEvent.click(screen.getByRole("button", { name: "Clear both" }));
    expect(screen.getByText("Funded Book Snapshot")).toBeInTheDocument();
    expect(screen.getByText("73")).toBeInTheDocument();
    // Individual clears are present too.
    expect(screen.getByRole("button", { name: /Clear chat/ })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Clear artifacts/ })).toBeInTheDocument();
  });
});
