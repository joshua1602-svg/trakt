import { afterEach, describe, expect, it, vi } from "vitest";
import { render, screen, waitFor, fireEvent, within } from "@testing-library/react";
import { AppShell } from "./AppShell";
import { MockAgentClient } from "@/api/MockAgentClient";
import type { PortfolioCapability, PortfolioContextIndex } from "@/domain";

/** A client whose portfolio-context genuinely has something to disclose (a
 *  pipeline-attribution note), for every capability the shell reads — so a
 *  test that asserts the scope banner is ABSENT is actually proving the
 *  suppression, not just that the mock never had anything to show. */
/** AppShell builds its own client internally (createAgentClient()), which
 *  resolves to MockAgentClient with no injected VITE_AGENT_API_URL — so a
 *  prototype spy, not a prop, is how a test gives it a governed context with
 *  something to disclose. Restored in afterEach below. */
function stubScopeDisclosure(): void {
  const capability = (id: string, detail: string): PortfolioCapability => ({
    capability: id as PortfolioCapability["capability"],
    enabled: true, reason_code: null, detail,
    contributing_portfolios: ["direct_001"], excluded_portfolios: ["acquired_001"],
    partial: true,
  });
  const detail = "Pipeline is originating-only for this scope; acquired books do not contribute.";
  const index: PortfolioContextIndex = {
    available: true, client_id: "client_001", default_context_id: "total",
    contexts: [{
      context_id: "total", context_kind: "total", label: "Total", parent_id: null,
      depth: 0, portfolio_ids: ["direct_001", "acquired_001"], portfolio_types: [],
      capabilities: {
        funded: capability("funded", detail),
        pipeline: capability("pipeline", detail),
        consolidated_forecast: capability("consolidated_forecast", detail),
        risk: capability("risk", detail),
      } as unknown as PortfolioContextIndex["contexts"][number]["capabilities"],
    }],
    portfolios: [], portfolio_types: [], pipeline_portfolios: null,
  };
  vi.spyOn(MockAgentClient.prototype, "getPortfolioContext").mockResolvedValue(index);
}

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

/**
 * Stage Movement is its own Pipeline sub-tab.
 *
 * It used to sit at the bottom of Pipeline → Evolution, below four stock
 * charts. Evolution answers "where are cases, over time"; Stage Movement
 * answers "what happened to them" — a different question, not another trend
 * line, so it gets a sibling tab rather than a fifth card.
 *
 * These pin the navigation only. The component and its governed payload are
 * unchanged, and are covered by StageTransitionPanel.test.tsx.
 */
describe("AppShell — Pipeline → Stage Movement sub-tab", () => {
  function pipelineSubtabs() {
    return screen.getByRole("tablist", { name: /pipeline sub-view/i });
  }

  async function openPipeline() {
    render(<AppShell />);
    await waitFor(() => expect(screen.getByText("Funded Book Snapshot")).toBeInTheDocument());
    fireEvent.click(screen.getByRole("tab", { name: /Pipeline/ }));
    await waitFor(() => expect(screen.getByText("Pipeline Snapshot")).toBeInTheDocument());
  }

  it("is a sibling of the existing Pipeline sub-tabs, in order", async () => {
    await openPipeline();
    expect(within(pipelineSubtabs()).getAllByRole("tab").map((t) => t.textContent))
      .toEqual(["Stratifications", "Evolution", "Stage Movement"]);
  });

  it("is not the default — Stratifications still opens first", async () => {
    await openPipeline();
    expect(within(pipelineSubtabs()).getByRole("tab", { name: "Stage Movement" }))
      .toHaveAttribute("aria-selected", "false");
    expect(screen.queryByTestId("pipeline-movement-pane")).not.toBeInTheDocument();
  });

  it("renders the stage-movement pane when selected", async () => {
    await openPipeline();
    fireEvent.click(within(pipelineSubtabs()).getByRole("tab", { name: "Stage Movement" }));
    await waitFor(() =>
      expect(screen.getByTestId("pipeline-movement-pane")).toBeInTheDocument());
    expect(within(pipelineSubtabs()).getByRole("tab", { name: "Stage Movement" }))
      .toHaveAttribute("aria-selected", "true");
  });

  it("mounts the real panel inside the pane when the layer is enabled", async () => {
    // With the flag off (the default everywhere else here) the panel renders
    // null, so the pane alone would not prove the wiring. This is the assertion
    // that the tab actually reaches StageTransitionPanel.
    vi.stubEnv("VITE_MI_ENHANCED_HOVERS", "true");
    await openPipeline();
    fireEvent.click(within(pipelineSubtabs()).getByRole("tab", { name: "Stage Movement" }));
    const pane = await screen.findByTestId("pipeline-movement-pane");
    // The demo client has no governed weekly pair, so the panel shows its
    // controlled unavailable state — which is still the panel.
    await waitFor(() => expect(
      within(pane).getByTestId("stage-transitions-unavailable")).toBeInTheDocument());
    expect(pane.textContent).toContain("Pipeline stage movement");
  });

  it("no longer appears under Evolution, so it exists exactly once", async () => {
    await openPipeline();
    // Evolution: the stock series, and NOT the movement pane.
    fireEvent.click(within(pipelineSubtabs()).getByRole("tab", { name: "Evolution" }));
    await waitFor(() =>
      expect(screen.getByTestId("pipeline-evo-pane")).toBeInTheDocument());
    expect(within(screen.getByTestId("pipeline-evo-pane"))
      .queryByTestId("stage-transitions")).toBeNull();
    expect(within(screen.getByTestId("pipeline-evo-pane"))
      .queryByTestId("stage-transitions-unavailable")).toBeNull();

    // And after visiting Stage Movement — both panes stay mounted (KeepMounted),
    // so a duplicate would be visible here if one existed.
    fireEvent.click(within(pipelineSubtabs()).getByRole("tab", { name: "Stage Movement" }));
    await waitFor(() =>
      expect(screen.getByTestId("pipeline-movement-pane")).toBeInTheDocument());
    expect(screen.queryAllByTestId("pipeline-movement-pane")).toHaveLength(1);
    expect(within(screen.getByTestId("pipeline-evo-pane"))
      .queryByTestId("stage-transitions-unavailable")).toBeNull();
  });
});

describe("AppShell — the portfolio scope banner is not shown on Pipeline or Forecast", () => {
  afterEach(() => vi.restoreAllMocks());

  it("shows the governed scope disclosure on Funded", async () => {
    stubScopeDisclosure();
    render(<AppShell />);
    await waitFor(() => expect(screen.getByText("Funded Book Snapshot")).toBeInTheDocument());
    expect(screen.getByTestId("portfolio-scope-banner")).toBeInTheDocument();
  });

  it("does not show it on Pipeline", async () => {
    stubScopeDisclosure();
    render(<AppShell />);
    await waitFor(() => expect(screen.getByText("Funded Book Snapshot")).toBeInTheDocument());
    fireEvent.click(screen.getByRole("tab", { name: /Pipeline/ }));
    await waitFor(() => expect(screen.getByText("Pipeline Snapshot")).toBeInTheDocument());
    expect(screen.queryByTestId("portfolio-scope-banner")).toBeNull();
  });

  it("does not show it on Forecast", async () => {
    stubScopeDisclosure();
    render(<AppShell />);
    await waitFor(() => expect(screen.getByText("Funded Book Snapshot")).toBeInTheDocument());
    fireEvent.click(screen.getByRole("tab", { name: /Forecast/ }));
    await waitFor(() => expect(screen.getByText("Funded + Pipeline Forecast")).toBeInTheDocument());
    expect(screen.queryByTestId("portfolio-scope-banner")).toBeNull();
  });

  it("shows it again on switching back to Funded", async () => {
    stubScopeDisclosure();
    render(<AppShell />);
    await waitFor(() => expect(screen.getByText("Funded Book Snapshot")).toBeInTheDocument());
    fireEvent.click(screen.getByRole("tab", { name: /Pipeline/ }));
    await waitFor(() => expect(screen.getByText("Pipeline Snapshot")).toBeInTheDocument());
    fireEvent.click(screen.getByRole("tab", { name: /Funded/ }));
    await waitFor(() => expect(screen.getByTestId("portfolio-scope-banner")).toBeInTheDocument());
  });
});

afterEach(() => vi.unstubAllEnvs());
