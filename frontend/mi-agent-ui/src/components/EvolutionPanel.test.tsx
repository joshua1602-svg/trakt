import { describe, expect, it, vi } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import type { AgentClient } from "@/api";
import { EvolutionPanel } from "./EvolutionPanel";
import {
  mockFundedEvolution,
  mockPipelineEvolution,
  mockForecastEvolution,
} from "@/data/mockEvolution";
import { mockFunnelEvolution } from "@/data/mockFunnel";
import { mockCohorts } from "@/data/mockCohorts";
import { mockRiskLimits } from "@/data/mockRiskLimits";
import { mockForecastExtrapolation } from "@/data/mockForecastExtrapolation";

// recharts needs a non-zero layout in jsdom; stub ResponsiveContainer dimensions.
vi.mock("recharts", async () => {
  const actual = await vi.importActual<typeof import("recharts")>("recharts");
  return {
    ...actual,
    ResponsiveContainer: ({ children }: { children: React.ReactNode }) => (
      <div style={{ width: 600, height: 200 }}>{children}</div>
    ),
  };
});

function client(over: Partial<AgentClient> = {}): AgentClient {
  return {
    id: "test",
    mock: true,
    ask: vi.fn(),
    getSnapshots: vi.fn(),
    getSourcePortfolios: vi.fn(),
    getSnapshot: vi.fn(),
    getForecastSnapshot: vi.fn(),
    getFundedEvolution: vi.fn(async () => mockFundedEvolution("client_001")),
    getPipelineEvolution: vi.fn(async () => mockPipelineEvolution("client_001")),
    getForecastEvolution: vi.fn(async () => mockForecastEvolution("client_001")),
    getFunnelEvolution: vi.fn(async () => mockFunnelEvolution("client_001")),
    getRiskLimits: vi.fn(async () => mockRiskLimits("client_001")),
    getForecastExtrapolation: vi.fn(async () => mockForecastExtrapolation("client_001")),
    getMe: vi.fn(async () => ({ authenticated: true, isOperator: true })),
    getDecks: vi.fn(async () => ({ available: false, latest: null, decks: [], client_id: "client_001" })),
    deckDownloadUrl: vi.fn(() => null),
    getCohorts: vi.fn(async () => mockCohorts("client_001")),
    ...over,
  };
}

describe("EvolutionPanel", () => {
  it("renders the funded evolution charts with source/coverage footers", async () => {
    render(<EvolutionPanel client={client()} portfolioId="client_001/mi_2025_11" />);
    expect(await screen.findByText("Funded balance by month")).toBeInTheDocument();
    expect(screen.getByText("Funded loan count by month")).toBeInTheDocument();
    expect(screen.getByText("WA LTV by month")).toBeInTheDocument();
    // Every chart carries a per-period reconciliation/coverage footer.
    expect(screen.getAllByText(/Coverage: 100%/).length).toBeGreaterThan(0);
  });

  it("switches to the pipeline series (amount + by stage over time)", async () => {
    const c = client();
    render(<EvolutionPanel client={c} portfolioId="client_001" />);
    await screen.findByText("Funded balance by month");
    fireEvent.click(screen.getByRole("tab", { name: "Pipeline" }));
    expect(await screen.findByText("Pipeline amount by week")).toBeInTheDocument();
    expect(screen.getByText("Pipeline by stage over time")).toBeInTheDocument();
    expect(screen.getByText("Weighted expected funded by week")).toBeInTheDocument();
    expect(c.getPipelineEvolution).toHaveBeenCalled();
  });

  it("renders the weekly origination funnel (KFI/Application/Offer/Completion)", async () => {
    const c = client();
    render(<EvolutionPanel client={c} portfolioId="client_001" />);
    await screen.findByText("Funded balance by month");
    fireEvent.click(screen.getByRole("tab", { name: "Origination" }));
    expect(await screen.findByTestId("origination-funnel")).toBeInTheDocument();
    expect(screen.getByText("KFIs")).toBeInTheDocument();
    expect(screen.getByText("Applications")).toBeInTheDocument();
    expect(screen.getByText("Offers")).toBeInTheDocument();
    expect(screen.getByText("Completions")).toBeInTheDocument();
    // Weekly-flow summary present per stage (5-week average of flow + latest flow).
    expect(screen.getAllByText("5-wk avg flow").length).toBeGreaterThan(0);
    expect(screen.getAllByText("Latest weekly flow").length).toBeGreaterThan(0);
    expect(c.getFunnelEvolution).toHaveBeenCalled();
  });

  it("renders the Cohorts sub-tab with computed vintage metrics (Task 2)", async () => {
    const c = client();
    render(<EvolutionPanel client={c} portfolioId="client_001/mi_2025_11" />);
    await screen.findByText("Funded balance by month");
    fireEvent.click(screen.getByRole("tab", { name: "Cohorts" }));
    expect(await screen.findByTestId("cohorts-view")).toBeInTheDocument();
    // The vintage table surfaces computed aggregates.
    const table = screen.getByTestId("cohorts-table");
    expect(table.textContent).toContain("2021");
    expect(table.textContent).toMatch(/WA LTV/);
    // No fabricated redemption/performance curves are shown.
    expect(screen.queryByText(/redemption curve/i)).toBeNull();
    expect(c.getCohorts).toHaveBeenCalled();
  });

  it("shows an honest empty state when cohort data is unavailable", async () => {
    const c = client();
    c.getCohorts = vi.fn(async () => ({
      dataset: "cohorts", portfolioId: "client_001/mi_2025_11", available: false,
      reason: "no origination date / vintage on the funded tape", cohorts: [],
    })) as AgentClient["getCohorts"];
    render(<EvolutionPanel client={c} portfolioId="client_001/mi_2025_11" />);
    await screen.findByText("Funded balance by month");
    fireEvent.click(screen.getByRole("tab", { name: "Cohorts" }));
    expect(await screen.findByTestId("cohorts-unavailable")).toBeInTheDocument();
  });

  // D — Forecast Evolution is distinct from the main Forecast tab: it shows the
  // HISTORY of the forecast across runs (+ actual funded vs the prior forecast).
  it("renders the Forecast Evolution sub-tab with distinct label, subtitle and lineage", async () => {
    const c = client();
    render(<EvolutionPanel client={c} portfolioId="client_001/mi_2025_11" />);
    await screen.findByText("Funded balance by month");
    // Sub-tab reads "Forecast Evolution", not a bare "forecast".
    fireEvent.click(screen.getByRole("tab", { name: "Forecast Evolution" }));
    expect(await screen.findByTestId("forecast-evolution")).toBeInTheDocument();
    // Distinct subtitle clarifies this is the forecast HISTORY, not the projection.
    expect(screen.getByTestId("evo-subtitle").textContent).toMatch(/historical movement/i);
    expect(screen.getByText("Forecast funded balance by reporting run")).toBeInTheDocument();
    // Actual-vs-prior-forecast chart present (mock has >1 run) + lineage caption.
    expect(screen.getByText("Actual funded vs prior-run forecast")).toBeInTheDocument();
    expect(screen.getByTestId("forecast-evolution-lineage").textContent).toMatch(/Forecast basis/i);
    expect(c.getForecastEvolution).toHaveBeenCalled();
  });

  it("flags insufficient forecast history when only one run is available", async () => {
    const single = client({
      getForecastEvolution: vi.fn(async () => ({
        ...mockForecastEvolution("client_001"),
        periods: [mockForecastEvolution("client_001").periods[0]],
        singlePeriod: true,
      })),
    });
    render(<EvolutionPanel client={single} portfolioId="client_001/mi_2025_11" />);
    await screen.findByText("Funded balance by month");
    fireEvent.click(screen.getByRole("tab", { name: "Forecast Evolution" }));
    expect(await screen.findByTestId("forecast-evolution-insufficient")).toBeInTheDocument();
    // With a single run there is no prior forecast to compare against.
    expect(screen.queryByText("Actual funded vs prior-run forecast")).toBeNull();
  });

  it("handles a single-period series with a clear notice", async () => {
    const single = client({
      getFundedEvolution: vi.fn(async () => ({
        ...mockFundedEvolution("client_001"),
        periods: [mockFundedEvolution("client_001").periods[0]],
        singlePeriod: true,
      })),
    });
    render(<EvolutionPanel client={single} portfolioId="client_001" />);
    expect(await screen.findByText(/needs at least two runs/i)).toBeInTheDocument();
  });

  it("renders an empty state without crashing when no data", async () => {
    const empty = client({
      getFundedEvolution: vi.fn(async () => ({
        ...mockFundedEvolution("client_001"),
        periods: [],
        singlePeriod: true,
      })),
    });
    render(<EvolutionPanel client={empty} portfolioId="client_001" />);
    await waitFor(() =>
      expect(screen.getAllByText(/No periods available/).length).toBeGreaterThan(0));
  });
});
