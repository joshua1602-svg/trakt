import { describe, expect, it, vi } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import type { AgentClient } from "@/api";
import { EvolutionPanel } from "./EvolutionPanel";
import {
  mockFundedEvolution,
  mockPipelineEvolution,
  mockForecastEvolution,
} from "@/data/mockEvolution";

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
    getSnapshot: vi.fn(),
    getForecastSnapshot: vi.fn(),
    getFundedEvolution: vi.fn(async () => mockFundedEvolution("client_001")),
    getPipelineEvolution: vi.fn(async () => mockPipelineEvolution("client_001")),
    getForecastEvolution: vi.fn(async () => mockForecastEvolution("client_001")),
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
    fireEvent.click(screen.getByRole("tab", { name: "pipeline" }));
    expect(await screen.findByText("Pipeline amount by week/month")).toBeInTheDocument();
    expect(screen.getByText("Pipeline by stage over time")).toBeInTheDocument();
    expect(screen.getByText("Weighted expected funded by month")).toBeInTheDocument();
    expect(c.getPipelineEvolution).toHaveBeenCalled();
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
