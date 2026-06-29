import { describe, expect, it, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import type { AgentClient } from "@/api";
import { ForecastExtrapolationPanel } from "./ForecastExtrapolationPanel";
import { mockForecastExtrapolation } from "@/data/mockForecastExtrapolation";

vi.mock("recharts", async () => {
  const actual = await vi.importActual<typeof import("recharts")>("recharts");
  return {
    ...actual,
    ResponsiveContainer: ({ children }: { children: React.ReactNode }) => (
      <div style={{ width: 600, height: 240 }}>{children}</div>
    ),
  };
});

function client(getForecastExtrapolation: AgentClient["getForecastExtrapolation"]): AgentClient {
  return {
    id: "test", mock: true,
    ask: vi.fn(), getSnapshots: vi.fn(), getSourcePortfolios: vi.fn(), getSnapshot: vi.fn(),
    getForecastSnapshot: vi.fn(), getFundedEvolution: vi.fn(),
    getPipelineEvolution: vi.fn(), getForecastEvolution: vi.fn(),
    getFunnelEvolution: vi.fn(), getRiskLimits: vi.fn(),
    getForecastExtrapolation,
  };
}

describe("ForecastExtrapolationPanel", () => {
  it("renders the scale-up curve, milestone table and assumptions for the run-rate model", async () => {
    render(<ForecastExtrapolationPanel
      client={client(async () => mockForecastExtrapolation("client_001"))}
      portfolioId="client_001/mi_2025_11" />);
    expect(await screen.findByText(/Scale-up forecast/)).toBeInTheDocument();
    // Default model = completion run-rate: curve + milestones + assumptions render.
    expect(screen.getByText(/Projected funded balance/)).toBeInTheDocument();
    expect(screen.getByTestId("milestone-table")).toBeInTheDocument();
    expect(screen.getByTestId("assumptions-panel")).toBeInTheDocument();
    // Milestone thresholds present.
    expect(screen.getByText("£100m")).toBeInTheDocument();
    expect(screen.getByText("£50m")).toBeInTheDocument();
    // Scenario bands labelled, not statistical CIs.
    expect(screen.getByText(/not statistical confidence intervals/)).toBeInTheDocument();
  });

  it("switches to the current weighted pipeline model and labels it clearly", async () => {
    render(<ForecastExtrapolationPanel
      client={client(async () => mockForecastExtrapolation("client_001"))}
      portfolioId="client_001" />);
    await screen.findByText(/Scale-up forecast/);
    fireEvent.click(screen.getByRole("tab", { name: "Current weighted pipeline" }));
    expect(await screen.findByTestId("weighted-pipeline-forecast")).toBeInTheDocument();
    expect(screen.getByText(/NOT the full scale-up forecast/)).toBeInTheDocument();
  });

  it("shows a controlled insufficient-history state", async () => {
    const insufficient = {
      ...mockForecastExtrapolation("client_001"),
      completionRunRateForecast: {
        model: "completion_run_rate" as const,
        available: false,
        status: "insufficient_data",
        caveat: "No completion history (need at least two funded runs).",
      },
    };
    render(<ForecastExtrapolationPanel client={client(async () => insufficient)}
      portfolioId="client_001" />);
    expect(await screen.findByTestId("forecast-insufficient")).toBeInTheDocument();
    expect(screen.getByText(/Insufficient history/)).toBeInTheDocument();
  });
});
