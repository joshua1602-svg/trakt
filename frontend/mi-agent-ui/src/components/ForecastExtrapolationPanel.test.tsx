import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
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
    getFunnelEvolution: vi.fn(), getRiskLimits: vi.fn(), getCohortProgression: vi.fn(),
    getMe: vi.fn(), getDecks: vi.fn(), deckDownloadUrl: vi.fn(() => null), getCohorts: vi.fn(),
    getGeoExposure: vi.fn(),
    getForecastExtrapolation,
  };
}

describe("ForecastExtrapolationPanel", () => {
  it("renders the single scale-up run-rate view (curve, milestones, assumptions)", async () => {
    render(<ForecastExtrapolationPanel
      client={client(async () => mockForecastExtrapolation("client_001"))}
      portfolioId="client_001/mi_2025_11" />);
    expect(await screen.findByText(/Scale-up run-rate/)).toBeInTheDocument();
    // The completion run-rate view: curve + milestones + assumptions render.
    expect(screen.getByText(/Projected funded balance/)).toBeInTheDocument();
    expect(screen.getByTestId("milestone-table")).toBeInTheDocument();
    expect(screen.getByTestId("assumptions-panel")).toBeInTheDocument();
    expect(screen.getByText("£100m")).toBeInTheDocument();
    expect(screen.getByText("£50m")).toBeInTheDocument();
    expect(screen.getByText(/not statistical confidence intervals/)).toBeInTheDocument();
  });

  it("does not show a model selector or the withdrawn/duplicate models", async () => {
    render(<ForecastExtrapolationPanel
      client={client(async () => mockForecastExtrapolation("client_001"))}
      portfolioId="client_001" />);
    await screen.findByText(/Scale-up run-rate/);
    // The point-in-time bridge is View i (shown above), not repeated here; the
    // KFI model is withdrawn. No model tabs, no duplicate weighted-pipeline card.
    expect(screen.queryByRole("tab", { name: "Current weighted pipeline" })).toBeNull();
    expect(screen.queryByRole("tab", { name: "KFI run-rate × conversion" })).toBeNull();
    expect(screen.queryByTestId("weighted-pipeline-forecast")).toBeNull();
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
