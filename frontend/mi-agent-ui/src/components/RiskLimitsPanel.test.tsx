import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import type { AgentClient } from "@/api";
import { RiskLimitsPanel } from "./RiskLimitsPanel";
import { mockRiskLimits } from "@/data/mockRiskLimits";

function client(getRiskLimits: AgentClient["getRiskLimits"]): AgentClient {
  return {
    id: "test", mock: true,
    ask: vi.fn(), getSnapshots: vi.fn(), getSnapshot: vi.fn(),
    getForecastSnapshot: vi.fn(), getFundedEvolution: vi.fn(),
    getPipelineEvolution: vi.fn(), getForecastEvolution: vi.fn(),
    getFunnelEvolution: vi.fn(), getForecastExtrapolation: vi.fn(),
    getRiskLimits,
  };
}

describe("RiskLimitsPanel", () => {
  it("renders summary cards, geographic + broker concentration and status pills", async () => {
    render(<RiskLimitsPanel client={client(async () => mockRiskLimits("client_001"))}
      portfolioId="client_001/mi_2025_11" />);
    expect(await screen.findByText("Geographic concentration")).toBeInTheDocument();
    expect(screen.getByText("Broker / intermediary concentration")).toBeInTheDocument();
    // Summary cards.
    expect(screen.getByText("Tests passed")).toBeInTheDocument();
    expect(screen.getByText("Breaches")).toBeInTheDocument();
    expect(screen.getByText("Closest headroom")).toBeInTheDocument();
    // Region row + status pills.
    expect(screen.getByText("London")).toBeInTheDocument();
    expect(screen.getAllByText("Pass").length).toBeGreaterThan(0);
    expect(screen.getAllByText("Warn").length).toBeGreaterThan(0);
  });

  it("renders needs_review and unavailable controlled states", async () => {
    render(<RiskLimitsPanel client={client(async () => mockRiskLimits("client_001"))}
      portfolioId="client_001/mi_2025_11" />);
    // "Needs review" appears as both a summary card and a status pill.
    expect((await screen.findAllByText("Needs review")).length).toBeGreaterThan(0);
    expect(screen.getByText("Unavailable")).toBeInTheDocument();
    // Missing field surfaced for the unavailable test.
    expect(screen.getByText(/variable_rate_flag/)).toBeInTheDocument();
  });

  it("renders observations and lineage (limit source + source document)", async () => {
    render(<RiskLimitsPanel client={client(async () => mockRiskLimits("client_001"))}
      portfolioId="client_001/mi_2025_11" />);
    expect(await screen.findByText("Key observations")).toBeInTheDocument();
    expect(screen.getByText("Lineage")).toBeInTheDocument();
    expect(screen.getByText("schedule_8_concentration.txt")).toBeInTheDocument();
  });

  it("shows a controlled 'limits unavailable' state when no limits", async () => {
    const empty = {
      ...mockRiskLimits("client_001"),
      available: false,
      limitsStatus: "unavailable",
      limitsReason: "No Schedule 8 limits available — extraction required.",
      tests: [],
      testsByCategory: {},
      observations: [],
    };
    render(<RiskLimitsPanel client={client(async () => empty)} portfolioId="client_001" />);
    expect(await screen.findByText(/Limits unavailable/)).toBeInTheDocument();
  });
});
