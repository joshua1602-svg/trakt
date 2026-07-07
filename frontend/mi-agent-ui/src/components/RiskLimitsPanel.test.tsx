import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import type { AgentClient } from "@/api";
import { RiskLimitsPanel } from "./RiskLimitsPanel";
import { mockRiskLimits } from "@/data/mockRiskLimits";

function client(getRiskLimits: AgentClient["getRiskLimits"]): AgentClient {
  return {
    id: "test", mock: true,
    ask: vi.fn(), getSnapshots: vi.fn(), getSourcePortfolios: vi.fn(), getSnapshot: vi.fn(),
    getForecastSnapshot: vi.fn(), getFundedEvolution: vi.fn(),
    getPipelineEvolution: vi.fn(), getForecastEvolution: vi.fn(),
    getFunnelEvolution: vi.fn(), getForecastExtrapolation: vi.fn(), getCohortProgression: vi.fn(),
    getMe: vi.fn(), getDecks: vi.fn(), deckDownloadUrl: vi.fn(() => null), getCohorts: vi.fn(),
    getGeoExposure: vi.fn(),
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

  // B — source provenance banner: Schedule 8 vs fallback vs placeholder.
  it("shows the Schedule 8 document source without a warning badge", async () => {
    render(<RiskLimitsPanel client={client(async () => mockRiskLimits("client_001"))}
      portfolioId="client_001/mi_2025_11" />);
    const banner = await screen.findByTestId("risk-source-banner");
    expect(banner.textContent).toMatch(/Schedule 8 document/);
    // schedule_8_doc → not a fallback/placeholder, so no warning badge.
    expect(banner.textContent).not.toMatch(/Fallback|Placeholder/);
  });

  it("warns clearly when the limits come from a fallback config", async () => {
    const fb = {
      ...mockRiskLimits("client_001"),
      limitsSource: "fallback config",
      sourceType: "fallback_config" as const,
      extractionStatus: "success" as const,
      isPlaceholder: false,
    };
    render(<RiskLimitsPanel client={client(async () => fb)} portfolioId="client_001" />);
    const banner = await screen.findByTestId("risk-source-banner");
    expect(banner.textContent).toMatch(/fallback config/);
    expect(banner.textContent).toMatch(/Fallback/);
    expect(banner.textContent).toMatch(/not from the client's Schedule 8/i);
  });

  it("says 'Schedule 8 not found in docs folder' for a placeholder source", async () => {
    const ph = {
      ...mockRiskLimits("client_001"),
      available: false,
      limitsStatus: "unavailable",
      limitsSource: "placeholder / missing source",
      sourceType: "placeholder" as const,
      sourceFile: null,
      extractionStatus: "not_found" as const,
      isPlaceholder: true,
      limitsReason: "Schedule 8 not found in docs folder.",
      tests: [],
      testsByCategory: {},
      observations: [],
    };
    render(<RiskLimitsPanel client={client(async () => ph)} portfolioId="client_001" />);
    expect(await screen.findByText("Schedule 8 not found in docs folder")).toBeInTheDocument();
    const banner = screen.getByTestId("risk-source-banner");
    expect(banner.textContent).toMatch(/Placeholder/);
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
