import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import type { AgentClient } from "@/api/AgentClient";
import type { GeoExposure } from "@/domain/geo";
import { mockGeoExposure } from "@/data/mockGeoExposure";
import { GeographyPanel } from "./GeographyPanel";

function client(geo: GeoExposure): AgentClient {
  return {
    id: "test", mock: true,
    ask: vi.fn(), getSnapshots: vi.fn(), getSourcePortfolios: vi.fn(), getSnapshot: vi.fn(),
    getForecastSnapshot: vi.fn(), getFundedEvolution: vi.fn(), getPipelineEvolution: vi.fn(),
    getForecastEvolution: vi.fn(), getFunnelEvolution: vi.fn(), getRiskLimits: vi.fn(),
    getForecastExtrapolation: vi.fn(), getMe: vi.fn(), getDecks: vi.fn(),
    deckDownloadUrl: vi.fn(() => null), getCohorts: vi.fn(), getCohortProgression: vi.fn(),
    getGeoExposure: vi.fn(async () => geo),
  } as unknown as AgentClient;
}

describe("GeographyPanel", () => {
  it("renders the choropleth, KPIs and ranked areas from the exposure data", async () => {
    const c = client(mockGeoExposure("client_001/mi_2025_11"));
    render(<GeographyPanel client={c} portfolioId="client_001/mi_2025_11" />);

    expect(await screen.findByTestId("geography-view")).toBeInTheDocument();
    // The SVG choropleth renders one path per ITL3 area.
    const svg = screen.getByTestId("uk-choropleth").querySelector("svg")!;
    expect(svg.querySelectorAll("path").length).toBeGreaterThan(150);
    // Bristol (the forced top area) leads the ranked list + the top-area KPI.
    const rank = screen.getByTestId("geo-rank");
    expect(rank.textContent).toMatch(/Bristol/);
    expect(c.getGeoExposure).toHaveBeenCalledWith("client_001/mi_2025_11");
  });

  it("shows the unavailable state when the tape has no geography", async () => {
    const geo: GeoExposure = {
      dataset: "geo_itl3", portfolioId: "client_001/mi_2025_11",
      available: false, reason: "no ITL3 field and no property postcode on the tape", areas: [],
    };
    render(<GeographyPanel client={client(geo)} portfolioId="client_001/mi_2025_11" />);
    const banner = await screen.findByTestId("geo-unavailable");
    expect(banner.textContent).toMatch(/no property postcode/i);
    expect(screen.queryByTestId("uk-choropleth")).toBeNull();
  });
});
