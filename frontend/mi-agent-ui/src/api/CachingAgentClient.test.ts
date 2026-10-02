import { describe, expect, it, vi } from "vitest";
import type { AgentClient } from "./AgentClient";
import type { AgentRequest, AgentResponse } from "@/domain";
import { buildCacheKey, withCache } from "./CachingAgentClient";

function req(overrides: Partial<AgentRequest> = {}): AgentRequest {
  return {
    question: "Show balance by region",
    portfolio: { id: "client_001/mi_2025_11", name: "ERM", entity: "2025-11-30" },
    reporting: { asOf: "2025-11-30" },
    datasetContext: "funded",
    ...overrides,
  };
}

function res(ok = true): AgentResponse {
  return { ok, question: "q", intent: "concentration_risk", narrative: "n", assumptions: [], artifacts: [], warnings: [] };
}

function fakeClient(response: AgentResponse): { client: AgentClient; ask: ReturnType<typeof vi.fn> } {
  const ask = vi.fn(async () => response);
  const client: AgentClient = {
    id: "fake",
    mock: true,
    ask,
    getSnapshots: vi.fn(),
    getSourcePortfolios: vi.fn(),
    getPortfolioContext: vi.fn(async () => ({ available: false, client_id: null, default_context_id: "total", contexts: [], portfolios: [], portfolio_types: [], pipeline_portfolios: null })),
    getSnapshot: vi.fn(),
    getForecastSnapshot: vi.fn(),
    getFundedEvolution: vi.fn(),
    getPipelineEvolution: vi.fn(),
    getPipelineMovement: vi.fn(async () => ({ dataset: "pipeline_movement" as const, portfolioId: "p", available: false, reason: "no extracts", stages: [] })),
    getForecastEvolution: vi.fn(),
    getFunnelEvolution: vi.fn(),
    getRiskLimits: vi.fn(),
    getConcentrationTests: vi.fn(),
    getConcentrationDrillthrough: vi.fn(),
    getConcentrationHistory: vi.fn(),
    getConcentrationDrivers: vi.fn(),
    getForecastExtrapolation: vi.fn(),
    getCohortProgression: vi.fn(),
    getCohortVintages: () => Promise.resolve({ dataset: "cohort_formation", portfolioId: "p", cohortBasis: "origination_date", grain: "M", available: true, vintages: [] }) as never,
    getMe: vi.fn(),
    getDecks: vi.fn(),
    deckDownloadUrl: vi.fn(() => null),
    getCohorts: vi.fn(),
    getGeoExposure: vi.fn(),
  };
  return { client, ask };
}

describe("buildCacheKey", () => {
  it("differs by portfolio and by as-of snapshot", () => {
    const base = buildCacheKey(req());
    expect(buildCacheKey(req({ portfolio: { id: "client_002/mi_2025_11", name: "X", entity: "" } }))).not.toBe(base);
    expect(buildCacheKey(req({ reporting: { asOf: "2025-10-31" } }))).not.toBe(base);
  });
  it("is stable regardless of filter key order", () => {
    const a = buildCacheKey(req({ filters: { a: 1, b: 2 } }));
    const b = buildCacheKey(req({ filters: { b: 2, a: 1 } }));
    expect(a).toBe(b);
  });
});

describe("withCache", () => {
  it("serves a repeat query from cache (cacheHit) without re-calling", async () => {
    const { client, ask } = fakeClient(res(true));
    const cached = withCache(client);
    const first = await cached.ask(req());
    const second = await cached.ask(req());
    expect(ask).toHaveBeenCalledTimes(1);
    expect(first.cacheHit).toBe(false);
    expect(second.cacheHit).toBe(true);
  });

  it("does not cache failed responses", async () => {
    const { client, ask } = fakeClient(res(false));
    const cached = withCache(client);
    await cached.ask(req());
    await cached.ask(req());
    expect(ask).toHaveBeenCalledTimes(2);
  });

  it("does not return a stale result across a different portfolio/snapshot", async () => {
    const { client, ask } = fakeClient(res(true));
    const cached = withCache(client);
    await cached.ask(req());
    await cached.ask(req({ portfolio: { id: "client_002/mi_2025_11", name: "X", entity: "" } }));
    await cached.ask(req({ reporting: { asOf: "2025-10-31" } }));
    expect(ask).toHaveBeenCalledTimes(3);
  });
});
