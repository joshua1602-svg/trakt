import { describe, expect, it, vi } from "vitest";
import { withCache } from "./CachingAgentClient";
import type { AgentClient } from "./AgentClient";

/** A client whose GET methods count calls so we can prove caching + dedupe. */
function countingClient(): { client: AgentClient; counts: Record<string, number> } {
  const counts: Record<string, number> = {};
  const bump = (k: string) => { counts[k] = (counts[k] ?? 0) + 1; };
  const client = {
    id: "test", mock: true,
    ask: vi.fn(),
    getSnapshots: vi.fn(async () => { bump("snapshots"); return {} as any; }),
    getSourcePortfolios: vi.fn(async () => ({}) as any),
    getSnapshot: vi.fn(async (p: string) => { bump(`snapshot:${p}`); return { portfolioId: p } as any; }),
    getForecastSnapshot: vi.fn(async () => ({}) as any),
    getFundedEvolution: vi.fn(async () => ({}) as any),
    getPipelineEvolution: vi.fn(async () => ({}) as any),
    getForecastEvolution: vi.fn(async () => ({}) as any),
    getFunnelEvolution: vi.fn(async (p: string) => { bump(`funnel:${p}`); return { portfolioId: p } as any; }),
    getRiskLimits: vi.fn(async () => ({}) as any),
    getForecastExtrapolation: vi.fn(async () => ({}) as any),
    getMe: vi.fn(async () => ({ authenticated: false })),
    getDecks: vi.fn(async () => ({ available: false, latest: null, decks: [], client_id: "" }) as any),
    deckDownloadUrl: vi.fn(() => null),
    getCohorts: vi.fn(async () => ({}) as any),
  } as unknown as AgentClient;
  return { client, counts };
}

describe("CachingAgentClient GET resource cache (Task 3)", () => {
  it("serves a repeat GET from cache within staleTime (one underlying call)", async () => {
    const { client, counts } = countingClient();
    const cached = withCache(client);
    await cached.getSnapshot("client_001/mi_2025_11");
    await cached.getSnapshot("client_001/mi_2025_11");
    await cached.getSnapshot("client_001/mi_2025_11");
    expect(counts["snapshot:client_001/mi_2025_11"]).toBe(1);
  });

  it("keys by endpoint + portfolioId (a different portfolio refetches)", async () => {
    const { client, counts } = countingClient();
    const cached = withCache(client);
    await cached.getSnapshot("a/1");
    await cached.getSnapshot("b/2");
    await cached.getSnapshot("a/1");
    expect(counts["snapshot:a/1"]).toBe(1);
    expect(counts["snapshot:b/2"]).toBe(1);
  });

  it("dedupes concurrent in-flight GETs into a single underlying call", async () => {
    const { client, counts } = countingClient();
    const cached = withCache(client);
    await Promise.all([
      cached.getFunnelEvolution("x"),
      cached.getFunnelEvolution("x"),
      cached.getFunnelEvolution("x"),
    ]);
    expect(counts["funnel:x"]).toBe(1);
  });

  it("invalidate() clears the cache so the next read refetches", async () => {
    const { client, counts } = countingClient();
    const cached = withCache(client);
    await cached.getSnapshot("client_001/mi_2025_11");
    cached.invalidate();
    await cached.getSnapshot("client_001/mi_2025_11");
    expect(counts["snapshot:client_001/mi_2025_11"]).toBe(2);
  });

  it("expires entries after the staleTime window", async () => {
    const { client, counts } = countingClient();
    const cached = withCache(client, 0); // staleMs=0 → always stale after settle
    await cached.getSnapshot("p/1");
    await cached.getSnapshot("p/1");
    expect(counts["snapshot:p/1"]).toBe(2);
  });
});
