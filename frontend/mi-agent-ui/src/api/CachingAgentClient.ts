/**
 * CachingAgentClient — a transparent, additive decorator over any AgentClient.
 *
 * Two caches, both session-scoped and centralised here so the policy lives in one
 * place (Task 3):
 *
 *  1. `ask` responses — keyed by portfolio + normalised question + filters + as-of
 *     + dataset view, so a repeat query is instant while a different scope never
 *     returns a stale result. Only `ok` responses are cached.
 *
 *  2. GET resources (snapshots, forecast, evolution, funnel, risk-limits, decks,
 *     cohorts, identity) — keyed by endpoint + portfolioId, held for a staleTime
 *     so switching tabs reuses cached data instead of refetching/remounting every
 *     chart. In-flight promises are shared (dedupe), and a rejected fetch is
 *     evicted so it retries. `invalidate()` clears everything for a manual refresh.
 *
 * If anything about keying fails, calls fall through to the underlying client.
 */

import type { AgentClient } from "./AgentClient";
import type { AgentRequest, AgentResponse } from "@/domain";

/** Resource staleTime — within this window a repeat GET reuses the cache. */
export const RESOURCE_STALE_MS = 7 * 60 * 1000; // 7 minutes (within the 5–10 min guidance)

/** A cached GET resource. `at` is when the value resolved (for staleness). */
interface ResourceEntry {
  promise: Promise<unknown>;
  at: number;
  settled: boolean;
}

export interface CachingAgentClient extends AgentClient {
  /** Drop all cached responses so the next reads fetch fresh (manual refresh). */
  invalidate(): void;
}

/** Stable, order-independent serialisation of a filters object. */
function stableFilters(filters?: Record<string, unknown>): string {
  if (!filters) return "";
  const keys = Object.keys(filters).sort();
  return keys.map((k) => `${k}=${JSON.stringify(filters[k])}`).join("&");
}

/** Build the cache key for a request (portfolio + snapshot scoped). */
export function buildCacheKey(req: AgentRequest): string {
  return [
    req.portfolio?.id ?? "",
    req.reporting?.asOf ?? "",
    req.datasetContext ?? "",
    req.sourceLens ?? "",
    req.question.trim().toLowerCase().replace(/\s+/g, " "),
    stableFilters(req.filters),
    req.options?.topN ?? "",
  ].join("|");
}

export function withCache(
  client: AgentClient,
  staleMs: number = RESOURCE_STALE_MS,
): CachingAgentClient {
  const askCache = new Map<string, AgentResponse>();
  const resourceCache = new Map<string, ResourceEntry>();
  const now = () => (typeof performance !== "undefined" ? performance.now() : Date.now());

  /** Memoise a GET by (endpoint + args) with a staleTime + in-flight dedupe. */
  function resource<T>(key: string, load: () => Promise<T>): Promise<T> {
    const hit = resourceCache.get(key);
    if (hit && (!hit.settled || now() - hit.at < staleMs)) {
      return hit.promise as Promise<T>;
    }
    const entry: ResourceEntry = { promise: Promise.resolve(), at: now(), settled: false };
    entry.promise = load().then(
      (value) => {
        entry.at = now();
        entry.settled = true;
        return value;
      },
      (err) => {
        // Evict a failed fetch so a retry re-runs instead of caching the failure.
        if (resourceCache.get(key) === entry) resourceCache.delete(key);
        throw err;
      },
    );
    resourceCache.set(key, entry);
    return entry.promise as Promise<T>;
  }

  return {
    id: client.id,
    mock: client.mock,

    invalidate() {
      askCache.clear();
      resourceCache.clear();
    },

    async ask(request: AgentRequest, signal?: AbortSignal): Promise<AgentResponse> {
      let key: string | undefined;
      try {
        key = buildCacheKey(request);
      } catch {
        key = undefined; // keying failed → behave like the underlying client
      }
      if (key) {
        const hit = askCache.get(key);
        if (hit) return { ...hit, cacheHit: true };
      }
      const res = await client.ask(request, signal);
      if (key && res.ok) {
        const stored = { ...res, cacheHit: false };
        askCache.set(key, stored);
        return stored;
      }
      return res;
    },

    // Discovery is stable for the session — cache without a portfolio scope.
    getSnapshots: (signal) => resource("snapshots", () => client.getSnapshots(signal)),
    getSourcePortfolios: (signal) =>
      resource("sourcePortfolios", () => client.getSourcePortfolios(signal)),
    getMe: (signal) => resource("me", () => client.getMe(signal)),

    // Portfolio-scoped resources — keyed by endpoint + portfolioId (which encodes
    // client + run/reporting date; the pipeline extract is always the latest).
    getSnapshot: (portfolioId, signal) =>
      resource(`snapshot|${portfolioId}`, () => client.getSnapshot(portfolioId, signal)),
    getForecastSnapshot: (portfolioId, signal) =>
      resource(`forecastSnapshot|${portfolioId}`, () => client.getForecastSnapshot(portfolioId, signal)),
    getFundedEvolution: (portfolioId, signal) =>
      resource(`fundedEvolution|${portfolioId}`, () => client.getFundedEvolution(portfolioId, signal)),
    getPipelineEvolution: (portfolioId, signal) =>
      resource(`pipelineEvolution|${portfolioId}`, () => client.getPipelineEvolution(portfolioId, signal)),
    getForecastEvolution: (portfolioId, signal) =>
      resource(`forecastEvolution|${portfolioId}`, () => client.getForecastEvolution(portfolioId, signal)),
    getFunnelEvolution: (portfolioId, signal) =>
      resource(`funnelEvolution|${portfolioId}`, () => client.getFunnelEvolution(portfolioId, signal)),
    getRiskLimits: (portfolioId, signal) =>
      resource(`riskLimits|${portfolioId}`, () => client.getRiskLimits(portfolioId, signal)),
    getForecastExtrapolation: (portfolioId, signal) =>
      resource(`forecastExtrapolation|${portfolioId}`, () => client.getForecastExtrapolation(portfolioId, signal)),
    getDecks: (portfolioId, signal) =>
      resource(`decks|${portfolioId}`, () => client.getDecks(portfolioId, signal)),
    getCohorts: (portfolioId, grain, signal) =>
      resource(`cohorts|${portfolioId}|${grain ?? "Y"}`,
        () => client.getCohorts(portfolioId, grain, signal)),
    getCohortProgression: (portfolioId, query, signal) =>
      resource(
        `cohortProg|${portfolioId}|${query?.lens ?? "total"}|${query?.vintage ?? ""}|${query?.grain ?? "Y"}`,
        () => client.getCohortProgression(portfolioId, query, signal)),
    getGeoExposure: (portfolioId, signal) =>
      resource(`geoExposure|${portfolioId}`, () => client.getGeoExposure(portfolioId, signal)),
    deckDownloadUrl: (portfolioId, period) => client.deckDownloadUrl(portfolioId, period),
  };
}
