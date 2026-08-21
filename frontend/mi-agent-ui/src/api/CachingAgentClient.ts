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

/** Resource staleTime — within this window a repeat GET reuses the cache.
 * Snapshot/forecast payloads are per reporting run and effectively immutable
 * once computed, and backend snapshot computes can take tens of seconds, so a
 * generous window makes client/scope switching instant; the header's manual
 * Refresh (invalidate()) always forces fresh reads. */
export const RESOURCE_STALE_MS = 30 * 60 * 1000; // 30 minutes

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
    getPortfolioContext: (signal) =>
      resource("portfolioContext", () => client.getPortfolioContext(signal)),
    getMe: (signal) => resource("me", () => client.getMe(signal)),

    // Portfolio-scoped resources — keyed by endpoint + portfolioId (which encodes
    // client + run/reporting date; the pipeline extract is always the latest).
    getSnapshot: (portfolioId, portfolioContext, signal) =>
      resource(`snapshot|${portfolioId}|${portfolioContext ?? ""}`,
        () => client.getSnapshot(portfolioId, portfolioContext, signal)),
    getForecastSnapshot: (portfolioId, portfolioContext, signal) =>
      resource(`forecastSnapshot|${portfolioId}|${portfolioContext ?? ""}`,
        () => client.getForecastSnapshot(portfolioId, portfolioContext, signal)),
    getFundedEvolution: (portfolioId, portfolioContext, signal) =>
      resource(`fundedEvolution|${portfolioId}|${portfolioContext ?? ""}`,
        () => client.getFundedEvolution(portfolioId, portfolioContext, signal)),
    getPipelineEvolution: (portfolioId, portfolioContext, signal) =>
      resource(`pipelineEvolution|${portfolioId}|${portfolioContext ?? ""}`,
        () => client.getPipelineEvolution(portfolioId, portfolioContext, signal)),
    getForecastEvolution: (portfolioId, portfolioContext, signal) =>
      resource(`forecastEvolution|${portfolioId}|${portfolioContext ?? ""}`,
        () => client.getForecastEvolution(portfolioId, portfolioContext, signal)),
    getFunnelEvolution: (portfolioId, portfolioContext, signal) =>
      resource(`funnelEvolution|${portfolioId}|${portfolioContext ?? ""}`,
        () => client.getFunnelEvolution(portfolioId, portfolioContext, signal)),
    // Movement detail is keyed by the WEEK as well as the scope, so hovering
    // back and forth along a series fetches each point at most once per session
    // and re-hovering the same point never fetches again. In-flight requests are
    // shared by the same mechanism, so a pointer crossing several points cannot
    // stack duplicate calls for one of them.
    // One request per portfolio + scope + week, reused for the session.
    getWeeklyBrief: (portfolioId, portfolioContext, asOf, signal) =>
      resource(`weeklyBrief|${portfolioId}|${portfolioContext ?? ""}|${asOf ?? "latest"}`,
        () => {
          const fetchBrief = client.getWeeklyBrief?.bind(client);
          return fetchBrief
            ? fetchBrief(portfolioId, portfolioContext, asOf, signal)
            : Promise.reject(new Error("weekly brief is not supported"));
        }),
    getMovementDetail: (portfolioId, detailType, asOf, portfolioContext, signal) =>
      resource(
        `movementDetail|${portfolioId}|${detailType}|${asOf ?? "latest"}|${portfolioContext ?? ""}`,
        () => {
          // The underlying client need not implement the optional capability.
          const fetchDetail = client.getMovementDetail?.bind(client);
          return fetchDetail
            ? fetchDetail(portfolioId, detailType, asOf, portfolioContext, signal)
            : Promise.reject(new Error("movement detail is not supported"));
        }),
    getRiskLimits: (portfolioId, portfolioContext, signal) =>
      resource(`riskLimits|${portfolioId}|${portfolioContext ?? ""}`,
        () => client.getRiskLimits(portfolioId, portfolioContext, signal)),
    getConcentrationTests: (portfolioId, portfolioContext, signal) =>
      resource(`concentrationTests|${portfolioId}|${portfolioContext ?? ""}`,
        () => client.getConcentrationTests(portfolioId, portfolioContext, signal)),
    getConcentrationDrillthrough: (portfolioId, testId, portfolioContext, signal) =>
      resource(`concentrationDrill|${portfolioId}|${testId}|${portfolioContext ?? ""}`,
        () => client.getConcentrationDrillthrough(portfolioId, testId, portfolioContext, signal)),
    getConcentrationHistory: (portfolioId, testId, portfolioContext, signal) =>
      resource(`concentrationHistory|${portfolioId}|${testId ?? ""}|${portfolioContext ?? ""}`,
        () => client.getConcentrationHistory(portfolioId, testId, portfolioContext, signal)),
    getConcentrationDrivers: (portfolioId, testId, portfolioContext, signal) =>
      resource(`concentrationDrivers|${portfolioId}|${testId}|${portfolioContext ?? ""}`,
        () => client.getConcentrationDrivers(portfolioId, testId, portfolioContext, signal)),
    getForecastExtrapolation: (portfolioId, portfolioContext, signal) =>
      resource(`forecastExtrapolation|${portfolioId}|${portfolioContext ?? ""}`,
        () => client.getForecastExtrapolation(portfolioId, portfolioContext, signal)),
    getDecks: (portfolioId, signal) =>
      resource(`decks|${portfolioId}`, () => client.getDecks(portfolioId, signal)),
    getCohorts: (portfolioId, grain, dimension, portfolioContext, signal) =>
      resource(
        `cohorts|${portfolioId}|${grain ?? "Y"}|${dimension ?? "vintage"}|${portfolioContext ?? ""}`,
        () => client.getCohorts(portfolioId, grain, dimension, portfolioContext, signal)),
    getCohortProgression: (portfolioId, query, signal) =>
      resource(
        `cohortProg|${portfolioId}|${query?.lens ?? "total"}|${query?.vintage ?? ""}|${query?.grain ?? "Y"}`,
        () => client.getCohortProgression(portfolioId, query, signal)),
    getCohortVintages: (portfolioId, query, signal) =>
      resource(
        `cohortVint|${portfolioId}|${query?.portfolioContext ?? "total"}`
        + `|${query?.vintage ?? ""}|${query?.grain ?? "M"}`,
        () => client.getCohortVintages(portfolioId, query, signal)),
    getGeoExposure: (portfolioId, portfolioContext, signal) =>
      resource(`geoExposure|${portfolioId}|${portfolioContext ?? ""}`,
        () => client.getGeoExposure(portfolioId, portfolioContext, signal)),
    deckDownloadUrl: (portfolioId, period) => client.deckDownloadUrl(portfolioId, period),
    // Uncached like the other deck commands: bytes, not a cacheable resource.
    downloadDeck: client.downloadDeck
      ? (portfolioId, period, signal) => client.downloadDeck!(portfolioId, period, signal)
      : undefined,
    // Deliberately uncached, both of them: generation is a command, and its job
    // state is the one thing in this client that is expected to change between
    // two identical calls.
    generateDeck: client.generateDeck
      ? (request, signal) => client.generateDeck!(request, signal)
      : null,
    getDeckGeneration: (jobId, signal) => {
      const poll = client.getDeckGeneration?.bind(client);
      return poll ? poll(jobId, signal)
                  : Promise.reject(new Error("deck generation is not supported"));
    },
  };
}
