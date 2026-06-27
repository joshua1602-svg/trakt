/**
 * CachingAgentClient — a transparent, additive decorator over any AgentClient
 * that caches SUCCESSFUL `ask` responses in memory for the session.
 *
 * The cache key includes the portfolio, normalised question, filters, as-of date
 * and dataset view, so a repeat of the same query is fast while a different
 * portfolio / snapshot / filter never returns a stale result. Only `ok`
 * responses are cached; failures always re-run. If anything about keying fails,
 * the call falls through to the underlying client unchanged.
 */

import type { AgentClient } from "./AgentClient";
import type { AgentRequest, AgentResponse } from "@/domain";

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
    req.question.trim().toLowerCase().replace(/\s+/g, " "),
    stableFilters(req.filters),
    req.options?.topN ?? "",
  ].join("|");
}

export function withCache(client: AgentClient): AgentClient {
  const cache = new Map<string, AgentResponse>();

  return {
    id: client.id,
    mock: client.mock,

    async ask(request: AgentRequest, signal?: AbortSignal): Promise<AgentResponse> {
      let key: string | undefined;
      try {
        key = buildCacheKey(request);
      } catch {
        key = undefined; // keying failed → behave like the underlying client
      }
      if (key) {
        const hit = cache.get(key);
        if (hit) return { ...hit, cacheHit: true };
      }
      const res = await client.ask(request, signal);
      if (key && res.ok) {
        const stored = { ...res, cacheHit: false };
        cache.set(key, stored);
        return stored;
      }
      return res;
    },

    getSnapshots: (signal) => client.getSnapshots(signal),
    getSnapshot: (portfolioId, signal) => client.getSnapshot(portfolioId, signal),
    getForecastSnapshot: (portfolioId, signal) => client.getForecastSnapshot(portfolioId, signal),
    getFundedEvolution: (portfolioId, signal) => client.getFundedEvolution(portfolioId, signal),
    getPipelineEvolution: (portfolioId, signal) => client.getPipelineEvolution(portfolioId, signal),
    getForecastEvolution: (portfolioId, signal) => client.getForecastEvolution(portfolioId, signal),
  };
}
