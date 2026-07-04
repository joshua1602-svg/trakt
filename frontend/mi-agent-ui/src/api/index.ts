import { MockAgentClient } from "./MockAgentClient";
import { HttpAgentClient } from "./HttpAgentClient";
import { withCache, type CachingAgentClient } from "./CachingAgentClient";

export type { AgentClient } from "./AgentClient";
export { AgentError } from "./AgentClient";
export { MockAgentClient } from "./MockAgentClient";
export { HttpAgentClient } from "./HttpAgentClient";
export { withCache, buildCacheKey, RESOURCE_STALE_MS } from "./CachingAgentClient";
export type { CachingAgentClient } from "./CachingAgentClient";

/** The Vite build-time env fields this module reads (subset, injectable for tests). */
export interface AgentEnv {
  VITE_AGENT_API_URL?: string;
  VITE_AGENT_MODE?: string;
  PROD?: boolean;
}

export interface AgentClientConfig {
  url?: string;
  /** True when the resolved client is the in-memory mock. */
  isMock: boolean;
  /** True when the mock was chosen by an explicit VITE_AGENT_MODE=mock opt-in. */
  explicitMock: boolean;
  /**
   * True when a PRODUCTION build fell back to the mock WITHOUT an explicit
   * VITE_AGENT_MODE=mock — i.e. VITE_AGENT_API_URL was forgotten and the bundle
   * would otherwise silently serve canned demo answers as if it were live.
   */
  misconfigured: boolean;
}

/**
 * Resolve the agent-client configuration from build-time env:
 *
 *   VITE_AGENT_API_URL  - base URL of the MI Agent API (e.g. http://localhost:8000)
 *   VITE_AGENT_MODE     - "mock" forces the mock client even if a URL is set
 *
 * Pure and env-injectable so the misconfiguration rule is unit-testable.
 */
export function resolveAgentClientConfig(
  env: AgentEnv = import.meta.env as unknown as AgentEnv,
): AgentClientConfig {
  const url = env.VITE_AGENT_API_URL;
  const explicitMock = env.VITE_AGENT_MODE === "mock";
  const useHttp = !!url && !explicitMock;
  const isMock = !useHttp;
  const misconfigured = isMock && !explicitMock && !!env.PROD;
  return { url, isMock, explicitMock, misconfigured };
}

/**
 * Resolve the active agent client from build-time config.
 *
 * A production build that would fall back to the mock without an explicit
 * VITE_AGENT_MODE=mock opt-in is a deployment fault: instead of silently
 * shipping demo data as if live, we log a prominent error (and the UI shows a
 * hard banner via {@link resolveAgentClientConfig}). Local dev still falls back
 * to the mock so the app runs without a backend, and an explicit demo build is
 * always honoured.
 */
export function createAgentClient(): CachingAgentClient {
  const cfg = resolveAgentClientConfig();
  if (cfg.misconfigured) {
    // eslint-disable-next-line no-console
    console.error(
      "[MI Agent] VITE_AGENT_API_URL is not set in a production build — the app " +
        "is serving the in-memory MOCK client (canned demo data), NOT a live " +
        "backend. Set VITE_AGENT_API_URL to the MI Agent API, or set " +
        "VITE_AGENT_MODE=mock to acknowledge an intentional demo build.",
    );
  }
  const base = cfg.isMock
    ? new MockAgentClient({ latencyMs: 1100 })
    : new HttpAgentClient(cfg.url!);
  // Transparent session result cache (additive; falls through on any miss/failure).
  return withCache(base);
}
