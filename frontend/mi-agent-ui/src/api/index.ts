import type { AgentClient } from "./AgentClient";
import { MockAgentClient } from "./MockAgentClient";
import { HttpAgentClient } from "./HttpAgentClient";
import { withCache } from "./CachingAgentClient";

export type { AgentClient } from "./AgentClient";
export { AgentError } from "./AgentClient";
export { MockAgentClient } from "./MockAgentClient";
export { HttpAgentClient } from "./HttpAgentClient";
export { withCache, buildCacheKey } from "./CachingAgentClient";

/**
 * Resolve the active agent client from build-time config:
 *
 *   VITE_AGENT_API_URL  - base URL of the MI Agent API (e.g. http://localhost:8000)
 *   VITE_AGENT_MODE     - "mock" forces the mock client even if a URL is set
 *
 * Default behaviour: use the HTTP client when a URL is configured, otherwise
 * fall back to the mock so the app always builds/runs without a backend.
 */
export function createAgentClient(): AgentClient {
  const url = import.meta.env.VITE_AGENT_API_URL as string | undefined;
  const mode = import.meta.env.VITE_AGENT_MODE as string | undefined;

  const base = url && mode !== "mock" ? new HttpAgentClient(url) : new MockAgentClient({ latencyMs: 1100 });
  // Transparent session result cache (additive; falls through on any miss/failure).
  return withCache(base);
}
