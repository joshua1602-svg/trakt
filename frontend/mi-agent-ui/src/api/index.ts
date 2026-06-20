import type { AgentClient } from "./AgentClient";
import { MockAgentClient } from "./MockAgentClient";

export type { AgentClient } from "./AgentClient";
export { AgentError } from "./AgentClient";
export { MockAgentClient } from "./MockAgentClient";

/**
 * Resolve the active agent client. Today this always returns the mock; when the
 * backend lands, branch on an env flag (e.g. import.meta.env.VITE_AGENT_API) to
 * return an HttpAgentClient — no component changes required.
 */
export function createAgentClient(): AgentClient {
  return new MockAgentClient({ latencyMs: 1100 });
}
