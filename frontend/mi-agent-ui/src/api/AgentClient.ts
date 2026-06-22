/**
 * AgentClient — the single boundary between the UI and the MI Agent.
 *
 * Components depend only on this interface; they never import mock logic or a
 * concrete transport. Swapping `MockAgentClient` for a future `HttpAgentClient`
 * (posting an AgentRequest to the MI Agent API) requires no component changes.
 */

import type { AgentRequest, AgentResponse, FundedSnapshot, SnapshotIndex } from "@/domain";

export interface AgentClient {
  /** Identifier surfaced in the UI (e.g. environment badge). */
  readonly id: string;
  /** True when responses are mocked (drives the mock-data disclosure). */
  readonly mock: boolean;

  /** Submit a question; resolves with a structured response. */
  ask(request: AgentRequest, signal?: AbortSignal): Promise<AgentResponse>;

  /** Discover available funded portfolios and reporting runs (data-driven dropdowns). */
  getSnapshots(signal?: AbortSignal): Promise<SnapshotIndex>;

  /** Deterministic funded-book snapshot for a `"<client_id>/<run_id>"` portfolio. */
  getSnapshot(portfolioId: string, signal?: AbortSignal): Promise<FundedSnapshot>;
}

/** Error thrown by clients for transport/agent failures. */
export class AgentError extends Error {
  constructor(
    message: string,
    readonly cause?: unknown,
  ) {
    super(message);
    this.name = "AgentError";
  }
}
