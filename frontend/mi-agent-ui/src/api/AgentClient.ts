/**
 * AgentClient — the single boundary between the UI and the MI Agent.
 *
 * Components depend only on this interface; they never import mock logic or a
 * concrete transport. Swapping `MockAgentClient` for a future `HttpAgentClient`
 * (posting an AgentRequest to the MI Agent API) requires no component changes.
 */

import type {
  AgentRequest,
  AgentResponse,
  ForecastEvolution,
  ForecastExtrapolation,
  ForecastSnapshot,
  FundedEvolution,
  FundedSnapshot,
  PipelineEvolution,
  PipelineFunnelEvolution,
  RiskLimitsSnapshot,
  SnapshotIndex,
} from "@/domain";

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

  /**
   * Deterministic funded + pipeline forecast bridge (pipeline snapshot, forecast
   * bridge and watchlist) for a `"<client_id>/<run_id>"` portfolio. The forecast
   * is backend-derived — the UI only renders it.
   */
  getForecastSnapshot(portfolioId: string, signal?: AbortSignal): Promise<ForecastSnapshot>;

  /** Funded time series (per-month metrics + breakdowns) up to the selected run. */
  getFundedEvolution(portfolioId: string, signal?: AbortSignal): Promise<FundedEvolution>;

  /** Pipeline time series (weekly amount/cases + by-stage over time). */
  getPipelineEvolution(portfolioId: string, signal?: AbortSignal): Promise<PipelineEvolution>;

  /** Forecast bridge over time (funded balance + weighted pipeline per run). */
  getForecastEvolution(portfolioId: string, signal?: AbortSignal): Promise<ForecastEvolution>;

  /** Weekly origination funnel trends (KFI / Application / Offer / Completion). */
  getFunnelEvolution(portfolioId: string, signal?: AbortSignal): Promise<PipelineFunnelEvolution>;

  /** Governed risk-limit / concentration monitor (Schedule 8 vs funded actuals). */
  getRiskLimits(portfolioId: string, signal?: AbortSignal): Promise<RiskLimitsSnapshot>;

  /** Securitisation scale-up forecast (run-rate / KFI extrapolation + milestones). */
  getForecastExtrapolation(portfolioId: string, signal?: AbortSignal): Promise<ForecastExtrapolation>;
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
