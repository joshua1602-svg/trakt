/**
 * MockAgentClient — in-memory implementation of AgentClient.
 *
 * Simulates streaming-style latency and (optionally) transient failures so the
 * UI's loading and error states are exercised. Replace with HttpAgentClient
 * when the MI Agent API is available — the contract is identical.
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
import { buildAgentResponse } from "@/data/mockResponses";
import { MOCK_SNAPSHOT_INDEX, mockSnapshot } from "@/data/mockSnapshots";
import { mockForecastSnapshot } from "@/data/mockForecast";
import {
  mockFundedEvolution,
  mockPipelineEvolution,
  mockForecastEvolution,
} from "@/data/mockEvolution";
import { mockFunnelEvolution } from "@/data/mockFunnel";
import { mockCohorts } from "@/data/mockCohorts";
import { mockRiskLimits } from "@/data/mockRiskLimits";
import { mockForecastExtrapolation } from "@/data/mockForecastExtrapolation";
import { AgentError, type AgentClient } from "./AgentClient";

export interface MockAgentClientOptions {
  /** Simulated latency in ms. */
  latencyMs?: number;
  /** Probability [0,1] of a simulated transient failure (default 0). */
  failureRate?: number;
}

export class MockAgentClient implements AgentClient {
  readonly id = "mock";
  readonly mock = true;

  private readonly latencyMs: number;
  private readonly failureRate: number;

  constructor(opts: MockAgentClientOptions = {}) {
    this.latencyMs = opts.latencyMs ?? 1100;
    this.failureRate = opts.failureRate ?? 0;
  }

  ask(request: AgentRequest, signal?: AbortSignal): Promise<AgentResponse> {
    return new Promise<AgentResponse>((resolve, reject) => {
      if (signal?.aborted) {
        reject(new AgentError("Request aborted"));
        return;
      }
      const timer = setTimeout(() => {
        if (Math.random() < this.failureRate) {
          reject(new AgentError("The MI Agent timed out composing this response."));
          return;
        }
        try {
          resolve(buildAgentResponse(request));
        } catch (err) {
          reject(new AgentError("Failed to compose agent response", err));
        }
      }, this.latencyMs);

      signal?.addEventListener("abort", () => {
        clearTimeout(timer);
        reject(new AgentError("Request aborted"));
      });
    });
  }

  getSnapshots(): Promise<SnapshotIndex> {
    return Promise.resolve(MOCK_SNAPSHOT_INDEX);
  }

  getSourcePortfolios() {
    return Promise.resolve({
      available: false,
      source: "mock",
      lenses: [
        { id: "total", kind: "total" as const, label: "Total", filters: {}, funded_only: false },
      ],
    });
  }

  getSnapshot(portfolioId: string): Promise<FundedSnapshot> {
    return Promise.resolve(mockSnapshot(portfolioId));
  }

  getForecastSnapshot(portfolioId: string): Promise<ForecastSnapshot> {
    return Promise.resolve(mockForecastSnapshot(portfolioId));
  }

  getFundedEvolution(portfolioId: string): Promise<FundedEvolution> {
    return Promise.resolve(mockFundedEvolution(portfolioId));
  }

  getPipelineEvolution(portfolioId: string): Promise<PipelineEvolution> {
    return Promise.resolve(mockPipelineEvolution(portfolioId));
  }

  getForecastEvolution(portfolioId: string): Promise<ForecastEvolution> {
    return Promise.resolve(mockForecastEvolution(portfolioId));
  }

  getFunnelEvolution(portfolioId: string): Promise<PipelineFunnelEvolution> {
    return Promise.resolve(mockFunnelEvolution(portfolioId));
  }

  getRiskLimits(portfolioId: string): Promise<RiskLimitsSnapshot> {
    return Promise.resolve(mockRiskLimits(portfolioId));
  }

  getForecastExtrapolation(portfolioId: string): Promise<ForecastExtrapolation> {
    return Promise.resolve(mockForecastExtrapolation(portfolioId));
  }

  getMe(): Promise<import("@/lib/identity").UserIdentity> {
    // Mock mode has no real auth — present a demo operator so role-gated controls
    // are exercisable in staging without leaking a hardcoded production name.
    return Promise.resolve({
      authenticated: true, user: "Demo Operator",
      roles: ["operator"], isOperator: true,
    });
  }

  getDecks(portfolioId: string): Promise<import("@/domain").DeckIndex> {
    // Deterministic mock: a latest deck + one dated period, so the download menu
    // and its enabled/disabled states are exercisable without a backend.
    const client = portfolioId.split("/")[0] || "client_001";
    return Promise.resolve({
      available: true,
      latest: { period: "2025-11", generatedAt: "2025-11-28T09:00:00Z" },
      decks: [{ period: "2025-11" }, { period: "2025-10" }],
      client_id: client,
    });
  }

  deckDownloadUrl(): string | null {
    return null; // mock cannot serve real .pptx bytes
  }

  getCohorts(portfolioId: string): Promise<import("@/domain").CohortAnalysis> {
    return Promise.resolve(mockCohorts(portfolioId));
  }
}
