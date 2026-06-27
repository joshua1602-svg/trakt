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
  ForecastSnapshot,
  FundedEvolution,
  FundedSnapshot,
  PipelineEvolution,
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
}
