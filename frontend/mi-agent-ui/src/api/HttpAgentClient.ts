/**
 * HttpAgentClient — talks to the MI Agent API (mi_agent_api, FastAPI).
 *
 * Implements the same AgentClient interface as MockAgentClient, so swapping
 * between them is a config decision (see ./index.ts). The API already returns
 * artifacts in the React artifact schema (the Python adapter does the mapping),
 * so this client is a thin transport + envelope translation.
 */

import type {
  AgentRequest,
  AgentResponse,
  Artifact,
  FundedSnapshot,
  Intent,
  SnapshotIndex,
} from "@/domain";
import { isArtifact } from "@/domain";
import { AgentError, type AgentClient } from "./AgentClient";

interface ApiResponse {
  ok: boolean;
  error?: string | null;
  question?: string;
  answer?: string;
  interpreted?: string;
  spec?: Record<string, unknown>;
  validation?: Record<string, unknown>;
  artifacts?: unknown[];
  warnings?: string[];
  diagnostics?: string[];
  assumptions?: string[];
  metadata?: Record<string, unknown>;
}

/** Best-effort coarse intent from the interpreted spec (display only). */
function deriveIntent(spec: Record<string, unknown> | undefined): Intent {
  if (!spec) return "unknown";
  if (spec.risk_monitor || spec.risk_monitor_mode) return "risk_monitoring";
  const state = spec.state;
  if (state === "total_pipeline" || state === "total_forecast_funded") return "pipeline";
  if (typeof state === "string" && state.startsWith("cohort")) return "static_pools";
  if (spec.dimension || spec.chart_type) return "concentration_risk";
  return "portfolio_overview";
}

export class HttpAgentClient implements AgentClient {
  readonly id = "http";
  readonly mock = false;

  constructor(private readonly baseUrl: string) {
    this.baseUrl = baseUrl.replace(/\/$/, "");
  }

  private async getJson<T>(path: string, signal?: AbortSignal): Promise<T> {
    let res: Response;
    try {
      res = await fetch(`${this.baseUrl}${path}`, { signal });
    } catch (err) {
      if ((err as Error)?.name === "AbortError") throw new AgentError("Request aborted", err);
      throw new AgentError(`Could not reach the MI Agent API at ${this.baseUrl}.`, err);
    }
    if (!res.ok) throw new AgentError(`MI Agent API returned ${res.status} ${res.statusText}`);
    try {
      return (await res.json()) as T;
    } catch (err) {
      throw new AgentError("MI Agent API returned an invalid response", err);
    }
  }

  getSnapshots(signal?: AbortSignal): Promise<SnapshotIndex> {
    return this.getJson<SnapshotIndex>("/mi/snapshots", signal);
  }

  getSnapshot(portfolioId: string, signal?: AbortSignal): Promise<FundedSnapshot> {
    return this.getJson<FundedSnapshot>(
      `/mi/snapshot?portfolioId=${encodeURIComponent(portfolioId)}`,
      signal,
    );
  }

  async ask(request: AgentRequest, signal?: AbortSignal): Promise<AgentResponse> {
    let res: Response;
    try {
      res = await fetch(`${this.baseUrl}/mi/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question: request.question,
          portfolio: request.portfolio,
          portfolioId: request.portfolio.id,
          asOfDate: request.reporting.asOf,
          filters: request.options?.topN ? { top_n: request.options.topN } : undefined,
        }),
        signal,
      });
    } catch (err) {
      if ((err as Error)?.name === "AbortError") throw new AgentError("Request aborted", err);
      throw new AgentError(
        "Could not reach the MI Agent API. Is the backend running on " + this.baseUrl + "?",
        err,
      );
    }

    if (!res.ok) {
      throw new AgentError(`MI Agent API returned ${res.status} ${res.statusText}`);
    }

    let body: ApiResponse;
    try {
      body = (await res.json()) as ApiResponse;
    } catch (err) {
      throw new AgentError("MI Agent API returned an invalid response", err);
    }

    const artifacts = (body.artifacts ?? []).filter(isArtifact) as Artifact[];

    return {
      ok: !!body.ok,
      question: body.question ?? request.question,
      intent: deriveIntent(body.spec),
      interpreted: body.interpreted,
      narrative: body.answer ?? (body.ok ? "Query executed." : body.error ?? "The query could not be completed."),
      assumptions: body.assumptions ?? [],
      artifacts,
      warnings: body.warnings ?? [],
      diagnostics: body.diagnostics ?? [],
      spec: body.spec,
      error: body.error ?? undefined,
    };
  }
}
