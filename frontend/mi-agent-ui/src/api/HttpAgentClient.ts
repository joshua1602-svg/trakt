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
  ForecastEvolution,
  ForecastExtrapolation,
  ForecastSnapshot,
  FundedEvolution,
  FundedSnapshot,
  Intent,
  PipelineEvolution,
  PipelineFunnelEvolution,
  RiskLimitsSnapshot,
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

  getSourcePortfolios(signal?: AbortSignal) {
    return this.getJson<import("@/domain").SourcePortfolioIndex>(
      "/mi/source-portfolios", signal);
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

  getForecastSnapshot(portfolioId: string, signal?: AbortSignal): Promise<ForecastSnapshot> {
    return this.getJson<ForecastSnapshot>(
      `/mi/forecast/snapshot?portfolioId=${encodeURIComponent(portfolioId)}`,
      signal,
    );
  }

  getFundedEvolution(portfolioId: string, signal?: AbortSignal): Promise<FundedEvolution> {
    return this.getJson<FundedEvolution>(
      `/mi/evolution/funded?portfolioId=${encodeURIComponent(portfolioId)}`, signal);
  }

  getPipelineEvolution(portfolioId: string, signal?: AbortSignal): Promise<PipelineEvolution> {
    return this.getJson<PipelineEvolution>(
      `/mi/evolution/pipeline?portfolioId=${encodeURIComponent(portfolioId)}`, signal);
  }

  getForecastEvolution(portfolioId: string, signal?: AbortSignal): Promise<ForecastEvolution> {
    return this.getJson<ForecastEvolution>(
      `/mi/evolution/forecast?portfolioId=${encodeURIComponent(portfolioId)}`, signal);
  }

  getFunnelEvolution(portfolioId: string, signal?: AbortSignal): Promise<PipelineFunnelEvolution> {
    return this.getJson<PipelineFunnelEvolution>(
      `/mi/evolution/funnel?portfolioId=${encodeURIComponent(portfolioId)}`, signal);
  }

  getRiskLimits(portfolioId: string, signal?: AbortSignal): Promise<RiskLimitsSnapshot> {
    return this.getJson<RiskLimitsSnapshot>(
      `/mi/risk-limits?portfolioId=${encodeURIComponent(portfolioId)}`, signal);
  }

  getForecastExtrapolation(portfolioId: string, signal?: AbortSignal): Promise<ForecastExtrapolation> {
    return this.getJson<ForecastExtrapolation>(
      `/mi/forecast/extrapolation?portfolioId=${encodeURIComponent(portfolioId)}`, signal);
  }

  async getMe(signal?: AbortSignal): Promise<import("@/lib/identity").UserIdentity> {
    try {
      return await this.getJson<import("@/lib/identity").UserIdentity>("/me", signal);
    } catch {
      // The header degrades gracefully when /me is unreachable (auth removed for
      // the test deployment) — treat as an unauthenticated, role-less caller.
      return { authenticated: false };
    }
  }

  getDecks(portfolioId: string, signal?: AbortSignal): Promise<import("@/domain").DeckIndex> {
    return this.getJson<import("@/domain").DeckIndex>(
      `/mi/decks?portfolioId=${encodeURIComponent(portfolioId)}`, signal);
  }

  deckDownloadUrl(portfolioId: string, period?: string | null): string {
    const q = new URLSearchParams({ portfolioId });
    if (period) q.set("period", period);
    return `${this.baseUrl}/mi/decks/download?${q.toString()}`;
  }

  getCohorts(portfolioId: string, signal?: AbortSignal): Promise<import("@/domain").CohortAnalysis> {
    return this.getJson<import("@/domain").CohortAnalysis>(
      `/mi/cohorts?portfolioId=${encodeURIComponent(portfolioId)}`, signal);
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
          datasetContext: request.datasetContext,
          sourcePortfolioLens: request.sourceLens,
          // Merge the top_n hint with any drill-through filters; send undefined
          // when neither is present so the contract stays additive.
          filters: ((): Record<string, unknown> | undefined => {
            const merged: Record<string, unknown> = {
              ...(request.options?.topN ? { top_n: request.options.topN } : {}),
              ...(request.filters ?? {}),
            };
            return Object.keys(merged).length ? merged : undefined;
          })(),
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
    const meta = body.metadata ?? {};
    const asString = (v: unknown): string | undefined =>
      typeof v === "string" && v.length > 0 ? v : undefined;

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
      datasetContext: asString(meta.datasetContext),
      parserMode: asString(meta.parserMode),
      error: body.error ?? undefined,
    };
  }
}
