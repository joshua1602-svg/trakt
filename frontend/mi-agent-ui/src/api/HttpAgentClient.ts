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
  MIQuerySpec,
  MovementDetail,
  MovementDetailType,
  WeeklyBrief,
  PipelineEvolution,
  PipelineFunnelEvolution,
  ConcentrationDrillthrough,
  ConcentrationDrivers,
  ConcentrationHistory,
  ConcentrationTestsSnapshot,
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
  portfolioCoverage?: import("@/domain").PortfolioCoverage;
}

/** Best-effort coarse intent from the interpreted spec (display only).
 * Reads the RAW wire spec (snake_case) — unchanged, so intent detection is
 * identical whether or not the spec is later normalised. */
function deriveIntent(spec: Record<string, unknown> | undefined): Intent {
  if (!spec) return "unknown";
  if (spec.risk_monitor || spec.risk_monitor_mode) return "risk_monitoring";
  const state = spec.state;
  if (state === "total_pipeline" || state === "total_forecast_funded") return "pipeline";
  if (typeof state === "string" && state.startsWith("cohort")) return "static_pools";
  if (spec.dimension || spec.chart_type) return "concentration_risk";
  return "portfolio_overview";
}

/**
 * Present the interpreted spec in the camelCase shape the `MIQuerySpec` type
 * declares, regardless of the wire's casing. The Python API emits snake_case
 * (`chart_type`, `top_n`, `risk_monitor_mode`); the mock already emits camelCase.
 * This maps snake→camel at the boundary (preferring an already-camel key when
 * present) so every downstream reader gets one consistent shape and the type
 * stops lying. Purely additive: it does NOT feed intent detection above, and it
 * never touches the request/parse path — so it cannot change chatbot behaviour.
 */
function normalizeSpec(raw: Record<string, unknown> | undefined): Partial<MIQuerySpec> | undefined {
  if (!raw) return undefined;
  const pick = (...keys: string[]): unknown => {
    for (const k of keys) {
      const v = raw[k];
      if (v !== undefined && v !== null) return v;
    }
    return undefined;
  };
  const out: Record<string, unknown> = {
    intent: pick("intent"),
    chartType: pick("chartType", "chart_type"),
    metric: pick("metric"),
    dimension: pick("dimension"),
    dimensions: pick("dimensions"),
    aggregation: pick("aggregation"),
    state: pick("state"),
    riskMode: pick("riskMode", "risk_monitor_mode"),
    topN: pick("topN", "top_n"),
    filters: pick("filters"),
  };
  for (const k of Object.keys(out)) if (out[k] === undefined) delete out[k];
  return out as Partial<MIQuerySpec>;
}

export class HttpAgentClient implements AgentClient {
  readonly id = "http";
  readonly mock = false;

  constructor(private readonly baseUrl: string) {
    this.baseUrl = baseUrl.replace(/\/$/, "");
  }

  /**
   * A 404 from the MI Agent API almost always means the BASE URL is wrong, not
   * that the route is missing — most often an absolute `http://localhost:8000`
   * base in an environment (Codespaces, any forwarded port) where the browser
   * cannot reach the API host directly, so the forwarding proxy answers instead.
   * Saying so turns a mystifying "404" into a one-line fix.
   */
  private describeStatus(res: Response, path: string): string {
    const where = this.baseUrl || "(same origin)";
    if (res.status === 404) {
      const absolute = /^https?:\/\//i.test(this.baseUrl);
      const hint = absolute
        ? ` The base URL ${this.baseUrl} is absolute — if the app is served from `
          + "another host (Codespaces / a forwarded port) the browser cannot reach "
          + "it. Use VITE_AGENT_API_URL=/ so the dev-server proxy forwards to the API."
        : " Check that the MI Agent API is running and that the dev-server proxy "
          + "forwards this path.";
      return `MI Agent API returned 404 for ${path} (base ${where}).${hint}`;
    }
    return `MI Agent API returned ${res.status} ${res.statusText} for ${path}`;
  }

  private async getJson<T>(path: string, signal?: AbortSignal): Promise<T> {
    let res: Response;
    try {
      res = await fetch(`${this.baseUrl}${path}`, { signal });
    } catch (err) {
      if ((err as Error)?.name === "AbortError") throw new AgentError("Request aborted", err);
      throw new AgentError(`Could not reach the MI Agent API at ${this.baseUrl}.`, err);
    }
    if (!res.ok) throw new AgentError(this.describeStatus(res, path));
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

  getPortfolioContext(signal?: AbortSignal) {
    return this.getJson<import("@/domain").PortfolioContextIndex>(
      "/mi/portfolio-context", signal);
  }

  /** `?portfolioId=…` plus the governed portfolio context, when one is selected.
   * One query-string builder for every portfolio-scoped route, so no call site
   * can forget to pass the workspace scope. */
  private scoped(portfolioId: string, portfolioContext?: string,
                 extra?: Record<string, string | undefined>): string {
    const q = new URLSearchParams({ portfolioId });
    if (portfolioContext) q.set("portfolioContext", portfolioContext);
    for (const [k, v] of Object.entries(extra ?? {})) if (v) q.set(k, v);
    return q.toString();
  }

  getSnapshots(signal?: AbortSignal): Promise<SnapshotIndex> {
    return this.getJson<SnapshotIndex>("/mi/snapshots", signal);
  }

  getSnapshot(portfolioId: string, portfolioContext?: string,
             signal?: AbortSignal): Promise<FundedSnapshot> {
    return this.getJson<FundedSnapshot>(
      `/mi/snapshot?${this.scoped(portfolioId, portfolioContext)}`, signal);
  }

  getForecastSnapshot(portfolioId: string, portfolioContext?: string,
                     signal?: AbortSignal): Promise<ForecastSnapshot> {
    return this.getJson<ForecastSnapshot>(
      `/mi/forecast/snapshot?${this.scoped(portfolioId, portfolioContext)}`, signal);
  }

  getFundedEvolution(portfolioId: string, portfolioContext?: string,
                    signal?: AbortSignal): Promise<FundedEvolution> {
    return this.getJson<FundedEvolution>(
      `/mi/evolution/funded?${this.scoped(portfolioId, portfolioContext)}`, signal);
  }

  getPipelineEvolution(portfolioId: string, portfolioContext?: string,
                      signal?: AbortSignal): Promise<PipelineEvolution> {
    return this.getJson<PipelineEvolution>(
      `/mi/evolution/pipeline?${this.scoped(portfolioId, portfolioContext)}`, signal);
  }

  getMovementDetail(portfolioId: string, detailType: MovementDetailType,
                   asOf?: string, portfolioContext?: string,
                   signal?: AbortSignal): Promise<MovementDetail> {
    return this.getJson<MovementDetail>(
      `/mi/insight/movement-detail?${this.scoped(portfolioId, portfolioContext,
        { detailType, asOf })}`, signal);
  }

  getWeeklyBrief(portfolioId: string, portfolioContext?: string, asOf?: string,
                signal?: AbortSignal): Promise<WeeklyBrief> {
    return this.getJson<WeeklyBrief>(
      `/mi/insights/weekly-brief?${this.scoped(portfolioId, portfolioContext,
        { asOf })}`, signal);
  }

  getForecastEvolution(portfolioId: string, portfolioContext?: string,
                      signal?: AbortSignal): Promise<ForecastEvolution> {
    return this.getJson<ForecastEvolution>(
      `/mi/evolution/forecast?${this.scoped(portfolioId, portfolioContext)}`, signal);
  }

  getFunnelEvolution(portfolioId: string, portfolioContext?: string,
                    signal?: AbortSignal): Promise<PipelineFunnelEvolution> {
    return this.getJson<PipelineFunnelEvolution>(
      `/mi/evolution/funnel?${this.scoped(portfolioId, portfolioContext)}`, signal);
  }

  getRiskLimits(portfolioId: string, portfolioContext?: string,
               signal?: AbortSignal): Promise<RiskLimitsSnapshot> {
    return this.getJson<RiskLimitsSnapshot>(
      `/mi/risk-limits?${this.scoped(portfolioId, portfolioContext)}`, signal);
  }

  getConcentrationTests(portfolioId: string, portfolioContext?: string,
                        signal?: AbortSignal): Promise<ConcentrationTestsSnapshot> {
    return this.getJson<ConcentrationTestsSnapshot>(
      `/mi/concentration-tests?${this.scoped(portfolioId, portfolioContext)}`,
      signal);
  }

  getConcentrationDrillthrough(portfolioId: string, testId: string,
                               portfolioContext?: string,
                               signal?: AbortSignal): Promise<ConcentrationDrillthrough> {
    return this.getJson<ConcentrationDrillthrough>(
      `/mi/concentration-tests/drillthrough?${this.scoped(portfolioId, portfolioContext, { testId })}`,
      signal);
  }

  getConcentrationHistory(portfolioId: string, testId?: string,
                          portfolioContext?: string,
                          signal?: AbortSignal): Promise<ConcentrationHistory> {
    return this.getJson<ConcentrationHistory>(
      `/mi/concentration-tests/history?${this.scoped(portfolioId, portfolioContext, { testId })}`,
      signal);
  }

  getConcentrationDrivers(portfolioId: string, testId: string,
                          portfolioContext?: string,
                          signal?: AbortSignal): Promise<ConcentrationDrivers> {
    return this.getJson<ConcentrationDrivers>(
      `/mi/concentration-tests/drivers?${this.scoped(portfolioId, portfolioContext, { testId })}`,
      signal);
  }

  getForecastExtrapolation(portfolioId: string, portfolioContext?: string,
                          signal?: AbortSignal): Promise<ForecastExtrapolation> {
    return this.getJson<ForecastExtrapolation>(
      `/mi/forecast/extrapolation?${this.scoped(portfolioId, portfolioContext)}`, signal);
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

  /**
   * Request a deck. The API answers 202 with a job, so a non-2xx here means the
   * request itself was refused (not authorised, no data for that period, the
   * deployment does not offer on-demand packs) — surface the API's own reason
   * rather than a generic transport message, because those reasons are the ones
   * a user can act on.
   */
  async generateDeck(request: import("@/domain").DeckGenerationRequest,
                     signal?: AbortSignal
                     ): Promise<import("@/domain").DeckGenerationJob> {
    let res: Response;
    try {
      res = await fetch(`${this.baseUrl}/mi/decks/generate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(request),
        signal,
      });
    } catch (err) {
      if ((err as Error)?.name === "AbortError") throw new AgentError("Request aborted", err);
      throw new AgentError(`Could not reach the MI Agent API at ${this.baseUrl}.`, err);
    }
    const body = await res.json().catch(() => null) as
      (import("@/domain").DeckGenerationJob & { error?: string }) | null;
    if (!res.ok) {
      throw new AgentError(body?.error || this.describeStatus(res, "/mi/decks/generate"));
    }
    if (!body) throw new AgentError("MI Agent API returned an invalid response");
    return body;
  }

  getDeckGeneration(jobId: string, signal?: AbortSignal
                    ): Promise<import("@/domain").DeckGenerationJob> {
    return this.getJson<import("@/domain").DeckGenerationJob>(
      `/mi/decks/generate/${encodeURIComponent(jobId)}`, signal);
  }

  getCohorts(portfolioId: string, grain?: import("@/domain").CohortGrain,
             dimension?: import("@/domain").CohortDimension,
             portfolioContext?: string,
             signal?: AbortSignal): Promise<import("@/domain").CohortAnalysis> {
    return this.getJson<import("@/domain").CohortAnalysis>(
      `/mi/cohorts?${this.scoped(portfolioId, portfolioContext, { grain, dimension })}`,
      signal);
  }

  getGeoExposure(portfolioId: string, portfolioContext?: string,
                 signal?: AbortSignal): Promise<import("@/domain").GeoExposure> {
    return this.getJson<import("@/domain").GeoExposure>(
      `/mi/geo/exposure?${this.scoped(portfolioId, portfolioContext)}`, signal);
  }

  getCohortProgression(portfolioId: string,
                       query?: import("@/domain").CohortProgressionQuery,
                       signal?: AbortSignal): Promise<import("@/domain").CohortProgression> {
    const p = new URLSearchParams({ portfolioId });
    if (query?.lens) p.set("portfolioContext", query.lens);
    if (query?.vintage) p.set("vintage", query.vintage);
    if (query?.grain) p.set("grain", query.grain);
    return this.getJson<import("@/domain").CohortProgression>(
      `/mi/cohorts/progression?${p.toString()}`, signal);
  }

  getCohortVintages(portfolioId: string,
                    query?: import("@/domain").CohortVintageQuery,
                    signal?: AbortSignal
  ): Promise<import("@/domain").CohortFormation | import("@/domain").CohortStaticPool> {
    const p = new URLSearchParams({ portfolioId });
    if (query?.portfolioContext) p.set("portfolioContext", query.portfolioContext);
    if (query?.vintage) p.set("vintage", query.vintage);
    if (query?.grain) p.set("grain", query.grain);
    return this.getJson<import("@/domain").CohortFormation
      | import("@/domain").CohortStaticPool>(
      `/mi/cohorts/vintages?${p.toString()}`, signal);
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
          // The governed workspace scope. A portfolio named in the question
          // overrides it backend-side; the UI never resolves it itself.
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
      // Governance refusals (unapproved source, unauthorised portfolio, data
      // store down) carry a mapped status WITH the full governed envelope —
      // surface the backend's client-facing reason, not a bare status line.
      let governedMessage: string | undefined;
      try {
        const errBody = (await res.json()) as ApiResponse;
        governedMessage =
          (typeof errBody.error === "string" && errBody.error) ||
          (typeof errBody.answer === "string" && errBody.answer) ||
          undefined;
      } catch {
        /* not a JSON envelope — fall back to the status description */
      }
      throw new AgentError(governedMessage ?? this.describeStatus(res, "/mi/query"));
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
      spec: normalizeSpec(body.spec),
      datasetContext: asString(meta.datasetContext),
      // Governed portfolio coverage, passed through verbatim. The UI shows what
      // the backend decided about consolidation — it never recomputes it.
      portfolioCoverage: body.portfolioCoverage,
      error: body.error ?? undefined,
    };
  }
}
