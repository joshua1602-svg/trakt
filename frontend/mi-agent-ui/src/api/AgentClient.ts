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
  CohortAnalysis,
  CohortDimension,
  CohortGrain,
  CohortFormation,
  CohortProgression,
  CohortProgressionQuery,
  CohortStaticPool,
  CohortVintageQuery,
  DeckGenerationJob,
  DeckGenerationRequest,
  DeckIndex,
  ForecastEvolution,
  ForecastExtrapolation,
  ForecastSnapshot,
  FundedEvolution,
  FundedSnapshot,
  GeoExposure,
  MovementDetail,
  MovementDetailType,
  WeeklyBrief,
  PipelineEvolution,
  ConcentrationDrillthrough,
  ConcentrationDrivers,
  ConcentrationHistory,
  ConcentrationTestsSnapshot,
  PipelineFunnelEvolution,
  PortfolioContextIndex,
  RiskLimitsSnapshot,
  SnapshotIndex,
  SourcePortfolioIndex,
} from "@/domain";
import type { UserIdentity } from "@/lib/identity";

export interface AgentClient {
  /** Identifier surfaced in the UI (e.g. environment badge). */
  readonly id: string;
  /** True when responses are mocked (drives the mock-data disclosure). */
  readonly mock: boolean;

  /** Submit a question; resolves with a structured response. */
  ask(request: AgentRequest, signal?: AbortSignal): Promise<AgentResponse>;

  /** Discover available funded portfolios and reporting runs (data-driven dropdowns). */
  getSnapshots(signal?: AbortSignal): Promise<SnapshotIndex>;

  /** Discover the source-portfolio lenses (Total / Direct / Acquired / cohorts)
   * present in the active dataset, for the portfolio-scope dropdown.
   * @deprecated Superseded by `getPortfolioContext()`, which carries the full
   * governed hierarchy AND the capability set per context. Retained so existing
   * API consumers keep working. */
  getSourcePortfolios(signal?: AbortSignal): Promise<SourcePortfolioIndex>;

  /**
   * THE governed portfolio contract: the dynamic hierarchy (Total → type groups
   * → every source portfolio), each portfolio's governed metadata, and the
   * resolved capability set for every selectable context.
   *
   * This is the single thing the workspace gates on. The UI never infers what a
   * group contains or whether an analysis applies — it reads it from here.
   */
  getPortfolioContext(signal?: AbortSignal): Promise<PortfolioContextIndex>;

  /** Deterministic funded-book snapshot for a `"<client_id>/<run_id>"` portfolio,
   * scoped to the governed `portfolioContext` (Total / a type group / one source
   * portfolio). The backend computes every figure over those rows. */
  getSnapshot(portfolioId: string, portfolioContext?: string,
              signal?: AbortSignal): Promise<FundedSnapshot>;

  /**
   * Deterministic funded + pipeline forecast bridge (pipeline snapshot, forecast
   * bridge and watchlist) for a `"<client_id>/<run_id>"` portfolio. The forecast
   * is backend-derived — the UI only renders it.
   */
  getForecastSnapshot(portfolioId: string, portfolioContext?: string,
                      signal?: AbortSignal): Promise<ForecastSnapshot>;

  /** Funded time series (per-month metrics + breakdowns) up to the selected run. */
  getFundedEvolution(portfolioId: string, portfolioContext?: string,
                    signal?: AbortSignal): Promise<FundedEvolution>;

  /** Pipeline time series (weekly amount/cases + by-stage over time). */
  getPipelineEvolution(portfolioId: string, portfolioContext?: string,
                      signal?: AbortSignal): Promise<PipelineEvolution>;

  /** Forecast bridge over time (funded balance + weighted pipeline per run). */
  getForecastEvolution(portfolioId: string, portfolioContext?: string,
                      signal?: AbortSignal): Promise<ForecastEvolution>;

  /** Weekly origination funnel trends (KFI / Application / Offer / Completion). */
  getFunnelEvolution(portfolioId: string, portfolioContext?: string,
                    signal?: AbortSignal): Promise<PipelineFunnelEvolution>;

  /**
   * OPTIONAL week-on-week movement detail for one point on a weekly chart.
   *
   * Phase 2A, additive: this is the only client method whose endpoint may not
   * exist. It is absent on a deployment that has not enabled the hover layer,
   * and an implementation is free to reject rather than fabricate — callers
   * must treat a rejection as "no detail", never as an error worth surfacing.
   *
   * OPTIONAL on the interface on purpose: a client that does not implement it
   * is still a valid AgentClient, so removing this phase removes it cleanly and
   * no existing implementation or test double has to know it ever existed.
   */
  getMovementDetail?(portfolioId: string, detailType: MovementDetailType,
                   asOf?: string, portfolioContext?: string,
                   signal?: AbortSignal): Promise<MovementDetail>;

  /**
   * OPTIONAL Weekly Portfolio Brief (Phase 3A).
   *
   * Optional on the interface for the same reason as `getMovementDetail`: a
   * client that does not implement it is still a valid AgentClient, so the
   * phase can be removed cleanly and no existing test double had to change.
   */
  getWeeklyBrief?(portfolioId: string, portfolioContext?: string,
                 asOf?: string, signal?: AbortSignal): Promise<WeeklyBrief>;

  /** Governed risk-limit / concentration monitor (Schedule 8 vs funded actuals). */
  getRiskLimits(portfolioId: string, portfolioContext?: string,
               signal?: AbortSignal): Promise<RiskLimitsSnapshot>;

  /** Approved concentration tests evaluated against the funded book (one
   *  governed evaluation service; legacy extracted limits marked unapproved). */
  getConcentrationTests(portfolioId: string, portfolioContext?: string,
                        signal?: AbortSignal): Promise<ConcentrationTestsSnapshot>;

  /** Contributing-loan population for one approved test — the evaluator's own
   *  mask, reconciling exactly to the numerator. */
  getConcentrationDrillthrough(portfolioId: string, testId: string,
                               portfolioContext?: string,
                               signal?: AbortSignal): Promise<ConcentrationDrillthrough>;

  /** Metric history across real governed snapshots (never fabricated). */
  getConcentrationHistory(portfolioId: string, testId?: string,
                          portfolioContext?: string,
                          signal?: AbortSignal): Promise<ConcentrationHistory>;

  /** Pipeline cases driving one test's Expected Forecast movement (the
   *  forecast engine's own probabilities; reconciles to the expected
   *  numerator). */
  getConcentrationDrivers(portfolioId: string, testId: string,
                          portfolioContext?: string,
                          signal?: AbortSignal): Promise<ConcentrationDrivers>;

  /** Securitisation scale-up forecast (run-rate / KFI extrapolation + milestones). */
  getForecastExtrapolation(portfolioId: string, portfolioContext?: string,
                          signal?: AbortSignal): Promise<ForecastExtrapolation>;

  /** The authenticated caller (Entra principal echoed by the API), for the
   * header identity + role-based control visibility. */
  getMe(signal?: AbortSignal): Promise<UserIdentity>;

  /** Discover the investor PPTX decks published for a portfolio (latest + dated). */
  getDecks(portfolioId: string, signal?: AbortSignal): Promise<DeckIndex>;

  /** A direct download URL for an investor deck (latest, or a specific period),
   * or `null` when the client cannot serve decks (e.g. the mock). */
  deckDownloadUrl(portfolioId: string, period?: string | null): string | null;

  /** Fetch an investor deck as a blob, WITH the caller's credential attached.
   *
   *  Optional, and it exists because a bearer token cannot ride on a browser
   *  navigation: `<a href>` sends cookies but no Authorization header, so under
   *  Entra bearer auth the URL form above answers 401. Clients that can attach a
   *  credential implement this; the UI prefers it when the bearer path is on and
   *  keeps using the plain URL otherwise. */
  downloadDeck?(portfolioId: string, period?: string | null,
                signal?: AbortSignal): Promise<Blob>;

  /** Ask the API to generate an investor pack. Resolves as soon as the work is
   *  accepted — generation takes seconds — so the caller polls
   *  {@link getDeckGeneration}.
   *
   *  Declared as a nullable PROPERTY, not a method, so a client that cannot
   *  generate decks (the mock) says so by being `null`. The UI can then present
   *  an unavailable action up front instead of discovering the limitation only
   *  when a user clicks. */
  generateDeck?:
    | ((request: DeckGenerationRequest, signal?: AbortSignal) => Promise<DeckGenerationJob>)
    | null;

  /** The state of a requested generation. Optional alongside
   *  {@link generateDeck} — a client that cannot start one is never asked. */
  getDeckGeneration?(jobId: string, signal?: AbortSignal): Promise<DeckGenerationJob>;

  /** Funded static-pool cohort analysis for a portfolio, grouped by
   *  ``dimension`` (vintage | age | ltv | channel; default vintage). ``grain``
   *  (Y|Q|M) sets the vintage grain (vintage dimension only). */
  getCohorts(portfolioId: string, grain?: CohortGrain, dimension?: CohortDimension,
             portfolioContext?: string, signal?: AbortSignal): Promise<CohortAnalysis>;

  /** Static-pool cohort PROGRESSION across reporting periods for a cohort — a
   *  source-portfolio ``lens`` optionally narrowed to an origination ``vintage``
   *  at ``grain``. */
  getCohortProgression(portfolioId: string, query?: CohortProgressionQuery,
                       signal?: AbortSignal): Promise<CohortProgression>;

  /** Vintage FORMATION — what entered the book in each origination period —
   *  or, when ``vintage`` is supplied, that vintage's fixed static pool
   *  followed forward. Not the book outstanding at a reporting date; that is
   *  `getFundedEvolution`. */
  getCohortVintages(portfolioId: string, query?: CohortVintageQuery,
                    signal?: AbortSignal): Promise<CohortFormation | CohortStaticPool>;

  /** Funded exposure per UK ITL3 area — the Geography tab's choropleth feed. */
  getGeoExposure(portfolioId: string, portfolioContext?: string,
                signal?: AbortSignal): Promise<GeoExposure>;
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
