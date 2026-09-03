/**
 * useWorkspace — central orchestration hook.
 *
 * Owns the data-driven funded portfolio / reporting-run selection, the
 * deterministic landing-page funded snapshot, chat history, the artifact canvas,
 * and all agent interaction (through the AgentClient only). The portfolio and
 * reporting-date dropdowns are populated from `GET /mi/snapshots` (real
 * onboarding output) — never hardcoded prototype options. This is the seam a
 * backend plugs into: swap the client passed in, nothing else changes.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

/** Delay before a NOT-CURRENTLY-VISIBLE forecast prefetch is issued. */
export const SPECULATIVE_FORECAST_DELAY_MS = 2500;
import { setDisplayCurrency } from "@/lib/utils";
import type {
  AgentRequest,
  Artifact,
  ChatMessage,
  ForecastSnapshot,
  FundedSnapshot,
  PortfolioContext,
  PortfolioCapability,
  PortfolioCapabilityId,
  PortfolioContextIndex,
  PortfolioContextOption,
  ReportingContext,
  SnapshotPortfolio,
  SnapshotRun,
  SourcePortfolioLens,
  WorkspaceView,
} from "@/domain";
import { EMPTY_PORTFOLIO_CONTEXT } from "@/domain";
import type { AgentClient } from "@/api";
import type { UserIdentity } from "@/lib/identity";
import { formatHeading, uid } from "@/lib/utils";
import {
  type AnalysisContext,
  deriveContext,
  looksLikeFollowUp,
  resolveFollowUp,
} from "@/lib/analysisContext";
import { buildSuggestedActions } from "@/lib/suggestedActions";
import { presentAnswer } from "@/lib/responsePresenter";
import { loadState, readUrlContext, saveState } from "./persistence";

/** Friendly display name for a client/portfolio label ("acquired_001" → "Acquired 001"). */
function clientDisplayName(label?: string | null): string | null {
  if (!label) return null;
  const pretty = formatHeading(label.toLowerCase());
  return pretty || label;
}

function greeting(portfolioLabel: string | null, asOf: string | null): ChatMessage {
  const name = clientDisplayName(portfolioLabel);
  const date = asOf
    ? new Date(asOf).toLocaleDateString("en-GB", { day: "numeric", month: "long", year: "numeric" })
    : "the latest reporting date";
  return {
    id: "greeting",
    role: "assistant",
    content: `Welcome back. I'm your MI Agent for the ${name ? `${name} portfolio` : "selected portfolio"}, as of ${date}. Ask me about portfolio movement, concentration, stratifications, risk monitoring or validation.`,
    createdAt: new Date().toISOString(),
  };
}

/** Stamp the originating question onto each artifact so a drill-through can
 *  re-run the same query with an added filter. */
function stampQuestion(artifacts: Artifact[], question: string): Artifact[] {
  return artifacts.map((a) => ({ ...a, source: { ...a.source, question } }));
}

/** Ceiling on unpinned artifacts kept in the workspace (localStorage-persisted;
 *  the oldest fall off the end once the session accumulates past it). */
const MAX_UNPINNED_ARTIFACTS = 24;

/** The logical-artifact identity the canvas groups by (title + type). */
function artifactKey(a: Artifact): string {
  return `${formatHeading(a.title).toLowerCase()}|${a.type}`;
}

/**
 * Accumulate fresh artifacts instead of replacing the workspace wholesale:
 * pinned artifacts stay first, fresh results come next (newest first), and
 * prior unpinned results are RETAINED so earlier answers stay comparable.
 * A fresh artifact with the same title + type as an older unpinned one
 * supersedes it (re-asking or drilling refreshes in place rather than
 * stacking duplicates into one grouped card).
 */
export function mergeArtifacts(prev: Artifact[], fresh: Artifact[]): Artifact[] {
  const pinned = prev.filter((a) => a.pinned);
  const pinnedIds = new Set(pinned.map((a) => a.id));
  const incoming = fresh.filter((a) => !pinnedIds.has(a.id));
  const incomingIds = new Set(incoming.map((a) => a.id));
  const incomingKeys = new Set(incoming.map(artifactKey));
  const kept = prev.filter(
    (a) => !a.pinned && !incomingIds.has(a.id) && !incomingKeys.has(artifactKey(a)),
  );
  return [...pinned, ...[...incoming, ...kept].slice(0, MAX_UNPINNED_ARTIFACTS)];
}

/** True when the failure is a caller-initiated cancellation, not an error. */
function isAbortError(err: unknown): boolean {
  return err instanceof Error && err.name === "AbortError";
}

export interface Workspace {
  /** Discovered funded portfolios (data-driven dropdown source). */
  portfolios: SnapshotPortfolio[];
  /** Why portfolio discovery failed (null when it succeeded) — surfaces an
   *  unreachable/unauthorised API instead of a silent "No client" header. */
  portfoliosError: string | null;
  /** Reporting runs available for the selected portfolio. */
  runs: SnapshotRun[];
  selectedClientId: string | null;
  selectedRunId: string | null;
  portfolio: PortfolioContext;
  reporting: ReportingContext;
  /** Deterministic landing-page funded snapshot for the selected run. */
  snapshot: FundedSnapshot | null;
  snapshotLoading: boolean;
  /** Why the snapshot fetch failed (null when it succeeded / is loading) — a
   *  failed load must render as a visible error, never as a blank panel. */
  snapshotError: string | null;
  /** Deterministic funded + pipeline forecast snapshot for the selected run. */
  forecast: ForecastSnapshot | null;
  forecastLoading: boolean;
  /** Why the forecast fetch failed (null when it succeeded / is loading). */
  forecastError: string | null;
  /** Active workspace view (funded | pipeline | forecast). */
  activeView: WorkspaceView;
  setActiveView: (view: WorkspaceView) => void;
  /** @deprecated Superseded by `portfolioContexts`; kept so any existing
   *  consumer of the flat lens list keeps compiling. */
  sourceLenses: SourcePortfolioLens[];
  /** THE governed portfolio contract for this workspace (hierarchy + per-context
   *  capabilities), exactly as the backend resolved it. */
  portfolioContextIndex: PortfolioContextIndex;
  /** The hierarchical selector options (Total → type groups → portfolios). */
  portfolioContexts: PortfolioContextOption[];
  /** The single selected portfolio context id controlling the whole workspace. */
  selectedContextId: string;
  setSelectedContextId: (contextId: string) => void;
  /** The selected context's node in the hierarchy (null before it loads). */
  activeContext: PortfolioContextOption | null;
  /** Backend-resolved capabilities for the selected context. */
  capabilities: Record<string, PortfolioCapability>;
  /** Is one governed capability enabled for the current context? */
  capabilityEnabled: (id: PortfolioCapabilityId) => boolean;
  /** The backend's business explanation for a capability, for the UI to show. */
  capabilityReason: (id: PortfolioCapabilityId) => string | null;
  /** Views the backend says do not apply to the selected context. */
  disabledViews: WorkspaceView[];
  /** Why each disabled view is unavailable, keyed by view. */
  disabledViewReasons: Record<string, string>;
  messages: ChatMessage[];
  artifacts: Artifact[];
  isWorking: boolean;
  /** Question carried on a Copilot "Open in Workspace" deep link, to pre-fill
   *  the composer once (empty string when the app was opened normally). */
  initialQuestion: string;
  setPortfolio: (clientId: string) => void;
  setRun: (runId: string) => void;
  ask: (question: string) => void;
  /** Cancel the in-flight agent request (no-op when nothing is running). */
  stop: () => void;
  /** Re-run an artifact's query with an added drill-through filter (backend). */
  drill: (artifact: Artifact, filters: Record<string, unknown>) => void;
  /** Spec-level memory of the last successful query (null when inactive). */
  context: AnalysisContext | null;
  /** Clear the analysis context (back to standalone-question behaviour). */
  clearContext: () => void;
  /** Declutter controls (view-only resets; never destroy loaded MI data). */
  clearArtifacts: () => void;
  clearChat: () => void;
  clearAll: () => void;
  retryLast: () => void;
  togglePin: (id: string) => void;
  resetWorkspace: () => void;
  /** The signed-in caller (for the header identity + role gating). */
  identity: UserIdentity | null;
  /** Bumps on manual refresh; panels key off it to reload after a cache clear. */
  dataVersion: number;
  refreshing: boolean;
  /** Clear the client cache and reload the active MI data. */
  refresh: () => void;
}

export function useWorkspace(client: AgentClient): Workspace {
  const persisted = useRef(loadState()).current;
  // Copilot "Open in Workspace" deep-link context (client, run, question,
  // filters). When present it takes precedence over persisted selection so the
  // link restores context; otherwise the existing localStorage flow is used.
  const urlContext = useRef(readUrlContext()).current;

  const [portfolios, setPortfolios] = useState<SnapshotPortfolio[]>([]);
  const [portfoliosError, setPortfoliosError] = useState<string | null>(null);
  const [selectedClientId, setSelectedClientId] = useState<string | null>(
    urlContext.clientId ?? persisted?.clientId ?? null);
  const [selectedRunId, setSelectedRunId] = useState<string | null>(
    urlContext.runId ?? persisted?.runId ?? null);
  const [snapshot, setSnapshot] = useState<FundedSnapshot | null>(null);
  const [snapshotLoading, setSnapshotLoading] = useState(false);
  const [snapshotError, setSnapshotError] = useState<string | null>(null);
  const [forecast, setForecast] = useState<ForecastSnapshot | null>(null);
  const [forecastLoading, setForecastLoading] = useState(false);
  const [forecastError, setForecastError] = useState<string | null>(null);
  // Signed-in caller (header identity + role gating) and manual-refresh plumbing.
  const [identity, setIdentity] = useState<UserIdentity | null>(null);
  const [dataVersion, setDataVersion] = useState(0);
  const [refreshing, setRefreshing] = useState(false);
  // Active view defaults to Funded. A ref mirrors it so runQuery reads the latest
  // value without being re-created on every view change.
  const [activeView, setActiveViewState] = useState<WorkspaceView>("funded");
  const activeViewRef = useRef<WorkspaceView>("funded");
  const setActiveView = useCallback((view: WorkspaceView) => {
    activeViewRef.current = view;
    setActiveViewState(view);
  }, []);

  // ---- THE governed portfolio context --------------------------------------
  // ONE selection controls the entire workspace: KPI cards, funded charts, risk,
  // evolution, cohorts, pipeline, forecast, drill-through, chat and exports all
  // send this same id to the backend. There is deliberately no second portfolio
  // control and no client-side filtering — the backend resolves what the context
  // means and what it may do.
  const [portfolioContextIndex, setPortfolioContextIndex] =
    useState<PortfolioContextIndex>(EMPTY_PORTFOLIO_CONTEXT);
  const [selectedContextId, setSelectedContextIdState] = useState<string>("total");
  // A ref mirrors the selection so runQuery/drill read the latest value without
  // being re-created on every change.
  const selectedContextRef = useRef<string>("total");

  /** The individual-portfolio context matching an onboarded client id, if any
   *  (deployments that onboard each book as its own client have both axes). */
  const contextForClient = useCallback(
    (clientId: string): PortfolioContextOption | null =>
      (portfolioContextIndex.contexts ?? []).find(
        (c) => c.context_kind === "portfolio"
          && (c.context_id === clientId
            || (c.portfolio_ids.length === 1 && c.portfolio_ids[0] === clientId)),
      ) ?? null,
    [portfolioContextIndex],
  );

  const setSelectedContextId = useCallback((contextId: string) => {
    selectedContextRef.current = contextId;
    setSelectedContextIdState(contextId);
    // CLIENT FOLLOWS SCOPE: drilling the scope into an individual portfolio
    // that is itself an onboarded client switches the workspace to that
    // client's runs, so the scope shown and the tape being read can never
    // contradict each other (e.g. "Client: acquired_001 · Scope: Total").
    const ctx = (portfolioContextIndex.contexts ?? []).find((c) => c.context_id === contextId);
    if (ctx?.context_kind === "portfolio") {
      const match = portfolios.find(
        (p) => p.client_id === contextId
          || (ctx.portfolio_ids.length === 1 && p.client_id === ctx.portfolio_ids[0]),
      );
      if (match) {
        setSelectedClientId((cur) => (cur === match.client_id ? cur : match.client_id));
        setSelectedRunId((cur) => {
          const stillValid = match.runs.some((r) => r.run_id === cur);
          return stillValid ? cur : (match.runs[match.runs.length - 1]?.run_id ?? null);
        });
      }
    }
  }, [portfolioContextIndex, portfolios]);

  // Resolve the signed-in caller once (drives the header identity + role gating).
  // Degrades silently to null when /me is unreachable (auth removed for testing).
  useEffect(() => {
    let cancelled = false;
    client
      .getMe()
      .then((me) => { if (!cancelled) setIdentity(me); })
      .catch(() => { if (!cancelled) setIdentity(null); });
    return () => { cancelled = true; };
  }, [client]);

  // The governed hierarchy + capability matrix. Everything the selector offers
  // and everything the navigation gates on comes from this one response.
  useEffect(() => {
    let cancelled = false;
    client
      .getPortfolioContext()
      .then((idx) => {
        if (cancelled) return;
        setPortfolioContextIndex(idx ?? EMPTY_PORTFOLIO_CONTEXT);
      })
      .catch(() => {
        if (!cancelled) setPortfolioContextIndex(EMPTY_PORTFOLIO_CONTEXT);
      });
    return () => {
      cancelled = true;
    };
  }, [client]);

  const portfolioContexts = portfolioContextIndex.contexts ?? [];

  // The flat lens list stays available for backward compatibility only; it is
  // derived from the governed hierarchy rather than fetched separately, so there
  // is exactly one source of portfolio truth in the app.
  const sourceLenses = useMemo<SourcePortfolioLens[]>(
    () => portfolioContexts.map((c) => ({
      id: c.context_id,
      kind: c.context_kind === "portfolio" ? "cohort" : c.context_kind,
      label: c.label,
      filters: {},
      funded_only: !(c.capabilities?.pipeline?.enabled ?? true),
      source_portfolio_type: c.portfolio_types[0] ?? null,
    })),
    [portfolioContexts],
  );

  const activeContext = useMemo(
    () => portfolioContexts.find((c) => c.context_id === selectedContextId) ?? null,
    [portfolioContexts, selectedContextId],
  );
  const capabilities = useMemo(
    () => (activeContext?.capabilities ?? {}) as Record<string, PortfolioCapability>,
    [activeContext],
  );
  // A capability with no backend entry is treated as ENABLED so a deployment
  // whose backend predates the contract keeps its existing navigation; a
  // capability the backend explicitly reports is always obeyed.
  const capabilityEnabled = useCallback(
    (id: PortfolioCapabilityId) => capabilities[id]?.enabled ?? true,
    [capabilities],
  );
  const capabilityReason = useCallback(
    (id: PortfolioCapabilityId) => capabilities[id]?.detail ?? null,
    [capabilities],
  );

  // Navigation follows the backend's capability decision — the UI never decides
  // that "acquired means no pipeline"; it renders what the resolver returned.
  const { disabledViews, disabledViewReasons } = useMemo(() => {
    const off: WorkspaceView[] = [];
    const reasons: Record<string, string> = {};
    const gate = (view: WorkspaceView, id: PortfolioCapabilityId) => {
      const state = capabilities[id];
      if (state && !state.enabled) {
        off.push(view);
        reasons[view] = state.detail
          ?? "This analysis does not apply to the selected portfolio scope.";
      }
    };
    gate("pipeline", "pipeline");
    gate("forecast", "consolidated_forecast");
    gate("risk_limits", "risk");
    return { disabledViews: off, disabledViewReasons: reasons };
  }, [capabilities]);

  // A disabled view is NOT silently swapped for another: switching the workspace
  // out from under the user would hide the fact that the analysis is unavailable
  // for the scope they chose. The tab is disabled with a reason, the selected
  // context is preserved, and the panel area explains why.

  // Discover available portfolios / runs once; default to the latest run.
  useEffect(() => {
    let cancelled = false;
    client
      .getSnapshots()
      .then((index) => {
        if (cancelled) return;
        const pfs = index.portfolios ?? [];
        setPortfolios(pfs);
        setPortfoliosError(null);
        // Diagnostic: what the store received + how it auto-selected (visible in
        // the browser console on the live app).
        console.debug("[useWorkspace] /mi/snapshots", {
          count: pfs.length,
          portfolioIds: pfs.map((p) => p.client_id),
          runsPerPortfolio: pfs.map((p) => p.runs?.length ?? 0),
        });
        if (pfs.length === 0) return;
        // A Workspace deep link's client/run wins over persisted selection.
        const preferredClientId = urlContext.clientId ?? persisted?.clientId;
        const preferredRunId = urlContext.runId ?? persisted?.runId;
        const preferredPf = pfs.find((p) => p.client_id === preferredClientId);
        const pf = pfs.length === 1 ? pfs[0] : (preferredPf ?? pfs[pfs.length - 1]);
        const preferredRun = pf.runs.find((r) => r.run_id === preferredRunId);
        const run = preferredRun ?? pf.runs[pf.runs.length - 1]; // latest reporting date
        if (pfs.length === 1) {
          // Exactly one funded portfolio → auto-select it. Honour a deep-link run
          // when it exists, else the latest reporting date (dropping any stale
          // persisted selection from another deployment).
          setSelectedClientId(pf.client_id);
          setSelectedRunId(run?.run_id ?? null);
        } else {
          setSelectedClientId((cur) => cur ?? pf.client_id);
          setSelectedRunId((cur) => cur ?? run?.run_id ?? null);
        }
        console.debug("[useWorkspace] auto-selected", {
          clientId: pf.client_id, runId: run?.run_id ?? null,
          reportingDate: run?.reporting_date ?? null,
        });
      })
      .catch((err) => {
        console.warn("[useWorkspace] /mi/snapshots failed", err);
        if (cancelled) return;
        setPortfolios([]);
        setPortfoliosError(err instanceof Error ? err.message
          : "The MI Agent API could not be reached.");
      });
    return () => {
      cancelled = true;
    };
  }, [client, persisted?.clientId, persisted?.runId,
      urlContext.clientId, urlContext.runId, dataVersion]);

  const activePortfolio = useMemo(
    () => portfolios.find((p) => p.client_id === selectedClientId) ?? portfolios[0] ?? null,
    [portfolios, selectedClientId],
  );
  const runs = activePortfolio?.runs ?? [];
  const activeRun = useMemo(
    () => runs.find((r) => r.run_id === selectedRunId) ?? runs[runs.length - 1] ?? null,
    [runs, selectedRunId],
  );

  const portfolioId =
    activePortfolio && activeRun ? `${activePortfolio.client_id}/${activeRun.run_id}` : "";

  const portfolio = useMemo<PortfolioContext>(
    () => ({
      id: portfolioId,
      name: activePortfolio?.label ?? "Funded Portfolio",
      entity: activeRun?.reporting_date ?? activeRun?.run_id ?? "",
    }),
    [portfolioId, activePortfolio?.label, activeRun?.reporting_date, activeRun?.run_id],
  );
  const reporting = useMemo<ReportingContext>(
    () => ({ asOf: activeRun?.reporting_date ?? "", comparedTo: undefined }),
    [activeRun?.reporting_date],
  );

  // Fetch the deterministic funded snapshot whenever the selected run changes.
  useEffect(() => {
    if (!portfolioId) return;
    let cancelled = false;
    setSnapshotLoading(true);
    client
      .getSnapshot(portfolioId, selectedContextId)
      .then((snap) => {
        if (cancelled) return;
        // The reporting currency is governed by the client configuration and
        // stated by the API. Adopt it before any money is rendered.
        setDisplayCurrency(snap?.currencyCode);
        setSnapshot(snap);
        setSnapshotError(null);
      })
      .catch((err: unknown) => {
        if (cancelled) return;
        // A failed load must be VISIBLE — keep the reason for the error panel.
        setSnapshot(null);
        setSnapshotError(err instanceof Error ? err.message
          : "The funded snapshot could not be loaded.");
      })
      .finally(() => {
        if (!cancelled) setSnapshotLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [client, portfolioId, selectedContextId, dataVersion]);

  // Fetch the deterministic forecast (funded + pipeline) for the same run, so the
  // pipeline + forecast + watchlist sections move with the funded selection.
  // PRIORITISED: unless a pipeline/forecast view needs it right now, the fetch
  // waits until the visible funded snapshot has settled — the slow backend
  // compute serves what's on screen first, then warms the rest.
  //: How long a forecast prefetch waits when its tab is not open. Long enough
  //: to clear a page-load burst, short enough that a user who reaches for the
  //: Pipeline tab still finds it warm. Not a tuning knob anybody has measured
  //: an optimum for -- it is the smallest delay that takes the request out of
  //: the burst window that failed.
  const forecastKeyRef = useRef<string | null>(null);
  //: A prefetch waiting out its delay. Held so that a user who opens the tab
  //: mid-delay cancels it and is served immediately.
  const deferredForecastRef = useRef<{ key: string; timer: ReturnType<typeof setTimeout> } | null>(null);
  const workspaceUnmounted = useRef(false);
  useEffect(() => () => { workspaceUnmounted.current = true; }, []);
  useEffect(() => {
    if (!portfolioId) return;
    const needNow = activeView === "pipeline" || activeView === "forecast";
    const key = `${portfolioId}|${selectedContextId}|${dataVersion}`;
    if (forecastKeyRef.current === key) return; // already fetched for this selection
    if (!needNow && snapshotLoading) return;    // let the visible panel land first
    // Loading is set SYNCHRONOUSLY even when the request is deferred below, so
    // the panel shows the state it shows today rather than briefly reading as
    // "no forecast" during the delay.
    setForecastLoading(true);

    const fire = () => {
      deferredForecastRef.current = null;
      // Claimed HERE, not before the delay. Setting it up front made the
      // "already fetched for this selection" guard above swallow the
      // immediate fetch when a user opened the Pipeline tab mid-delay, so
      // they waited out the remainder of a prefetch instead of being served
      // at once -- a deferral turned into a stall for the one person it was
      // supposed to be invisible to.
      forecastKeyRef.current = key;
    // Resolution is guarded by the KEY (not an effect-cleanup flag): dep churn
    // like snapshotLoading flipping must not cancel the in-flight fetch, and a
    // late response for a superseded selection is simply ignored.
    client
      .getForecastSnapshot(portfolioId, selectedContextId)
      .then((fc) => {
        if (workspaceUnmounted.current || forecastKeyRef.current !== key) return;
        setForecast(fc);
        setForecastError(null);
        setForecastLoading(false);
      })
      .catch((err: unknown) => {
        if (workspaceUnmounted.current || forecastKeyRef.current !== key) return;
        forecastKeyRef.current = null; // a failed fetch stays retryable
        setForecast(null);
        setForecastError(err instanceof Error ? err.message
          : "The pipeline / forecast snapshot could not be loaded.");
        setForecastLoading(false);
      });
    };

    // ON SCREEN NOW: fetch immediately, exactly as before — and cancel any
    // prefetch already waiting, so the user's own request replaces the
    // speculative one rather than queueing behind it.
    if (needNow) {
      if (deferredForecastRef.current) {
        clearTimeout(deferredForecastRef.current.timer);
        deferredForecastRef.current = null;
      }
      fire();
      return;
    }

    if (deferredForecastRef.current?.key === key) return;  // already scheduled

    // NOT ON SCREEN: this is SPECULATION — a prefetch so that opening the
    // Pipeline or Forecast tab is instant. Worth doing, but not worth doing
    // inside the page-load burst.
    //
    // MEASURED 2026-09-03, six users opening the dashboard together on a B2:
    // 18 of 54 requests returned 500. The three that failed were the three
    // heaviest, each pinned at ~45.4s against the auth layer's ~46s ceiling --
    // and `forecast/snapshot` was one of them, fetched for a tab nobody had
    // opened. The weekly brief then waits on the forecast and fires too, so
    // one speculative request was pulling a second in behind it.
    //
    // Six users' speculation competing with six users' actual requests is the
    // whole of that failure. Deferring it past the burst costs a tab switch
    // within the delay window one wait it would otherwise have avoided; it
    // costs a crowded morning nothing at all. The same reasoning already
    // narrowed the other-client warm below from three clients to one.
    const timer = setTimeout(fire, SPECULATIVE_FORECAST_DELAY_MS);
    deferredForecastRef.current = { key, timer };
    return () => {
      clearTimeout(timer);
      if (deferredForecastRef.current?.timer === timer) {
        deferredForecastRef.current = null;
      }
    };
  }, [client, portfolioId, selectedContextId, dataVersion, activeView, snapshotLoading]);

  // Background-warm the OTHER clients' latest runs once the active snapshot has
  // settled, so switching client hits the session cache instead of waiting out
  // a full backend snapshot compute. Sequential, bounded and best-effort — a
  // failure only means the switch falls back to a live fetch as before.
  //
  // NARROWED to ONE other client (was three). These are the two most expensive
  // endpoints in the app, and the API serves them from a small worker pool, so
  // speculative warming competed with whatever the user was actually waiting
  // for. Server-side caching now makes a cold client switch far cheaper, which
  // is what this loop existed to hide.
  useEffect(() => {
    if (!portfolioId || snapshotLoading || portfolios.length <= 1) return;
    let cancelled = false;
    const others = portfolios.filter((p) => p.client_id !== selectedClientId).slice(0, 1);
    if (others.length === 0) return;
    const timer = setTimeout(async () => {
      for (const p of others) {
        const run = p.runs[p.runs.length - 1];
        if (!run || cancelled) return;
        const pid = `${p.client_id}/${run.run_id}`;
        // Warm with the scope that selecting this client would sync to.
        const scope = contextForClient(p.client_id)?.context_id ?? selectedContextRef.current;
        try { await client.getSnapshot(pid, scope); } catch { /* best-effort */ }
        if (cancelled) return;
        try { await client.getForecastSnapshot(pid, scope); } catch { /* best-effort */ }
      }
    }, 3500);
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [client, portfolioId, snapshotLoading, portfolios, selectedClientId, contextForClient]);

  // Manual refresh: clear the client cache (if it exposes one) and bump the
  // data version so the snapshot/forecast effects and the keyed panels reload.
  const refresh = useCallback(() => {
    const c = client as AgentClient & { invalidate?: () => void };
    if (typeof c.invalidate === "function") c.invalidate();
    setRefreshing(true);
    setDataVersion((v) => v + 1);
    // The spinner is a brief acknowledgement; the effects clear their own loading.
    setTimeout(() => setRefreshing(false), 600);
  }, [client]);

  const [messages, setMessages] = useState<ChatMessage[]>(
    () => persisted?.messages ?? [greeting(null, null)],
  );
  const [artifacts, setArtifacts] = useState<Artifact[]>(
    () => (persisted?.artifacts ?? []).filter((a) => !a.mock),
  );
  const [isWorking, setIsWorking] = useState(false);

  // Analysis context: spec-level memory of the last successful query. A ref
  // mirrors it so `ask` reads the latest value without being re-created.
  const [context, setContextState] = useState<AnalysisContext | null>(null);
  const contextRef = useRef<AnalysisContext | null>(null);
  const setContext = useCallback((ctx: AnalysisContext | null) => {
    contextRef.current = ctx;
    setContextState(ctx);
  }, []);
  const clearContext = useCallback(() => setContext(null), [setContext]);

  // Declutter controls (A8). Clearing artifacts/chat is a VIEW reset only — it
  // never touches the loaded portfolio/run data or the active dataset.
  const clearArtifacts = useCallback(() => setArtifacts([]), []);
  const clearChat = useCallback(() => {
    setMessages([greeting(activePortfolio?.label ?? null, activeRun?.reporting_date ?? null)]);
    setArtifacts((prev) => prev);  // keep workspace artifacts; chat is independent
  }, [activePortfolio?.label, activeRun?.reporting_date]);
  const clearAll = useCallback(() => {
    setArtifacts([]);
    setMessages([greeting(activePortfolio?.label ?? null, activeRun?.reporting_date ?? null)]);
    setContext(null);
  }, [setContext, activePortfolio?.label, activeRun?.reporting_date]);

  // Personalise the greeting once the portfolio/run resolves — only while the
  // chat is pristine (nothing but the greeting), so history is never rewritten.
  useEffect(() => {
    setMessages((prev) => {
      if (prev.length !== 1 || prev[0].id !== "greeting") return prev;
      const fresh = greeting(activePortfolio?.label ?? null, activeRun?.reporting_date ?? null);
      return prev[0].content === fresh.content ? prev : [fresh];
    });
  }, [activePortfolio?.label, activeRun?.reporting_date]);

  const lastQuestion = useRef<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  // Switching portfolio / reporting run invalidates the prior analysis context.
  useEffect(() => {
    setContext(null);
  }, [portfolioId, selectedContextId, setContext]);

  useEffect(() => {
    saveState({
      clientId: selectedClientId ?? undefined,
      runId: selectedRunId ?? undefined,
      // Strip the inline result artifacts from each message before persisting —
      // they are kept in memory for the conversation and re-derivable from the
      // canvas; persisting them would bloat localStorage.
      messages: messages.map((m) => (m.artifacts ? { ...m, artifacts: undefined } : m)),
      artifacts,
    });
  }, [selectedClientId, selectedRunId, messages, artifacts]);

  const runQuery = useCallback(
    (
      params: {
        /** Raw text the user typed (drives context + retry). */
        display: string;
        /** Question actually sent (may be a context-resolved rewrite). */
        send: string;
        filters?: Record<string, unknown>;
        usedContext?: boolean;
        contextNote?: string;
        /** Retry once with the raw question if a resolved query fails. */
        allowRawFallback?: boolean;
      },
      pendingId: string,
    ) => {
      const request: AgentRequest = {
        question: params.send,
        portfolio,
        reporting,
        options: { parserMode: "deterministic" },
        // Tab-aware: the active view sets the dataset context (funded / pipeline
        // / forecast); explicit wording in the question overrides it backend-side.
        datasetContext: activeViewRef.current,
        filters: params.filters,
        // Source-portfolio scope (the dropdown default); a portfolio named in
        // the question overrides it backend-side.
        sourceLens: selectedContextRef.current,
      };
      const controller = new AbortController();
      abortRef.current = controller;
      setIsWorking(true);

      client
        .ask(request, controller.signal)
        .then((res) => {
          // A context-resolved query that failed falls back ONCE to the raw
          // question (the existing flow) so context never breaks a request.
          if (!res.ok && params.allowRawFallback && params.send !== params.display) {
            runQuery({ display: params.display, send: params.display, allowRawFallback: false }, pendingId);
            return;
          }
          const primary = res.artifacts.find((a) => a.type === "chart" || a.type === "table");
          const suggestions = res.ok && primary ? buildSuggestedActions(res.spec, primary) : undefined;
          const stampedArtifacts = stampQuestion(res.artifacts, params.send);
          const content = presentAnswer({
            question: params.display,
            ok: res.ok,
            spec: res.spec,
            artifacts: res.artifacts,
            narrative: res.narrative,
            error: res.error,
          });
          // Accumulate: fresh results land first, prior answers stay for
          // comparison (same title + type refreshes in place, never duplicates).
          setArtifacts((prev) => mergeArtifacts(prev, stampedArtifacts));
          setMessages((prev) =>
            prev.map((m) =>
              m.id === pendingId
                ? {
                    ...m,
                    pending: false,
                    error: !res.ok && res.artifacts.length === 0,
                    content,
                    artifacts: stampedArtifacts,
                    interpreted: res.interpreted,
                    assumptions: res.assumptions,
                    warnings: res.warnings,
                    diagnostics: res.diagnostics,
                    intent: res.intent,
                    // Only surface the book that answered when it DIFFERS from the
                    // active tab — i.e. the question's wording routed it to another
                    // book (funded tab -> a pipeline answer). When it matches the
                    // tab there is nothing surprising to flag, so no badge.
                    datasetContext:
                      res.datasetContext && res.datasetContext !== request.datasetContext
                        ? res.datasetContext
                        : undefined,
                    spec: res.spec,
                    confidence: res.confidence,
                    usedContext: params.usedContext,
                    contextNote: params.usedContext ? params.contextNote : undefined,
                    cacheHit: res.cacheHit,
                    // Backend-authored coverage disclosure — rendered as-is.
                    portfolioCoverage: res.portfolioCoverage,
                    suggestions,
                    artifactRefs: res.artifacts.map((a) => ({ id: a.id, title: a.title, type: a.type })),
                  }
                : m,
            ),
          );
          // Update analysis context ONLY after a successful query.
          if (res.ok) {
            setContext(
              deriveContext({
                spec: res.spec,
                question: params.display,
                artifactId: primary?.id,
                portfolioId: portfolio.id,
                now: new Date().toISOString(),
              }),
            );
          }
        })
        .catch((err: unknown) => {
          const message = isAbortError(err)
            ? "Request cancelled."
            : err instanceof Error ? err.message : "The MI Agent could not complete this request.";
          setMessages((prev) =>
            prev.map((m) => (m.id === pendingId ? { ...m, pending: false, error: true, content: message } : m)),
          );
        })
        .finally(() => {
          setIsWorking(false);
          abortRef.current = null;
        });
    },
    [client, portfolio, reporting, setContext],
  );

  // Backend drill-through: re-run the artifact's originating query with an added
  // filter so the result is computed from the FULL dataset (not just the rows on
  // screen). Refreshes the unpinned artifacts in place; on any failure the
  // current artifacts (and the client-side drill panel) are left untouched.
  const drill = useCallback(
    (artifact: Artifact, filters: Record<string, unknown>) => {
      const question = artifact.source.question ?? lastQuestion.current;
      if (!question || isWorking) return;
      const request: AgentRequest = {
        question,
        portfolio,
        reporting,
        options: { parserMode: "deterministic" },
        datasetContext: activeViewRef.current,
        filters,
        sourceLens: selectedContextRef.current,
      };
      const controller = new AbortController();
      abortRef.current = controller;
      setIsWorking(true);
      client
        .ask(request, controller.signal)
        .then((res) => {
          if (!res.ok) return; // keep current artifacts; client-side panel remains
          // Stamp the active drill filters so the card can show removable
          // filter chips (the breadcrumb back to the unfiltered result).
          const active = Object.keys(filters).length > 0 ? filters : undefined;
          const fresh = stampQuestion(res.artifacts, question).map((a) => ({
            ...a,
            drillFilters: active,
          }));
          setArtifacts((prev) => mergeArtifacts(prev, fresh));
        })
        .catch(() => {
          /* leave artifacts untouched — the client-side drill fallback stays */
        })
        .finally(() => {
          setIsWorking(false);
          abortRef.current = null;
        });
    },
    [client, portfolio, reporting, isWorking],
  );

  const ask = useCallback(
    (question: string) => {
      const text = question.trim();
      if (!text || isWorking) return;
      lastQuestion.current = text;

      // Lightweight, guarded follow-up resolution: only when context is active
      // AND the prompt looks like a follow-up. Any failure → send the raw text.
      let send = text;
      let filters: Record<string, unknown> | undefined;
      let usedContext = false;
      let contextNote: string | undefined;
      try {
        const ctx = contextRef.current;
        if (ctx && looksLikeFollowUp(text, ctx)) {
          const resolved = resolveFollowUp(text, ctx);
          if (resolved) {
            send = resolved.question;
            filters = resolved.filters;
            usedContext = true;
            contextNote = resolved.note;
          }
        }
      } catch {
        send = text; // resolution failed → fall back to the raw question
      }

      const userMsg: ChatMessage = {
        id: uid("msg"),
        role: "user",
        content: text,
        createdAt: new Date().toISOString(),
      };
      const pendingId = uid("msg");
      const pendingMsg: ChatMessage = {
        id: pendingId,
        role: "assistant",
        content: "",
        pending: true,
        createdAt: new Date().toISOString(),
      };
      setMessages((prev) => [...prev, userMsg, pendingMsg]);
      runQuery({ display: text, send, filters, usedContext, contextNote, allowRawFallback: usedContext }, pendingId);
    },
    [isWorking, runQuery],
  );

  // Cancel the in-flight request. The abort surfaces through the query's own
  // catch path, which marks the pending message "Request cancelled." with Retry.
  const stop = useCallback(() => {
    abortRef.current?.abort();
  }, []);

  const retryLast = useCallback(() => {
    if (isWorking || !lastQuestion.current) return;
    const pendingId = uid("msg");
    setMessages((prev) => [
      ...prev,
      { id: pendingId, role: "assistant", content: "", pending: true, createdAt: new Date().toISOString() },
    ]);
    runQuery({ display: lastQuestion.current, send: lastQuestion.current }, pendingId);
  }, [isWorking, runQuery]);

  const togglePin = useCallback((id: string) => {
    setArtifacts((prev) => prev.map((a) => (a.id === id ? { ...a, pinned: !a.pinned } : a)));
  }, []);

  const setPortfolio = useCallback(
    (clientId: string) => {
      setSelectedClientId(clientId);
      // Default to the latest run of the newly-selected portfolio.
      const pf = portfolios.find((p) => p.client_id === clientId);
      setSelectedRunId(pf?.runs[pf.runs.length - 1]?.run_id ?? null);
      // SCOPE FOLLOWS CLIENT: when the chosen client is itself a governed
      // source portfolio, align the scope so the capability gating (pipeline /
      // forecast / risk) reflects the book actually being read — an acquired
      // book immediately disables Pipeline + Forecast with the backend's reason.
      const ctx = contextForClient(clientId);
      if (ctx && selectedContextRef.current !== ctx.context_id) {
        selectedContextRef.current = ctx.context_id;
        setSelectedContextIdState(ctx.context_id);
      }
    },
    [portfolios, contextForClient],
  );

  const resetWorkspace = useCallback(() => {
    abortRef.current?.abort();
    setMessages([greeting(activePortfolio?.label ?? "selected", activeRun?.reporting_date ?? null)]);
    setArtifacts([]);
    setContext(null);
  }, [activePortfolio?.label, activeRun?.reporting_date, setContext]);

  return {
    portfolios,
    portfoliosError,
    runs,
    selectedClientId: activePortfolio?.client_id ?? null,
    selectedRunId: activeRun?.run_id ?? null,
    portfolio,
    reporting,
    snapshot,
    snapshotLoading,
    snapshotError,
    forecast,
    forecastLoading,
    forecastError,
    activeView,
    setActiveView,
    sourceLenses,
    portfolioContextIndex,
    portfolioContexts,
    selectedContextId,
    setSelectedContextId,
    activeContext,
    capabilities,
    capabilityEnabled,
    capabilityReason,
    disabledViews,
    disabledViewReasons,
    messages,
    artifacts,
    isWorking,
    initialQuestion: urlContext.question ?? "",
    setPortfolio,
    setRun: setSelectedRunId,
    ask,
    stop,
    drill,
    context,
    clearContext,
    clearArtifacts,
    clearChat,
    clearAll,
    retryLast,
    togglePin,
    resetWorkspace,
    identity,
    dataVersion,
    refreshing,
    refresh,
  };
}
