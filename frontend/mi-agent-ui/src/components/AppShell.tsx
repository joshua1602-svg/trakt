import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { ChevronDown, LayoutDashboard } from "lucide-react";
import { HeaderBar } from "@/components/HeaderBar";
import { AgentChatPanel } from "@/components/AgentChatPanel";
import { ArtifactCanvas } from "@/components/ArtifactCanvas";
import { FundedSnapshotPanel } from "@/components/FundedSnapshotPanel";
import { PipelineSnapshotPanel } from "@/components/PipelineSnapshotPanel";
import { PipelineWatchlist } from "@/components/PipelineWatchlist";
import { ForecastView } from "@/components/ForecastView";
import { ForecastExtrapolationPanel } from "@/components/ForecastExtrapolationPanel";
import { EvolutionPanel } from "@/components/EvolutionPanel";
import { RiskLimitsWorkspace } from "@/components/risk/RiskLimitsWorkspace";
import { GeographyPanel } from "@/components/GeographyPanel";
import { ViewToggle } from "@/components/ViewToggle";
import { SubTabs } from "@/components/SubTabs";
import { KeepMounted } from "@/components/KeepMounted";
import { WeeklyBriefPanel } from "@/components/insight/WeeklyBriefPanel";
import { ErrorState } from "@/components/states/States";
import { PortfolioScopeBanner } from "@/components/PortfolioScopeBanner";
import { LineagePanel } from "@/components/LineagePanel";
import type { ViewLineage, WorkspaceView } from "@/domain";
import { createAgentClient, resolveAgentClientConfig } from "@/api";
import { useWorkspace } from "@/state/useWorkspace";

// Display-only lineage (mirrors backend workspace.lineage_for); no calculation.
function fundedLineage(reportingDate: string | null): ViewLineage {
  return {
    view: "funded",
    source: "18_central_lender_tape.csv",
    metric: "current_outstanding_balance",
    fundedReportingDate: reportingDate,
    explanation: "Funded book actuals from the governed central lender tape.",
  };
}

function pipelineLineage(forecast: ReturnType<typeof useWorkspace>["forecast"]): ViewLineage {
  const snap = forecast?.pipelineSnapshot;
  const ev = snap?.historicalModelEvidence;
  return {
    view: "pipeline",
    source: snap?.sourceFile ?? "governed weekly pipeline files",
    metric: "expected_funded_amount",
    weightedMetric: "expected_funded_amount × completion_probability",
    pipelineAsOfDate: snap?.pipelineAsOfDate ?? null,
    pipelineSourceFolderDate: snap?.pipelineSourceFolderDate ?? null,
    observationWindowStart: ev?.observationWindowStart ?? null,
    observationWindowEnd: ev?.observationWindowEnd ?? null,
    completionProbabilityBasis:
      snap?.completionProbabilityBasis ?? forecast?.forecastBridge?.completionProbabilityBasis ?? null,
    historicalModelEvidence: ev,
    explanation:
      "Origination pipeline (pre-funded), governed weekly extract; completion probabilities from the historical weekly-snapshot model.",
  };
}

// Tab semantics (A3 / A10): what each top-level view represents. Each of
// Funded / Pipeline / Forecast hosts sub-tabs (see the workspace render below).
const VIEW_CAPABILITY: Record<string, string> = {
  funded: "funded",
  pipeline: "pipeline",
  forecast: "consolidated_forecast",
  risk_limits: "risk",
};

const VIEW_SUBTITLES: Record<string, string> = {
  funded: "Funded book — the funded-loan book as of the selected reporting date: stratifications, geographic exposure, time-series evolution and static-pool cohorts.",
  pipeline: "Pipeline — the open origination pipeline: current stratifications, and its evolution (stock levels over time and the weekly origination funnel flow).",
  forecast: "Forecast — forward projection from the latest run (funded + weighted pipeline + run-rate scale-up), and how the forecast has moved across runs.",
  risk_limits: "Risk Limits — Schedule 8 concentration limits vs funded actual exposure, headroom and status.",
};

type FundedTab = "strat" | "geo" | "evo" | "cohorts";
type PipelineTab = "strat" | "evo";
type ForecastTab = "projection" | "evolution";

const WORKSPACE_VIEWS: WorkspaceView[] = ["funded", "pipeline", "forecast", "risk_limits"];

/** Read the shareable navigation params off the entry URL (view, sub-tabs,
 *  scope). Client / run / question stay handled by the workspace hook. */
function readNavParams(): {
  view: WorkspaceView | null;
  funded: FundedTab | null;
  pipeline: PipelineTab | null;
  forecast: ForecastTab | null;
  scope: string | null;
} {
  const empty = { view: null, funded: null, pipeline: null, forecast: null, scope: null } as const;
  if (typeof window === "undefined" || !window.location?.search) return { ...empty };
  try {
    const p = new URLSearchParams(window.location.search);
    const pick = <T extends string>(key: string, valid: readonly T[]): T | null => {
      const v = p.get(key);
      return v && (valid as readonly string[]).includes(v) ? (v as T) : null;
    };
    return {
      view: pick("view", WORKSPACE_VIEWS),
      funded: pick("ftab", ["strat", "geo", "evo", "cohorts"] as const),
      pipeline: pick("ptab", ["strat", "evo"] as const),
      forecast: pick("xtab", ["projection", "evolution"] as const),
      scope: p.get("scope"),
    };
  } catch {
    return { ...empty };
  }
}

export function AppShell() {
  // One client for the app lifetime; swap createAgentClient() for the real
  // backend later with zero component changes.
  const client = useMemo(() => createAgentClient(), []);
  // A production build that silently fell back to the mock (VITE_AGENT_API_URL
  // unset) must be unmistakable — never let canned demo data pass for live MI.
  const agentMisconfigured = useMemo(() => resolveAgentClientConfig().misconfigured, []);
  const ws = useWorkspace(client);

  // The `"<client_id>/<run_id>"` id used by the evolution / risk / extrapolation
  // panels (falls back to the client when no run is selected).
  const workspacePortfolioId = ws.selectedRunId
    ? `${ws.selectedClientId ?? ""}/${ws.selectedRunId}`
    : (ws.selectedClientId ?? "");

  // A — core-dashboard collapse (persisted), distinct from the artifact-workspace
  // collapse below. Clearing never touches loaded MI data.
  const [dashCollapsed, setDashCollapsed] = useState<boolean>(
    () => (typeof localStorage !== "undefined"
      && localStorage.getItem("mi.coreDashboard.collapsed") === "1"));
  useEffect(() => {
    if (typeof localStorage !== "undefined") {
      localStorage.setItem("mi.coreDashboard.collapsed", dashCollapsed ? "1" : "0");
    }
  }, [dashCollapsed]);

  // Artifact-workspace collapse is owned HERE (same persisted key the canvas
  // used) so a completed query can auto-expand the workspace before revealing
  // its result — a collapsed workspace no longer swallows new answers.
  const [wsCollapsed, setWsCollapsed] = useState<boolean>(
    () => (typeof localStorage !== "undefined"
      && localStorage.getItem("mi.artifactWorkspace.collapsed") === "1"));
  useEffect(() => {
    if (typeof localStorage !== "undefined") {
      localStorage.setItem("mi.artifactWorkspace.collapsed", wsCollapsed ? "1" : "0");
    }
  }, [wsCollapsed]);

  // Deep-linkable navigation: view + sub-tabs + scope restore from the URL.
  const nav = useRef(readNavParams()).current;

  // Sub-tab state per top-level workspace (Funded / Pipeline / Forecast). Each is
  // independent so switching top-level tabs preserves the last sub-view.
  const [fundedTab, setFundedTab] = useState<FundedTab>(nav.funded ?? "strat");
  const [pipelineTab, setPipelineTab] = useState<PipelineTab>(nav.pipeline ?? "strat");
  const [forecastTab, setForecastTab] = useState<ForecastTab>(nav.forecast ?? "projection");

  // Apply the URL's view + scope once on mount (workspace owns those states).
  const applied = useRef(false);
  useEffect(() => {
    if (applied.current) return;
    applied.current = true;
    if (nav.view) ws.setActiveView(nav.view);
    if (nav.scope) ws.setSelectedContextId(nav.scope);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Mirror the current navigation into the URL (replaceState — no history spam)
  // so any view is shareable and survives a refresh. Existing params (client,
  // run, q, filters) are preserved, not clobbered.
  useEffect(() => {
    if (typeof window === "undefined" || !window.history?.replaceState) return;
    try {
      const p = new URLSearchParams(window.location.search);
      const set = (k: string, v: string | null | undefined) => {
        if (v) p.set(k, v);
        else p.delete(k);
      };
      set("client", ws.selectedClientId);
      set("run", ws.selectedRunId);
      set("view", ws.activeView !== "funded" ? ws.activeView : null);
      set("scope", ws.selectedContextId !== "total" ? ws.selectedContextId : null);
      set("ftab", fundedTab !== "strat" ? fundedTab : null);
      set("ptab", pipelineTab !== "strat" ? pipelineTab : null);
      set("xtab", forecastTab !== "projection" ? forecastTab : null);
      const qs = p.toString();
      window.history.replaceState(
        null, "", `${window.location.pathname}${qs ? `?${qs}` : ""}${window.location.hash}`,
      );
    } catch {
      /* URL sync is best-effort — never break the app for it */
    }
  }, [ws.selectedClientId, ws.selectedRunId, ws.activeView, ws.selectedContextId,
      fundedTab, pipelineTab, forecastTab]);

  // Warm a top-level view's data when the user points at its tab, so the click
  // lands on data that is already in flight or already cached. De-duplicated
  // here (a hover fires repeatedly) AND again by the caching client's in-flight
  // map, so pointing at a tab can never issue a second identical request.
  const prefetched = useRef<Set<string>>(new Set());
  const prefetchView = useCallback((view: WorkspaceView) => {
    if (!workspacePortfolioId) return;
    const key = `${view}|${workspacePortfolioId}|${ws.selectedContextId}|${ws.dataVersion}`;
    if (prefetched.current.has(key)) return;
    prefetched.current.add(key);
    const scope = ws.selectedContextId;
    const swallow = () => { /* best-effort: a failed warm just means a live fetch */ };
    if (view === "pipeline" || view === "forecast") {
      // Both views render from the forecast snapshot (it embeds the pipeline).
      client.getForecastSnapshot(workspacePortfolioId, scope).catch(swallow);
    } else if (view === "risk_limits") {
      client.getConcentrationTests(workspacePortfolioId, scope).catch(swallow);
    } else {
      client.getSnapshot(workspacePortfolioId, scope).catch(swallow);
    }
  }, [client, workspacePortfolioId, ws.selectedContextId, ws.dataVersion]);

  const scrollToArtifact = useCallback((id: string) => {
    // Guarded: jsdom (tests) has no scrollIntoView.
    document.getElementById(`artifact-${id}`)?.scrollIntoView?.({ behavior: "smooth", block: "center" });
  }, []);

  const openArtifact = useCallback((id: string) => {
    // A collapsed workspace renders no anchors — expand it first, then scroll
    // on the next frame so the link never silently does nothing.
    setWsCollapsed((collapsed) => {
      if (collapsed) {
        window.setTimeout(() => scrollToArtifact(id), 80);
        return false;
      }
      scrollToArtifact(id);
      return collapsed;
    });
  }, [scrollToArtifact]);

  // Auto-reveal: when a query lands new artifacts, expand the workspace, scroll
  // to the newest result and flash-highlight the new cards. The initial render
  // (persisted artifacts) seeds the baseline without yanking the viewport.
  const [highlightIds, setHighlightIds] = useState<string[]>([]);
  const prevArtifactIds = useRef<Set<string> | null>(null);
  useEffect(() => {
    const ids = new Set(ws.artifacts.map((a) => a.id));
    const prev = prevArtifactIds.current;
    prevArtifactIds.current = ids;
    if (!prev) return; // first render — nothing "new"
    const added = ws.artifacts.filter((a) => !prev.has(a.id));
    if (added.length === 0) return;
    setWsCollapsed(false);
    setHighlightIds(added.map((a) => a.id));
    const target = added[0].id;
    const scrollTimer = window.setTimeout(() => scrollToArtifact(target), 80);
    const clearTimer = window.setTimeout(() => setHighlightIds([]), 2600);
    return () => {
      window.clearTimeout(scrollTimer);
      window.clearTimeout(clearTimer);
    };
  }, [ws.artifacts, scrollToArtifact]);

  // Ids currently present in the workspace, so the chat can mark links to
  // since-cleared artifacts as stale instead of silently doing nothing.
  const workspaceArtifactIds = useMemo(
    () => new Set(ws.artifacts.map((a) => a.id)),
    [ws.artifacts],
  );

  return (
    <div className="flex h-screen flex-col overflow-hidden">
      {agentMisconfigured && (
        <div
          role="alert"
          className="flex items-center justify-center gap-2 bg-red-600 px-4 py-2 text-center text-sm font-semibold text-white"
        >
          Configuration error: no backend URL (VITE_AGENT_API_URL) is set, so this
          build is showing MOCK demo data — not live portfolio MI. Do not rely on
          these figures.
        </div>
      )}
      <HeaderBar
        portfolios={ws.portfolios}
        runs={ws.runs}
        selectedClientId={ws.selectedClientId}
        selectedRunId={ws.selectedRunId}
        onPortfolioChange={ws.setPortfolio}
        onRunChange={ws.setRun}
        mock={client.mock}
        client={client}
        portfolioId={workspacePortfolioId}
        reportingPeriod={ws.reporting.asOf ? ws.reporting.asOf.slice(0, 7) : null}
        identity={ws.identity}
        onRefresh={ws.refresh}
        refreshing={ws.refreshing}
        // ONE portfolio-scope control for the whole workspace, in the permanent
        // header (it scopes chat + exports, so it must never collapse away).
        contexts={ws.portfolioContexts}
        selectedContextId={ws.selectedContextId}
        onContextChange={ws.setSelectedContextId}
      />
      <div className="flex min-h-0 flex-1">
        <AgentChatPanel
          messages={ws.messages}
          isWorking={ws.isWorking}
          mock={client.mock}
          initialInput={ws.initialQuestion}
          onSubmit={ws.ask}
          onStop={ws.stop}
          onOpenArtifact={openArtifact}
          onRetry={ws.retryLast}
          context={ws.context}
          onClearContext={ws.clearContext}
          onClearChat={ws.clearChat}
          onTogglePin={ws.togglePin}
          workspaceArtifactIds={workspaceArtifactIds}
        />
        <div className="flex min-h-0 flex-1 flex-col overflow-y-auto">
          {/* Phase 3A — OPTIONAL Weekly Portfolio Brief. Renders nothing at all
              unless its flag is on AND the API returned insights, so a build
              without it is byte-identical to today. Replaces no chart. */}
          <WeeklyBriefPanel
            client={client}
            portfolioId={workspacePortfolioId}
            portfolioContext={ws.selectedContextId}
          />

          {/* Region 3/4 — CORE DASHBOARD: neutral panel, distinct from the grey
              artifact workspace below. Collapsible to focus the chart area. */}
          <section
            data-testid="core-dashboard"
            className="mx-6 mt-5 rounded-2xl border border-[var(--surface-dashboard-line)] bg-[var(--surface-dashboard)] shadow-sm"
          >
            <header className="flex items-center justify-between gap-3 border-b border-[var(--surface-dashboard-line)] px-4 py-3">
              <div className="flex items-center gap-2.5">
                <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-navy-700 text-peri-300">
                  <LayoutDashboard size={18} />
                </div>
                <h2 className="text-base font-semibold text-ink-100">Core Dashboard</h2>
              </div>
              <button
                type="button"
                onClick={() => setDashCollapsed((c) => !c)}
                aria-label={dashCollapsed ? "Expand core dashboard" : "Collapse core dashboard"}
                aria-expanded={!dashCollapsed}
                data-testid="core-dashboard-toggle"
                className="inline-flex items-center rounded-md px-1.5 py-1 text-ink-400 hover:text-ink-100"
              >
                <ChevronDown size={15} className={dashCollapsed ? "-rotate-90 transition-transform" : "transition-transform"} />
              </button>
            </header>
            {/* Dashboard navigation: the lens tabs span the full panel width.
                The portfolio-scope selector lives in the permanent header. */}
            {!dashCollapsed && (
              <div className="flex flex-wrap items-center gap-3 border-b border-[var(--surface-dashboard-line)] px-4 py-3">
                <ViewToggle
                  active={ws.activeView}
                  onChange={ws.setActiveView}
                  disabledViews={ws.disabledViews}
                  disabledReasons={ws.disabledViewReasons}
                  onPrefetch={prefetchView}
                  className="min-w-0 flex-1"
                />
              </div>
            )}
            {!dashCollapsed && (
              <p className="px-4 pt-3 text-[11px] text-ink-500" data-testid="view-subtitle">
                {VIEW_SUBTITLES[ws.activeView]}
              </p>
            )}
            {dashCollapsed ? (
              <p className="px-4 py-3 text-[11px] text-ink-500">
                Core Dashboard collapsed — expand to show the {ws.activeView} view.
              </p>
            ) : (
              <div className="space-y-4 p-4">
                {/* Discovery failure (API unreachable / unauthorised) renders as
                    a visible error — never a bare "No client" header. */}
                {ws.portfolios.length === 0 && ws.portfoliosError && (
                  <ErrorState
                    message={`The MI Agent API could not be reached — ${ws.portfoliosError}`}
                    onRetry={ws.refresh}
                  />
                )}
                {/* Governed disclosure for the active view: which portfolios
                    contribute, which are excluded and why. Backend prose only. */}
                <PortfolioScopeBanner
                  context={ws.activeContext}
                  capability={ws.capabilities[VIEW_CAPABILITY[ws.activeView]]}
                />
                {/* FUNDED — stratifications · geography · evolution · cohorts.
                    Risk Limits lives ONCE, as the top-level tab (no duplicate
                    sub-tab entry pointing at the same workspace). */}
                {ws.activeView === "funded" && (
                  <div className="space-y-4">
                    <SubTabs ariaLabel="Funded sub-view" testId="funded-subtabs"
                      active={fundedTab} onChange={setFundedTab}
                      tabs={[
                        { id: "strat", label: "Stratifications" },
                        { id: "geo", label: "Geography" },
                        { id: "evo", label: "Evolution" },
                        { id: "cohorts", label: "Cohorts" },
                      ]} />
                    <KeepMounted active={fundedTab === "strat"} testId="funded-strat-pane">
                      {/* A failed load renders a VISIBLE error with the real
                          reason + retry — never a silent blank panel. */}
                      {!ws.snapshot && !ws.snapshotLoading && ws.snapshotError && (
                        <ErrorState
                          message={`Funded Book Snapshot could not be loaded — ${ws.snapshotError}`}
                          onRetry={ws.refresh}
                        />
                      )}
                      <FundedSnapshotPanel snapshot={ws.snapshot} loading={ws.snapshotLoading} />
                      <LineagePanel lineage={fundedLineage(ws.snapshot?.portfolio.reporting_date ?? null)} />
                    </KeepMounted>
                    <KeepMounted active={fundedTab === "geo"} testId="funded-geo-pane">
                      <GeographyPanel key={`geo-${ws.dataVersion}-${ws.selectedContextId}`}
                        client={client} portfolioId={workspacePortfolioId}
                        portfolioContext={ws.selectedContextId} />
                    </KeepMounted>
                    <KeepMounted active={fundedTab === "evo"} testId="funded-evo-pane">
                      <EvolutionPanel key={`evo-funded-${ws.dataVersion}-${ws.selectedContextId}`} heading={false}
                        tabs={["funded"]} client={client} portfolioId={workspacePortfolioId}
                        portfolioContext={ws.selectedContextId} />
                    </KeepMounted>
                    <KeepMounted active={fundedTab === "cohorts"} testId="funded-cohorts-pane">
                      <EvolutionPanel key={`evo-cohorts-${ws.dataVersion}-${ws.selectedContextId}`} heading={false}
                        tabs={["cohorts"]} client={client} portfolioId={workspacePortfolioId}
                        portfolioContext={ws.selectedContextId} />
                    </KeepMounted>
                  </div>
                )}

                {/* PIPELINE — stratifications · evolution (stock + origination flow).
                    Rendered only where the governed capability allows it; when it
                    does not, the banner above carries the business explanation and
                    no empty analysis is drawn. */}
                {ws.activeView === "pipeline" && ws.capabilityEnabled("pipeline") && (
                  <div className="space-y-4">
                    <SubTabs ariaLabel="Pipeline sub-view" testId="pipeline-subtabs"
                      active={pipelineTab} onChange={setPipelineTab}
                      tabs={[
                        { id: "strat", label: "Stratifications" },
                        { id: "evo", label: "Evolution" },
                      ]} />
                    <KeepMounted active={pipelineTab === "strat"} testId="pipeline-strat-pane">
                      {!ws.forecast && !ws.forecastLoading && ws.forecastError && (
                        <ErrorState
                          message={`Pipeline snapshot could not be loaded — ${ws.forecastError}`}
                          onRetry={ws.refresh}
                        />
                      )}
                      <PipelineSnapshotPanel
                        snapshot={ws.forecast?.pipelineSnapshot ?? null}
                        loading={ws.forecastLoading}
                      />
                      <LineagePanel lineage={ws.forecast?.lineage ?? pipelineLineage(ws.forecast)} />
                      <PipelineWatchlist items={ws.forecast?.watchlist ?? []} />
                    </KeepMounted>
                    <KeepMounted active={pipelineTab === "evo"} testId="pipeline-evo-pane">
                      <EvolutionPanel key={`evo-pipeline-${ws.dataVersion}-${ws.selectedContextId}`} heading={false}
                        tabs={["pipeline", "origination"]} client={client} portfolioId={workspacePortfolioId}
                        portfolioContext={ws.selectedContextId} />
                    </KeepMounted>
                  </div>
                )}

                {/* FORECAST — projection · forecast evolution */}
                {ws.activeView === "forecast" && ws.capabilityEnabled("consolidated_forecast") && (
                  <div className="space-y-4">
                    <SubTabs ariaLabel="Forecast sub-view" testId="forecast-subtabs"
                      active={forecastTab} onChange={setForecastTab}
                      tabs={[
                        { id: "projection", label: "Projection" },
                        { id: "evolution", label: "Forecast Evolution" },
                      ]} />
                    <KeepMounted active={forecastTab === "projection"} testId="forecast-projection-pane">
                      {!ws.forecast && !ws.forecastLoading && ws.forecastError && (
                        <ErrorState
                          message={`Forecast could not be loaded — ${ws.forecastError}`}
                          onRetry={ws.refresh}
                        />
                      )}
                      <ForecastView forecast={ws.forecast} loading={ws.forecastLoading} />
                      <ForecastExtrapolationPanel key={`fx-${ws.dataVersion}-${ws.selectedContextId}`}
                        client={client} portfolioId={workspacePortfolioId}
                        portfolioContext={ws.selectedContextId} />
                    </KeepMounted>
                    <KeepMounted active={forecastTab === "evolution"} testId="forecast-evolution-pane">
                      <EvolutionPanel key={`evo-forecast-${ws.dataVersion}-${ws.selectedContextId}`} heading={false}
                        tabs={["forecast"]} client={client} portfolioId={workspacePortfolioId}
                        portfolioContext={ws.selectedContextId} />
                    </KeepMounted>
                  </div>
                )}

                {ws.activeView === "risk_limits" && ws.capabilityEnabled("risk") && (
                  <RiskLimitsWorkspace key={`risk-${ws.dataVersion}-${ws.selectedContextId}`}
                    client={client} portfolioId={workspacePortfolioId}
                    portfolioContext={ws.selectedContextId} />
                )}
              </div>
            )}
          </section>

          {/* Region 2 — ARTIFACT WORKSPACE: faint blue-grey surface + accent rail,
              clearly divided from the core dashboard above. */}
          <div
            data-testid="artifact-region"
            className="mx-6 mb-6 mt-4 rounded-2xl border border-[var(--surface-artifact-line)] border-l-2 border-l-[var(--surface-artifact-accent)] bg-[var(--surface-artifact)] shadow-sm"
            style={{ backgroundImage: "linear-gradient(var(--surface-artifact-tint), var(--surface-artifact-tint))" }}
          >
            <ArtifactCanvas
              artifacts={ws.artifacts}
              onTogglePin={ws.togglePin}
              isWorking={ws.isWorking}
              portfolioName={ws.portfolio.name}
              collapsed={wsCollapsed}
              onToggleCollapsed={() => setWsCollapsed((c) => !c)}
              highlightIds={highlightIds}
              // Backend drill-through only when wired to a live backend; the mock
              // client keeps the client-side drill panel as the fallback.
              onDrill={client.mock ? undefined : ws.drill}
              // Insight investigations re-ask through the context-aware flow.
              onAsk={ws.ask}
              // Declutter: clear workspace artifacts (loaded MI data untouched).
              onClear={ws.clearArtifacts}
            />
          </div>
        </div>
      </div>
    </div>
  );
}
