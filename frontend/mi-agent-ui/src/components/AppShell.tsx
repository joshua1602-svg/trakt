import { useCallback, useEffect, useMemo, useState } from "react";
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
import { RiskLimitsPanel } from "@/components/RiskLimitsPanel";
import { GeographyPanel } from "@/components/GeographyPanel";
import { ViewToggle } from "@/components/ViewToggle";
import { SubTabs } from "@/components/SubTabs";
import { SourcePortfolioSelector } from "@/components/SourcePortfolioSelector";
import { LineagePanel } from "@/components/LineagePanel";
import type { ViewLineage } from "@/domain";
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
const VIEW_SUBTITLES: Record<string, string> = {
  funded: "Funded book — the funded-loan book as of the selected reporting date: stratifications, geographic exposure, time-series evolution and static-pool cohorts.",
  pipeline: "Pipeline — the open origination pipeline: current stratifications, and its evolution (stock levels over time and the weekly origination funnel flow).",
  forecast: "Forecast — forward projection from the latest run (funded + weighted pipeline + run-rate scale-up), and how the forecast has moved across runs.",
  risk_limits: "Risk Limits — Schedule 8 concentration limits vs funded actual exposure, headroom and status.",
};

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

  const openArtifact = useCallback((id: string) => {
    document.getElementById(`artifact-${id}`)?.scrollIntoView({ behavior: "smooth", block: "center" });
  }, []);

  // A — core-dashboard collapse (persisted), distinct from the artifact-workspace
  // collapse owned by ArtifactCanvas. Clearing never touches loaded MI data.
  const [dashCollapsed, setDashCollapsed] = useState<boolean>(
    () => (typeof localStorage !== "undefined"
      && localStorage.getItem("mi.coreDashboard.collapsed") === "1"));
  useEffect(() => {
    if (typeof localStorage !== "undefined") {
      localStorage.setItem("mi.coreDashboard.collapsed", dashCollapsed ? "1" : "0");
    }
  }, [dashCollapsed]);

  // Sub-tab state per top-level workspace (Funded / Pipeline / Forecast). Each is
  // independent so switching top-level tabs preserves the last sub-view.
  const [fundedTab, setFundedTab] = useState<"strat" | "geo" | "evo" | "cohorts">("strat");
  const [pipelineTab, setPipelineTab] = useState<"strat" | "evo">("strat");
  const [forecastTab, setForecastTab] = useState<"projection" | "evolution">("projection");

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
      />
      <div className="flex min-h-0 flex-1">
        <AgentChatPanel
          messages={ws.messages}
          isWorking={ws.isWorking}
          mock={client.mock}
          onSubmit={ws.ask}
          onOpenArtifact={openArtifact}
          onRetry={ws.retryLast}
          context={ws.context}
          onClearContext={ws.clearContext}
          onClearChat={ws.clearChat}
          onTogglePin={ws.togglePin}
        />
        <div className="flex min-h-0 flex-1 flex-col overflow-y-auto">
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
            {/* Dashboard navigation: the lens tabs span the full panel width; the
                portfolio-scope selector sits alongside, visually separated so it
                reads as a selector rather than another tab. */}
            {!dashCollapsed && (
              <div className="flex flex-wrap items-center gap-3 border-b border-[var(--surface-dashboard-line)] px-4 py-3">
                <ViewToggle
                  active={ws.activeView}
                  onChange={ws.setActiveView}
                  disabledViews={ws.disabledViews}
                  className="min-w-0 flex-1"
                />
                {ws.sourceLenses.length > 1 && (
                  <>
                    <div className="h-6 w-px shrink-0 bg-[var(--surface-dashboard-line)]" />
                    <SourcePortfolioSelector
                      lenses={ws.sourceLenses}
                      value={ws.selectedLens}
                      onChange={ws.setSelectedLens}
                    />
                  </>
                )}
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
                {/* FUNDED — stratifications · geography · evolution · cohorts */}
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
                    {fundedTab === "strat" && (
                      <>
                        <FundedSnapshotPanel snapshot={ws.snapshot} loading={ws.snapshotLoading} />
                        <LineagePanel lineage={fundedLineage(ws.snapshot?.portfolio.reporting_date ?? null)} />
                      </>
                    )}
                    {fundedTab === "geo" && (
                      <GeographyPanel key={`geo-${ws.dataVersion}`}
                        client={client} portfolioId={workspacePortfolioId} />
                    )}
                    {fundedTab === "evo" && (
                      <EvolutionPanel key={`evo-funded-${ws.dataVersion}`} heading={false}
                        tabs={["funded"]} client={client} portfolioId={workspacePortfolioId} />
                    )}
                    {fundedTab === "cohorts" && (
                      <EvolutionPanel key={`evo-cohorts-${ws.dataVersion}`} heading={false}
                        tabs={["cohorts"]} client={client} portfolioId={workspacePortfolioId} />
                    )}
                  </div>
                )}

                {/* PIPELINE — stratifications · evolution (stock + origination flow) */}
                {ws.activeView === "pipeline" && (
                  <div className="space-y-4">
                    <SubTabs ariaLabel="Pipeline sub-view" testId="pipeline-subtabs"
                      active={pipelineTab} onChange={setPipelineTab}
                      tabs={[
                        { id: "strat", label: "Stratifications" },
                        { id: "evo", label: "Evolution" },
                      ]} />
                    {pipelineTab === "strat" && (
                      <>
                        <PipelineSnapshotPanel
                          snapshot={ws.forecast?.pipelineSnapshot ?? null}
                          loading={ws.forecastLoading}
                        />
                        <LineagePanel lineage={ws.forecast?.lineage ?? pipelineLineage(ws.forecast)} />
                        <PipelineWatchlist items={ws.forecast?.watchlist ?? []} />
                      </>
                    )}
                    {pipelineTab === "evo" && (
                      <EvolutionPanel key={`evo-pipeline-${ws.dataVersion}`} heading={false}
                        tabs={["pipeline", "origination"]} client={client} portfolioId={workspacePortfolioId} />
                    )}
                  </div>
                )}

                {/* FORECAST — projection · forecast evolution */}
                {ws.activeView === "forecast" && (
                  <div className="space-y-4">
                    <SubTabs ariaLabel="Forecast sub-view" testId="forecast-subtabs"
                      active={forecastTab} onChange={setForecastTab}
                      tabs={[
                        { id: "projection", label: "Projection" },
                        { id: "evolution", label: "Forecast Evolution" },
                      ]} />
                    {forecastTab === "projection" && (
                      <>
                        <ForecastView forecast={ws.forecast} loading={ws.forecastLoading} />
                        <ForecastExtrapolationPanel key={`fx-${ws.dataVersion}`}
                          client={client} portfolioId={workspacePortfolioId} />
                      </>
                    )}
                    {forecastTab === "evolution" && (
                      <EvolutionPanel key={`evo-forecast-${ws.dataVersion}`} heading={false}
                        tabs={["forecast"]} client={client} portfolioId={workspacePortfolioId} />
                    )}
                  </div>
                )}

                {ws.activeView === "risk_limits" && (
                  <RiskLimitsPanel key={`risk-${ws.dataVersion}`}
                    client={client} portfolioId={workspacePortfolioId} />
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
