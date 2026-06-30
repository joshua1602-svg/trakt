import { useCallback, useEffect, useMemo, useState } from "react";
import { ChevronDown, Eraser, LayoutDashboard, Trash2 } from "lucide-react";
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
import { ViewToggle } from "@/components/ViewToggle";
import { SourcePortfolioSelector } from "@/components/SourcePortfolioSelector";
import { LineagePanel } from "@/components/LineagePanel";
import type { ViewLineage } from "@/domain";
import { createAgentClient } from "@/api";
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

// Tab semantics (A3 / A10): what each top-level view represents and its source.
const VIEW_SUBTITLES: Record<string, string> = {
  funded: "Funded book — latest funded-loan snapshot as of the selected reporting date (governed central lender tape).",
  pipeline: "Pipeline — latest open-pipeline snapshot (weighted expected funded balance) as of the selected reporting date.",
  forecast: "Scenario / portfolio forecast — forward-looking projection from the latest selected run (funded balance + weighted pipeline + run-rate scale-up). For how the forecast changed across runs, see Evolution → Forecast Evolution.",
  evolution: "Evolution — time-series movement across multiple reporting extracts (funded / pipeline / origination funnel / forecast).",
  risk_limits: "Risk Limits — Schedule 8 concentration limits vs funded actual exposure, headroom and status.",
};

/** Declutter cluster (A): clear chat / artifacts / both in one place. Each is a
 * VIEW reset only — the loaded portfolio / run / dataset is never touched. */
function DeclutterControls({
  onClearChat, onClearArtifacts, onClearBoth,
}: {
  onClearChat: () => void;
  onClearArtifacts: () => void;
  onClearBoth: () => void;
}) {
  return (
    <div
      role="group"
      aria-label="Clear workspace"
      data-testid="declutter-controls"
      className="inline-flex items-center gap-0.5 rounded-lg border border-[var(--color-line)] bg-navy-900/60 p-0.5"
    >
      <button type="button" onClick={onClearChat}
        title="Clear the conversation (loaded MI data is untouched)"
        className="inline-flex items-center gap-1 rounded-md px-2 py-1 text-[11px] font-medium text-ink-400 hover:text-rose-300">
        <Eraser size={12} /> Clear chat
      </button>
      <button type="button" onClick={onClearArtifacts}
        title="Clear the artifact workspace (loaded MI data is untouched)"
        className="inline-flex items-center gap-1 rounded-md px-2 py-1 text-[11px] font-medium text-ink-400 hover:text-rose-300">
        <Trash2 size={12} /> Clear artifacts
      </button>
      <button type="button" onClick={onClearBoth}
        title="Clear chat and artifacts (loaded MI data is untouched)"
        className="inline-flex items-center gap-1 rounded-md px-2 py-1 text-[11px] font-medium text-ink-400 hover:text-rose-300">
        Clear both
      </button>
    </div>
  );
}

export function AppShell() {
  // One client for the app lifetime; swap createAgentClient() for the real
  // backend later with zero component changes.
  const client = useMemo(() => createAgentClient(), []);
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

  return (
    <div className="flex h-screen flex-col overflow-hidden">
      <HeaderBar
        portfolios={ws.portfolios}
        runs={ws.runs}
        selectedClientId={ws.selectedClientId}
        selectedRunId={ws.selectedRunId}
        onPortfolioChange={ws.setPortfolio}
        onRunChange={ws.setRun}
        mock={client.mock}
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
          // Inline-result drill uses the backend when live; mock keeps the
          // client-side drill fallback inside the embedded card.
          onDrill={client.mock ? undefined : ws.drill}
        />
        <div className="flex min-h-0 flex-1 flex-col overflow-y-auto">
          {/* One coherent workspace: a view toggle selects which schema-aligned
              view is foregrounded (no stacking of all sections at once). */}
          <div className="flex items-center justify-between gap-3 px-6 pt-5">
            <div className="flex items-center gap-3">
              <ViewToggle
                active={ws.activeView}
                onChange={ws.setActiveView}
                disabledViews={ws.disabledViews}
              />
              {ws.sourceLenses.length > 1 && (
                <SourcePortfolioSelector
                  lenses={ws.sourceLenses}
                  value={ws.selectedLens}
                  onChange={ws.setSelectedLens}
                />
              )}
            </div>
            <span className="text-[11px] text-ink-500">
              {ws.portfolio.name}
              {ws.reporting.asOf ? ` · ${ws.reporting.asOf}` : ""}
            </span>
          </div>
          <p className="px-6 pt-1 text-[11px] text-ink-500" data-testid="view-subtitle">
            {VIEW_SUBTITLES[ws.activeView]}
          </p>

          {/* Region 3/4 — CORE DASHBOARD: neutral panel, distinct from the faint
              blue artifact workspace below. Collapsible to focus the chart area. */}
          <section
            data-testid="core-dashboard"
            className="mx-6 mt-4 rounded-2xl border border-[var(--surface-dashboard-line)] bg-[var(--surface-dashboard)] shadow-sm"
          >
            <header className="flex items-center justify-between gap-3 border-b border-[var(--surface-dashboard-line)] px-4 py-2.5">
              <div className="flex items-center gap-2 text-[12px] font-semibold text-ink-200">
                <LayoutDashboard size={14} className="text-peri-300" /> Core dashboard
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
            {dashCollapsed ? (
              <p className="px-4 py-3 text-[11px] text-ink-500">
                Core dashboard collapsed — expand to show the {ws.activeView} view.
              </p>
            ) : (
              <div className="space-y-4 p-4">
                {ws.activeView === "funded" && (
                  <>
                    <FundedSnapshotPanel snapshot={ws.snapshot} loading={ws.snapshotLoading} />
                    <LineagePanel lineage={fundedLineage(ws.snapshot?.portfolio.reporting_date ?? null)} />
                  </>
                )}
                {ws.activeView === "pipeline" && (
                  <>
                    <PipelineSnapshotPanel
                      snapshot={ws.forecast?.pipelineSnapshot ?? null}
                      loading={ws.forecastLoading}
                    />
                    <LineagePanel lineage={ws.forecast?.lineage ?? pipelineLineage(ws.forecast)} />
                    <PipelineWatchlist items={ws.forecast?.watchlist ?? []} />
                  </>
                )}
                {ws.activeView === "forecast" && (
                  <>
                    <ForecastView forecast={ws.forecast} loading={ws.forecastLoading} />
                    <ForecastExtrapolationPanel client={client} portfolioId={workspacePortfolioId} />
                  </>
                )}
                {ws.activeView === "evolution" && (
                  <EvolutionPanel client={client} portfolioId={workspacePortfolioId} />
                )}
                {ws.activeView === "risk_limits" && (
                  <RiskLimitsPanel client={client} portfolioId={workspacePortfolioId} />
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
