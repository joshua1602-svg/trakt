import { useCallback, useMemo } from "react";
import { HeaderBar } from "@/components/HeaderBar";
import { AgentChatPanel } from "@/components/AgentChatPanel";
import { ArtifactCanvas } from "@/components/ArtifactCanvas";
import { FundedSnapshotPanel } from "@/components/FundedSnapshotPanel";
import { PipelineSnapshotPanel } from "@/components/PipelineSnapshotPanel";
import { PipelineWatchlist } from "@/components/PipelineWatchlist";
import { ForecastView } from "@/components/ForecastView";
import { ViewToggle } from "@/components/ViewToggle";
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

export function AppShell() {
  // One client for the app lifetime; swap createAgentClient() for the real
  // backend later with zero component changes.
  const client = useMemo(() => createAgentClient(), []);
  const ws = useWorkspace(client);

  const openArtifact = useCallback((id: string) => {
    document.getElementById(`artifact-${id}`)?.scrollIntoView({ behavior: "smooth", block: "center" });
  }, []);

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
        />
        <div className="flex min-h-0 flex-1 flex-col overflow-y-auto">
          {/* One coherent workspace: a view toggle selects which schema-aligned
              view is foregrounded (no stacking of all sections at once). */}
          <div className="flex items-center justify-between gap-3 px-6 pt-5">
            <ViewToggle active={ws.activeView} onChange={ws.setActiveView} />
            <span className="text-[11px] text-ink-500">
              {ws.portfolio.name}
              {ws.reporting.asOf ? ` · ${ws.reporting.asOf}` : ""}
            </span>
          </div>
          <div className="space-y-4 px-6 pt-4">
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
              <ForecastView forecast={ws.forecast} loading={ws.forecastLoading} />
            )}
          </div>
          <ArtifactCanvas
            artifacts={ws.artifacts}
            onTogglePin={ws.togglePin}
            isWorking={ws.isWorking}
            portfolioName={ws.portfolio.name}
            // Backend drill-through only when wired to a live backend; the mock
            // client keeps the client-side drill panel as the fallback.
            onDrill={client.mock ? undefined : ws.drill}
          />
        </div>
      </div>
    </div>
  );
}
