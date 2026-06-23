import { useCallback, useMemo } from "react";
import { HeaderBar } from "@/components/HeaderBar";
import { AgentChatPanel } from "@/components/AgentChatPanel";
import { ArtifactCanvas } from "@/components/ArtifactCanvas";
import { FundedSnapshotPanel } from "@/components/FundedSnapshotPanel";
import { ForecastBridgeCard } from "@/components/ForecastBridgeCard";
import { PipelineSnapshotPanel } from "@/components/PipelineSnapshotPanel";
import { PipelineWatchlist } from "@/components/PipelineWatchlist";
import { createAgentClient } from "@/api";
import { useWorkspace } from "@/state/useWorkspace";

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
          {/* Deterministic funded snapshot — shown before any AI query. */}
          <div className="space-y-4 px-6 pt-5">
            <FundedSnapshotPanel snapshot={ws.snapshot} loading={ws.snapshotLoading} />
            {/* Governed pipeline SSoT + deterministic forecast bridge + watchlist. */}
            <ForecastBridgeCard bridge={ws.forecast?.forecastBridge ?? null} />
            <PipelineSnapshotPanel
              snapshot={ws.forecast?.pipelineSnapshot ?? null}
              loading={ws.forecastLoading}
            />
            <PipelineWatchlist items={ws.forecast?.watchlist ?? []} />
          </div>
          <ArtifactCanvas
            artifacts={ws.artifacts}
            onTogglePin={ws.togglePin}
            isWorking={ws.isWorking}
            portfolioName={ws.portfolio.name}
          />
        </div>
      </div>
    </div>
  );
}
