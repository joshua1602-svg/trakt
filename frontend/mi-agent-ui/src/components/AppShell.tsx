import { useCallback, useMemo } from "react";
import { HeaderBar } from "@/components/HeaderBar";
import { AgentChatPanel } from "@/components/AgentChatPanel";
import { ArtifactCanvas } from "@/components/ArtifactCanvas";
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
        portfolio={ws.portfolio.id}
        onPortfolioChange={ws.setPortfolio}
        reportingDate={ws.reporting.asOf}
        onReportingDateChange={ws.setReportingDate}
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
        <ArtifactCanvas
          artifacts={ws.artifacts}
          onTogglePin={ws.togglePin}
          isWorking={ws.isWorking}
          portfolioName={ws.portfolio.name}
        />
      </div>
    </div>
  );
}
