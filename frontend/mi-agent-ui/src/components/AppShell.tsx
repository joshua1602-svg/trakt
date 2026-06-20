import { useCallback, useState } from "react";
import type { Artifact, ChatMessage } from "@/types";
import { HeaderBar } from "@/components/HeaderBar";
import { AgentChatPanel } from "@/components/AgentChatPanel";
import { ArtifactCanvas } from "@/components/ArtifactCanvas";
import { LANDING_ARTIFACTS, PORTFOLIOS, REPORTING_DATES } from "@/data/mockData";
import { MOCK_LATENCY_MS, runAgent } from "@/data/agentEngine";
import { uid } from "@/lib/utils";

const GREETING: ChatMessage = {
  id: "greeting",
  role: "assistant",
  content:
    "Welcome back. I'm your MI Agent for ERM UK — Master, as of 31 May 2026. Ask me about portfolio movement, concentration risk, the funding pipeline, static pool performance or validation status. I'll surface the analysis and supporting artifacts on the right.",
  createdAt: new Date().toISOString(),
};

export function AppShell() {
  const [portfolio, setPortfolio] = useState(PORTFOLIOS[0].id);
  const [reportingDate, setReportingDate] = useState(REPORTING_DATES[0]);
  const [messages, setMessages] = useState<ChatMessage[]>([GREETING]);
  const [artifacts, setArtifacts] = useState<Artifact[]>(LANDING_ARTIFACTS);
  const [isWorking, setIsWorking] = useState(false);

  const togglePin = useCallback((id: string) => {
    setArtifacts((prev) =>
      prev.map((a) => (a.id === id ? { ...a, pinned: !a.pinned } : a)),
    );
  }, []);

  const openArtifact = useCallback((id: string) => {
    document
      .getElementById(`artifact-${id}`)
      ?.scrollIntoView({ behavior: "smooth", block: "center" });
  }, []);

  const handleSubmit = useCallback(
    (text: string) => {
      if (isWorking) return;

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
      setIsWorking(true);

      // Mocked latency — stands in for streaming agent execution.
      window.setTimeout(() => {
        const res = runAgent(text);

        // Keep pinned artifacts; replace the rest with this turn's output.
        setArtifacts((prev) => {
          const pinned = prev.filter((a) => a.pinned);
          const pinnedIds = new Set(pinned.map((a) => a.id));
          const fresh = res.artifacts.filter((a) => !pinnedIds.has(a.id));
          return [...pinned, ...fresh];
        });

        setMessages((prev) =>
          prev.map((m) =>
            m.id === pendingId
              ? {
                  ...m,
                  pending: false,
                  content: res.narrative,
                  assumptions: res.assumptions,
                  intent: res.intent,
                  artifactRefs: res.artifacts.map((a) => ({
                    id: a.id,
                    title: a.title,
                  })),
                }
              : m,
          ),
        );
        setIsWorking(false);
      }, MOCK_LATENCY_MS);
    },
    [isWorking],
  );

  return (
    <div className="flex h-screen flex-col overflow-hidden">
      <HeaderBar
        portfolio={portfolio}
        onPortfolioChange={setPortfolio}
        reportingDate={reportingDate}
        onReportingDateChange={setReportingDate}
      />
      <div className="flex min-h-0 flex-1">
        <AgentChatPanel
          messages={messages}
          isWorking={isWorking}
          onSubmit={handleSubmit}
          onOpenArtifact={openArtifact}
        />
        <ArtifactCanvas
          artifacts={artifacts}
          onTogglePin={togglePin}
          isWorking={isWorking}
        />
      </div>
    </div>
  );
}
