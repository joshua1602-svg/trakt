/**
 * useWorkspace — central orchestration hook.
 *
 * Owns portfolio/reporting context, chat history, the artifact canvas, and all
 * agent interaction (through the AgentClient only). Handles loading, error and
 * empty states and persists across reloads. This is the seam a future backend
 * plugs into: swap the client passed in, nothing else changes.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type {
  AgentRequest,
  Artifact,
  ChatMessage,
  PortfolioContext,
  ReportingContext,
} from "@/domain";
import type { AgentClient } from "@/api";
import {
  DEFAULT_PORTFOLIO,
  DEFAULT_REPORTING_DATE,
  PORTFOLIOS,
  PRIOR_PERIOD,
} from "@/data/catalog";
import { landingArtifacts } from "@/data/mockArtifacts";
import { uid } from "@/lib/utils";
import { loadState, saveState } from "./persistence";

function greeting(portfolio: PortfolioContext, asOf: string): ChatMessage {
  const date = new Date(asOf).toLocaleDateString("en-GB", {
    day: "numeric",
    month: "long",
    year: "numeric",
  });
  return {
    id: "greeting",
    role: "assistant",
    content: `Welcome back. I'm your MI Agent for ${portfolio.name}, as of ${date}. Ask me about portfolio movement, concentration, the funding pipeline, static pools, risk monitoring, scenarios or validation — I'll surface the analysis and supporting artifacts on the right.`,
    createdAt: new Date().toISOString(),
  };
}

export interface Workspace {
  portfolio: PortfolioContext;
  reporting: ReportingContext;
  messages: ChatMessage[];
  artifacts: Artifact[];
  isWorking: boolean;
  setPortfolio: (id: string) => void;
  setReportingDate: (asOf: string) => void;
  ask: (question: string) => void;
  retryLast: () => void;
  togglePin: (id: string) => void;
  resetWorkspace: () => void;
}

export function useWorkspace(client: AgentClient): Workspace {
  const persisted = useRef(loadState()).current;

  const [portfolioId, setPortfolioId] = useState(persisted?.portfolioId ?? DEFAULT_PORTFOLIO.id);
  const [asOf, setAsOf] = useState(persisted?.asOf ?? DEFAULT_REPORTING_DATE);

  const portfolio = useMemo(
    () => PORTFOLIOS.find((p) => p.id === portfolioId) ?? DEFAULT_PORTFOLIO,
    [portfolioId],
  );
  const reporting = useMemo<ReportingContext>(
    () => ({ asOf, comparedTo: PRIOR_PERIOD[asOf] }),
    [asOf],
  );

  const [messages, setMessages] = useState<ChatMessage[]>(
    () => persisted?.messages ?? [greeting(portfolio, asOf)],
  );
  const [artifacts, setArtifacts] = useState<Artifact[]>(() => {
    // Demo mode seeds the sample landing artifacts; LIVE mode starts clean and
    // never shows mock cards (and drops any mock artifacts left in localStorage
    // from a previous demo session). Real artifacts appear as queries succeed.
    if (client.mock) return persisted?.artifacts ?? landingArtifacts(reporting, portfolioId);
    return (persisted?.artifacts ?? []).filter((a) => !a.mock);
  });
  const [isWorking, setIsWorking] = useState(false);

  const lastQuestion = useRef<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  // Persist (debounced via microtask coalescing through React batching).
  useEffect(() => {
    saveState({ portfolioId, asOf, messages, artifacts });
  }, [portfolioId, asOf, messages, artifacts]);

  const runQuery = useCallback(
    (question: string, pendingId: string) => {
      const request: AgentRequest = {
        question,
        portfolio,
        reporting,
        options: { parserMode: "deterministic" },
      };
      const controller = new AbortController();
      abortRef.current = controller;
      setIsWorking(true);

      client
        .ask(request, controller.signal)
        .then((res) => {
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
                    error: false,
                    content: res.narrative,
                    interpreted: res.interpreted,
                    assumptions: res.assumptions,
                    warnings: res.warnings,
                    intent: res.intent,
                    artifactRefs: res.artifacts.map((a) => ({ id: a.id, title: a.title, type: a.type })),
                  }
                : m,
            ),
          );
        })
        .catch((err: unknown) => {
          const message = err instanceof Error ? err.message : "The MI Agent could not complete this request.";
          setMessages((prev) =>
            prev.map((m) =>
              m.id === pendingId
                ? { ...m, pending: false, error: true, content: message }
                : m,
            ),
          );
        })
        .finally(() => {
          setIsWorking(false);
          abortRef.current = null;
        });
    },
    [client, portfolio, reporting],
  );

  const ask = useCallback(
    (question: string) => {
      const text = question.trim();
      if (!text || isWorking) return;
      lastQuestion.current = text;

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
      runQuery(text, pendingId);
    },
    [isWorking, runQuery],
  );

  const retryLast = useCallback(() => {
    if (isWorking || !lastQuestion.current) return;
    const pendingId = uid("msg");
    setMessages((prev) => [
      ...prev,
      {
        id: pendingId,
        role: "assistant",
        content: "",
        pending: true,
        createdAt: new Date().toISOString(),
      },
    ]);
    runQuery(lastQuestion.current, pendingId);
  }, [isWorking, runQuery]);

  const togglePin = useCallback((id: string) => {
    setArtifacts((prev) => prev.map((a) => (a.id === id ? { ...a, pinned: !a.pinned } : a)));
  }, []);

  const setReportingDate = useCallback(
    (next: string) => {
      setAsOf(next);
      // Refresh non-pinned landing artifacts for the new date when idle.
      setArtifacts((prev) => {
        const pinned = prev.filter((a) => a.pinned);
        if (prev.length === pinned.length) return prev;
        return prev;
      });
    },
    [],
  );

  const resetWorkspace = useCallback(() => {
    abortRef.current?.abort();
    setMessages([greeting(portfolio, asOf)]);
    setArtifacts(client.mock ? landingArtifacts({ asOf }, portfolioId) : []);
  }, [client.mock, portfolio, asOf, portfolioId]);

  return {
    portfolio,
    reporting,
    messages,
    artifacts,
    isWorking,
    setPortfolio: setPortfolioId,
    setReportingDate,
    ask,
    retryLast,
    togglePin,
    resetWorkspace,
  };
}
