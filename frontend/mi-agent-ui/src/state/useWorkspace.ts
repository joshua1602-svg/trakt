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
import type {
  AgentRequest,
  Artifact,
  ChatMessage,
  ForecastSnapshot,
  FundedSnapshot,
  PortfolioContext,
  ReportingContext,
  SnapshotPortfolio,
  SnapshotRun,
  WorkspaceView,
} from "@/domain";
import type { AgentClient } from "@/api";
import { uid } from "@/lib/utils";
import { loadState, saveState } from "./persistence";

function greeting(portfolioLabel: string, asOf: string | null): ChatMessage {
  const date = asOf
    ? new Date(asOf).toLocaleDateString("en-GB", { day: "numeric", month: "long", year: "numeric" })
    : "the latest reporting date";
  return {
    id: "greeting",
    role: "assistant",
    content: `Welcome back. I'm your MI Agent for the ${portfolioLabel} funded portfolio, as of ${date}. The funded book snapshot is on the right — ask me about portfolio movement, concentration, stratifications, risk monitoring or validation to go deeper.`,
    createdAt: new Date().toISOString(),
  };
}

/** Stamp the originating question onto each artifact so a drill-through can
 *  re-run the same query with an added filter. */
function stampQuestion(artifacts: Artifact[], question: string): Artifact[] {
  return artifacts.map((a) => ({ ...a, source: { ...a.source, question } }));
}

export interface Workspace {
  /** Discovered funded portfolios (data-driven dropdown source). */
  portfolios: SnapshotPortfolio[];
  /** Reporting runs available for the selected portfolio. */
  runs: SnapshotRun[];
  selectedClientId: string | null;
  selectedRunId: string | null;
  portfolio: PortfolioContext;
  reporting: ReportingContext;
  /** Deterministic landing-page funded snapshot for the selected run. */
  snapshot: FundedSnapshot | null;
  snapshotLoading: boolean;
  /** Deterministic funded + pipeline forecast snapshot for the selected run. */
  forecast: ForecastSnapshot | null;
  forecastLoading: boolean;
  /** Active workspace view (funded | pipeline | forecast). */
  activeView: WorkspaceView;
  setActiveView: (view: WorkspaceView) => void;
  messages: ChatMessage[];
  artifacts: Artifact[];
  isWorking: boolean;
  setPortfolio: (clientId: string) => void;
  setRun: (runId: string) => void;
  ask: (question: string) => void;
  /** Re-run an artifact's query with an added drill-through filter (backend). */
  drill: (artifact: Artifact, filters: Record<string, unknown>) => void;
  retryLast: () => void;
  togglePin: (id: string) => void;
  resetWorkspace: () => void;
}

export function useWorkspace(client: AgentClient): Workspace {
  const persisted = useRef(loadState()).current;

  const [portfolios, setPortfolios] = useState<SnapshotPortfolio[]>([]);
  const [selectedClientId, setSelectedClientId] = useState<string | null>(persisted?.clientId ?? null);
  const [selectedRunId, setSelectedRunId] = useState<string | null>(persisted?.runId ?? null);
  const [snapshot, setSnapshot] = useState<FundedSnapshot | null>(null);
  const [snapshotLoading, setSnapshotLoading] = useState(false);
  const [forecast, setForecast] = useState<ForecastSnapshot | null>(null);
  const [forecastLoading, setForecastLoading] = useState(false);
  // Active view defaults to Funded. A ref mirrors it so runQuery reads the latest
  // value without being re-created on every view change.
  const [activeView, setActiveViewState] = useState<WorkspaceView>("funded");
  const activeViewRef = useRef<WorkspaceView>("funded");
  const setActiveView = useCallback((view: WorkspaceView) => {
    activeViewRef.current = view;
    setActiveViewState(view);
  }, []);

  // Discover available portfolios / runs once; default to the latest run.
  useEffect(() => {
    let cancelled = false;
    client
      .getSnapshots()
      .then((index) => {
        if (cancelled) return;
        const pfs = index.portfolios ?? [];
        setPortfolios(pfs);
        if (pfs.length === 0) return;
        const persistedPf = pfs.find((p) => p.client_id === persisted?.clientId);
        const pf = persistedPf ?? pfs[pfs.length - 1];
        setSelectedClientId((cur) => cur ?? pf.client_id);
        const run =
          (persistedPf && pf.runs.find((r) => r.run_id === persisted?.runId)) ??
          pf.runs[pf.runs.length - 1];
        setSelectedRunId((cur) => cur ?? run?.run_id ?? null);
      })
      .catch(() => {
        if (!cancelled) setPortfolios([]);
      });
    return () => {
      cancelled = true;
    };
  }, [client, persisted?.clientId, persisted?.runId]);

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
      .getSnapshot(portfolioId)
      .then((snap) => {
        if (!cancelled) setSnapshot(snap);
      })
      .catch(() => {
        if (!cancelled) setSnapshot(null);
      })
      .finally(() => {
        if (!cancelled) setSnapshotLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [client, portfolioId]);

  // Fetch the deterministic forecast (funded + pipeline) for the same run, so the
  // pipeline + forecast + watchlist sections move with the funded selection.
  useEffect(() => {
    if (!portfolioId) return;
    let cancelled = false;
    setForecastLoading(true);
    client
      .getForecastSnapshot(portfolioId)
      .then((fc) => {
        if (!cancelled) setForecast(fc);
      })
      .catch(() => {
        if (!cancelled) setForecast(null);
      })
      .finally(() => {
        if (!cancelled) setForecastLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [client, portfolioId]);

  const [messages, setMessages] = useState<ChatMessage[]>(
    () => persisted?.messages ?? [greeting("selected", persisted?.runId ? null : null)],
  );
  const [artifacts, setArtifacts] = useState<Artifact[]>(
    () => (persisted?.artifacts ?? []).filter((a) => !a.mock),
  );
  const [isWorking, setIsWorking] = useState(false);

  const lastQuestion = useRef<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    saveState({
      clientId: selectedClientId ?? undefined,
      runId: selectedRunId ?? undefined,
      messages,
      artifacts,
    });
  }, [selectedClientId, selectedRunId, messages, artifacts]);

  const runQuery = useCallback(
    (question: string, pendingId: string) => {
      const request: AgentRequest = {
        question,
        portfolio,
        reporting,
        options: { parserMode: "deterministic" },
        // Tab-aware: the active view sets the dataset context (funded / pipeline
        // / forecast); explicit wording in the question overrides it backend-side.
        datasetContext: activeViewRef.current,
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
            const fresh = stampQuestion(res.artifacts.filter((a) => !pinnedIds.has(a.id)), question);
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
                    diagnostics: res.diagnostics,
                    intent: res.intent,
                    spec: res.spec,
                    confidence: res.confidence,
                    artifactRefs: res.artifacts.map((a) => ({ id: a.id, title: a.title, type: a.type })),
                  }
                : m,
            ),
          );
        })
        .catch((err: unknown) => {
          const message = err instanceof Error ? err.message : "The MI Agent could not complete this request.";
          setMessages((prev) =>
            prev.map((m) => (m.id === pendingId ? { ...m, pending: false, error: true, content: message } : m)),
          );
        })
        .finally(() => {
          setIsWorking(false);
          abortRef.current = null;
        });
    },
    [client, portfolio, reporting],
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
      };
      const controller = new AbortController();
      abortRef.current = controller;
      setIsWorking(true);
      client
        .ask(request, controller.signal)
        .then((res) => {
          if (!res.ok) return; // keep current artifacts; client-side panel remains
          setArtifacts((prev) => {
            const pinned = prev.filter((a) => a.pinned);
            const pinnedIds = new Set(pinned.map((a) => a.id));
            const fresh = stampQuestion(res.artifacts.filter((a) => !pinnedIds.has(a.id)), question);
            return [...pinned, ...fresh];
          });
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
      { id: pendingId, role: "assistant", content: "", pending: true, createdAt: new Date().toISOString() },
    ]);
    runQuery(lastQuestion.current, pendingId);
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
    },
    [portfolios],
  );

  const resetWorkspace = useCallback(() => {
    abortRef.current?.abort();
    setMessages([greeting(activePortfolio?.label ?? "selected", activeRun?.reporting_date ?? null)]);
    setArtifacts([]);
  }, [activePortfolio?.label, activeRun?.reporting_date]);

  return {
    portfolios,
    runs,
    selectedClientId: activePortfolio?.client_id ?? null,
    selectedRunId: activeRun?.run_id ?? null,
    portfolio,
    reporting,
    snapshot,
    snapshotLoading,
    forecast,
    forecastLoading,
    activeView,
    setActiveView,
    messages,
    artifacts,
    isWorking,
    setPortfolio,
    setRun: setSelectedRunId,
    ask,
    drill,
    retryLast,
    togglePin,
    resetWorkspace,
  };
}
