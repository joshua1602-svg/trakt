import { act, renderHook, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { AgentClient } from "@/api";
import type { AgentRequest, AgentResponse } from "@/domain";
import { useWorkspace } from "./useWorkspace";

const INDEX = {
  portfolios: [
    {
      client_id: "client_001",
      label: "ERM",
      runs: [{ run_id: "mi_2025_11", reporting_date: "2025-11-30", loan_count: 73, current_outstanding_balance: 1 }],
    },
  ],
};

function regionResult(question: string): AgentResponse {
  return {
    ok: true,
    question,
    intent: "concentration_risk",
    narrative: "ok",
    assumptions: [],
    warnings: [],
    spec: { metric: "current_outstanding_balance", dimension: "geographic_region_obligor" },
    artifacts: [
      {
        id: "a1",
        type: "chart",
        title: "Balance by Region",
        source: {
          engine: "mi_agent.workflow",
          label: "MI Agent · bar",
          spec: { metric: "current_outstanding_balance", dimension: "geographic_region_obligor" },
        },
        createdAt: "2026-06-26T08:00:00Z",
        mock: false,
        chartType: "bar",
        xKey: "geographic_region_obligor",
        series: [{ key: "current_outstanding_balance", label: "Balance", color: "#000" }],
        rows: [{ geographic_region_obligor: "London", current_outstanding_balance: 400 }],
      },
    ],
  };
}

function makeClient(ask: (req: AgentRequest) => Promise<AgentResponse>): AgentClient {
  return {
    id: "fake",
    mock: true,
    ask: (req) => ask(req),
    getSnapshots: async () => INDEX,
    getPortfolioContext: async () => ({ available: false, client_id: null,
      default_context_id: "total", contexts: [], portfolios: [],
      portfolio_types: [], pipeline_portfolios: null }),
    getSourcePortfolios: async () => ({ available: false, lenses: [], source: "test" }),
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    getSnapshot: async () => ({}) as any,
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    getForecastSnapshot: async () => ({}) as any,
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    getFundedEvolution: async () => ({}) as any,
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    getPipelineEvolution: async () => ({}) as any,
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    getForecastEvolution: async () => ({}) as any,
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    getFunnelEvolution: async () => ({}) as any,
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    getRiskLimits: async () => ({}) as any,
    getConcentrationTests: async () => ({}) as any,
    getConcentrationDrillthrough: async () => ({}) as any,
    getConcentrationHistory: async () => ({}) as any,
    getConcentrationDrivers: async () => ({}) as any,
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    getForecastExtrapolation: async () => ({}) as any,
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    getCohortProgression: async () => ({}) as any,
    getCohortVintages: () => Promise.resolve({ dataset: "cohort_formation", portfolioId: "p", cohortBasis: "origination_date", grain: "M", available: true, vintages: [] }) as never,
    getMe: async () => ({ authenticated: false }),
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    getDecks: async () => ({ available: false, latest: null, decks: [], client_id: "" }) as any,
    deckDownloadUrl: () => null,
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    getCohorts: async () => ({ available: false, cohorts: [] }) as any,
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    getGeoExposure: async () => ({ available: false, areas: [] }) as any,
  };
}

beforeEach(() => {
  localStorage.clear();
});

const withMemory = (question: string, continuation: string, readAs?: string): AgentResponse => ({
  ...regionResult(question),
  conversation: { kind: "follow_up", continuation, expiresInSeconds: 300, ...(readAs ? { readAs } : {}) },
});

/**
 * The governed conversation (P0 design §38, §39). When the server holds the
 * conversation, a follow-up is sent AS TYPED with the memory the last answer
 * handed back, and the server reads it with the question before it. The
 * browser's own rewriting is used only where the server held no memory.
 */
describe("useWorkspace — the governed conversation", () => {
  it("sends the server's memory with the next message, as typed", async () => {
    const ask = vi.fn(async (req: AgentRequest) => withMemory(req.question, `tok-${ask.mock.calls.length}`));
    const { result } = renderHook(() => useWorkspace(makeClient(ask)));
    await waitFor(() => expect(result.current.selectedRunId).toBe("mi_2025_11"));

    act(() => result.current.ask("show balance by region"));
    await waitFor(() => expect(result.current.isWorking).toBe(false));
    act(() => result.current.ask("split by broker"));
    await waitFor(() => expect(ask).toHaveBeenCalledTimes(2));

    const [first, second] = ask.mock.calls.map((c) => c[0]);
    expect(first.continuation).toBeUndefined();
    expect(second.question).toBe("split by broker");         // never rewritten here
    expect(second.continuation).toBe("tok-1");
    expect(second.conversationId).toBe(first.conversationId);
  });

  it("records what the server read a follow-up as", async () => {
    const ask = vi.fn(async (req: AgentRequest) => withMemory(
      req.question, "tok", ask.mock.calls.length > 1 ? "Show balance by broker." : undefined));
    const { result } = renderHook(() => useWorkspace(makeClient(ask)));
    await waitFor(() => expect(result.current.selectedRunId).toBe("mi_2025_11"));

    act(() => result.current.ask("show balance by region"));
    await waitFor(() => expect(result.current.isWorking).toBe(false));
    act(() => result.current.ask("and by broker?"));
    await waitFor(() => expect(ask).toHaveBeenCalledTimes(2));
    await waitFor(() => expect(result.current.isWorking).toBe(false));

    const answer = [...result.current.messages].reverse().find((m) => m.role === "assistant" && !m.pending)!;
    expect(answer.usedContext).toBe(true);
    expect(answer.contextNote).toBe("Read as “Show balance by broker.”");
  });

  it("an answer that hands back no memory ends it: the next message is read on its own", async () => {
    const ask = vi.fn(async (req: AgentRequest) =>
      ask.mock.calls.length === 1 ? withMemory(req.question, "tok-1") : regionResult(req.question));
    const { result } = renderHook(() => useWorkspace(makeClient(ask)));
    await waitFor(() => expect(result.current.selectedRunId).toBe("mi_2025_11"));

    for (const q of ["show balance by region", "and by broker?", "what is the funded balance?"]) {
      act(() => result.current.ask(q));
      await waitFor(() => expect(result.current.isWorking).toBe(false));
    }
    expect(ask.mock.calls[1][0].continuation).toBe("tok-1");
    expect(ask.mock.calls[2][0].continuation).toBeUndefined();
  });

  it("clearing the chat starts a new conversation", async () => {
    const ask = vi.fn(async (req: AgentRequest) => withMemory(req.question, "tok-1"));
    const { result } = renderHook(() => useWorkspace(makeClient(ask)));
    await waitFor(() => expect(result.current.selectedRunId).toBe("mi_2025_11"));

    act(() => result.current.ask("show balance by region"));
    await waitFor(() => expect(result.current.isWorking).toBe(false));
    act(() => result.current.clearChat());
    act(() => result.current.ask("and by broker?"));
    await waitFor(() => expect(ask).toHaveBeenCalledTimes(2));

    const [first, second] = ask.mock.calls.map((c) => c[0]);
    expect(second.continuation).toBeUndefined();
    expect(second.conversationId).not.toBe(first.conversationId);
  });

  it("where the server holds no memory, the browser's own follow-up reading still applies", async () => {
    const ask = vi.fn(async (req: AgentRequest) => regionResult(req.question));
    const { result } = renderHook(() => useWorkspace(makeClient(ask)));
    await waitFor(() => expect(result.current.selectedRunId).toBe("mi_2025_11"));

    act(() => result.current.ask("show balance by region"));
    await waitFor(() => expect(result.current.context).not.toBeNull());
    act(() => result.current.ask("split by broker"));
    await waitFor(() => expect(ask).toHaveBeenCalledTimes(2));
    expect(ask.mock.calls[1][0].question).toBe("Balance by Broker");
    expect(ask.mock.calls[1][0].continuation).toBeUndefined();
  });
});
