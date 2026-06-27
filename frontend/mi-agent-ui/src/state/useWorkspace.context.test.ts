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

function failureResult(question: string): AgentResponse {
  return { ok: false, question, intent: "unknown", narrative: "no", assumptions: [], warnings: [], artifacts: [] };
}

function makeClient(ask: (req: AgentRequest) => Promise<AgentResponse>): AgentClient {
  return {
    id: "fake",
    mock: true,
    ask: (req) => ask(req),
    getSnapshots: async () => INDEX,
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
  };
}

beforeEach(() => {
  localStorage.clear();
});

describe("useWorkspace — analysis context", () => {
  it("updates context only after a successful query", async () => {
    const ask = vi.fn(async (req: AgentRequest) =>
      req.question.includes("fail") ? failureResult(req.question) : regionResult(req.question),
    );
    const { result } = renderHook(() => useWorkspace(makeClient(ask)));
    await waitFor(() => expect(result.current.selectedRunId).toBe("mi_2025_11"));

    act(() => result.current.ask("show balance by region"));
    await waitFor(() => expect(result.current.context).not.toBeNull());
    expect(result.current.context?.activeMeasure).toBe("current_outstanding_balance");
    expect(result.current.context?.activeDimensions).toEqual(["geographic_region_obligor"]);

    // A failed query must NOT overwrite the existing context.
    act(() => result.current.ask("fail please"));
    await waitFor(() => expect(result.current.isWorking).toBe(false));
    expect(result.current.context?.activeMeasure).toBe("current_outstanding_balance");
  });

  it("renders a conversational answer even when the backend returns parser/debug text", async () => {
    const DEBUG = "Chart: Bar · Metric: Balance · Dimension: Region · Aggregation: Sum · Parser: deterministic · Validation: Passed — 3 group(s).";
    const ask = vi.fn(async (req: AgentRequest) => ({ ...regionResult(req.question), narrative: DEBUG, interpreted: DEBUG }));
    const { result } = renderHook(() => useWorkspace(makeClient(ask)));
    await waitFor(() => expect(result.current.selectedRunId).toBe("mi_2025_11"));

    act(() => result.current.ask("what is the concentration in London"));
    await waitFor(() => expect(result.current.messages.some((m) => m.role === "assistant" && !m.pending && m.artifacts?.length)).toBe(true));

    const answer = [...result.current.messages].reverse().find((m) => m.role === "assistant" && !m.pending)!;
    // The visible content must be plain English, never the parser/validation dump.
    expect(answer.content).not.toContain("Parser");
    expect(answer.content).not.toContain("Validation");
    expect(answer.content).not.toContain("Aggregation");
    expect(answer.content.toLowerCase()).toContain("balance"); // conversational lead
    // The result is embedded on the message for inline rendering.
    expect(answer.artifacts?.[0]?.type).toBe("chart");
    // The raw interpretation is still retained for the Query Logic disclosure.
    expect(answer.interpreted).toBeDefined();
  });

  it("resolves a follow-up against context and dispatches the rewritten query", async () => {
    const ask = vi.fn(async (req: AgentRequest) => regionResult(req.question));
    const { result } = renderHook(() => useWorkspace(makeClient(ask)));
    await waitFor(() => expect(result.current.selectedRunId).toBe("mi_2025_11"));

    act(() => result.current.ask("show balance by region"));
    await waitFor(() => expect(result.current.context).not.toBeNull());

    act(() => result.current.ask("split by broker"));
    await waitFor(() => expect(ask).toHaveBeenCalledTimes(2));
    // The second call carries the context-resolved standalone question.
    expect(ask.mock.calls[1][0].question).toBe("Balance by Broker");
  });

  it("clears context on demand", async () => {
    const ask = vi.fn(async (req: AgentRequest) => regionResult(req.question));
    const { result } = renderHook(() => useWorkspace(makeClient(ask)));
    await waitFor(() => expect(result.current.selectedRunId).toBe("mi_2025_11"));

    act(() => result.current.ask("show balance by region"));
    await waitFor(() => expect(result.current.context).not.toBeNull());
    act(() => result.current.clearContext());
    expect(result.current.context).toBeNull();
  });
});
