/**
 * The governed portfolio context in the React workspace.
 *
 * These tests pin the architectural rule that the UI is a PRESENTATION layer:
 * one selection drives everything, the backend decides what a group contains and
 * what it may do, and the workspace never quietly changes scope or re-derives a
 * business rule for itself.
 */

import { beforeEach, describe, expect, it, vi } from "vitest";
import { act, renderHook, waitFor } from "@testing-library/react";
import type { AgentClient } from "@/api";
import type {
  AgentRequest,
  AgentResponse,
  PortfolioCapabilities,
  PortfolioContextIndex,
} from "@/domain";
import { useWorkspace } from "./useWorkspace";

const INDEX = {
  portfolios: [{
    client_id: "ERE",
    label: "ERE",
    runs: [{ run_id: "mi_2025_12", reporting_date: "2025-12-31" }],
  }],
};

/** Capability sets exactly as the governed backend emits them. */
function caps(over: Partial<Record<string, boolean>> = {},
              detail: Record<string, string> = {}): PortfolioCapabilities {
  const build = (id: string): [string, unknown] => [id, {
    capability: id,
    enabled: over[id] ?? true,
    reason_code: (over[id] ?? true) ? null : "NON_ORIGINATING",
    detail: detail[id] ?? null,
    contributing_portfolios: [],
    excluded_portfolios: [],
    partial: false,
  }];
  return Object.fromEntries([
    "funded", "risk", "evolution", "cohorts", "pipeline",
    "origination_forecast", "runoff_forecast", "consolidated_forecast",
    "drill_through", "export",
  ].map(build)) as PortfolioCapabilities;
}

/** A four-portfolio governed hierarchy: two direct books, two acquired. */
const CONTEXT_INDEX: PortfolioContextIndex = {
  available: true,
  client_id: "ERE",
  default_context_id: "total",
  portfolio_types: ["direct", "acquired"],
  pipeline_portfolios: ["direct_001", "direct_002"],
  portfolios: [],
  contexts: [
    {
      context_id: "total", context_kind: "total", label: "Total",
      parent_id: null, depth: 0,
      portfolio_ids: ["direct_001", "direct_002", "acquired_001", "acquired_002"],
      portfolio_types: ["direct", "acquired"],
      capabilities: caps({}, {
        pipeline: "Pipeline is sourced from originating portfolios only. "
          + "Included: direct_001, direct_002. Excluded — non-originating: "
          + "acquired_001, acquired_002.",
      }),
    },
    {
      context_id: "direct", context_kind: "type", label: "Direct",
      parent_id: "total", depth: 1,
      portfolio_ids: ["direct_001", "direct_002"], portfolio_types: ["direct"],
      capabilities: caps(),
    },
    {
      context_id: "direct_001", context_kind: "portfolio", label: "Direct 001",
      parent_id: "direct", depth: 2,
      portfolio_ids: ["direct_001"], portfolio_types: ["direct"],
      capabilities: caps(),
    },
    {
      context_id: "direct_002", context_kind: "portfolio", label: "Direct 002",
      parent_id: "direct", depth: 2,
      portfolio_ids: ["direct_002"], portfolio_types: ["direct"],
      capabilities: caps(),
    },
    {
      context_id: "acquired", context_kind: "type", label: "Acquired",
      parent_id: "total", depth: 1,
      portfolio_ids: ["acquired_001", "acquired_002"], portfolio_types: ["acquired"],
      capabilities: caps({ pipeline: false, origination_forecast: false }, {
        pipeline: "No portfolio in this scope originates new lending, so there "
          + "is no origination pipeline to report.",
      }),
    },
    {
      context_id: "acquired_001", context_kind: "portfolio", label: "Acquired 001",
      parent_id: "acquired", depth: 2,
      portfolio_ids: ["acquired_001"], portfolio_types: ["acquired"],
      capabilities: caps({ pipeline: false, origination_forecast: false,
                           consolidated_forecast: false }, {
        pipeline: "acquired_001 does not originate new lending.",
        consolidated_forecast: "No forecast applies to acquired_001.",
      }),
    },
    {
      context_id: "acquired_002", context_kind: "portfolio", label: "Acquired 002",
      parent_id: "acquired", depth: 2,
      portfolio_ids: ["acquired_002"], portfolio_types: ["acquired"],
      capabilities: caps({ pipeline: false, origination_forecast: false }),
    },
  ],
};

function ok(question: string): AgentResponse {
  return {
    ok: true, question, intent: "portfolio_overview", narrative: "answer",
    assumptions: [], warnings: [], artifacts: [],
    portfolioCoverage: {
      scope_requested: { context_id: "total", context_kind: "total", label: "Total" },
      portfolios_in_scope: ["direct_001", "direct_002", "acquired_001"],
      portfolios_used: ["direct_001", "acquired_001"],
      portfolios_excluded: {
        direct_002: { reason_code: "FIELD_NOT_AVAILABLE", detail: "borrower_age is not available" },
      },
      is_fully_consolidated: false,
      field_coverage: {
        borrower_age: {
          available_in: ["direct_001", "acquired_001"],
          unavailable_in: ["direct_002"],
        },
      },
      disclosure: "Answered from direct_001, acquired_001. Excluded direct_002: "
        + "borrower_age is not available. This is NOT a fully consolidated Total answer.",
    },
  };
}

function makeClient(over: Partial<AgentClient> = {}): AgentClient {
  const stub = async () => ({}) as never;
  return {
    id: "fake",
    mock: false,
    ask: vi.fn(async (req: AgentRequest) => ok(req.question)),
    getSnapshots: async () => INDEX as never,
    getPortfolioContext: vi.fn(async () => CONTEXT_INDEX),
    getSourcePortfolios: async () => ({ available: false, lenses: [], source: "test" }),
    getSnapshot: vi.fn(stub),
    getForecastSnapshot: vi.fn(stub),
    getFundedEvolution: stub,
    getPipelineEvolution: stub,
    getForecastEvolution: stub,
    getFunnelEvolution: stub,
    getRiskLimits: stub,
    getForecastExtrapolation: stub,
    getCohortProgression: stub,
    getMe: async () => ({ authenticated: false }),
    getDecks: stub,
    deckDownloadUrl: () => null,
    getCohorts: stub,
    getGeoExposure: stub,
    ...over,
  } as AgentClient;
}

beforeEach(() => {
  localStorage.clear();
});

describe("useWorkspace — governed portfolio context", () => {
  it("builds the selector from the governed hierarchy, not from hard-coded ids", async () => {
    const { result } = renderHook(() => useWorkspace(makeClient()));
    await waitFor(() => expect(result.current.portfolioContexts.length).toBe(7));
    expect(result.current.portfolioContexts.map((c) => c.context_id)).toEqual([
      "total", "direct", "direct_001", "direct_002",
      "acquired", "acquired_001", "acquired_002",
    ]);
    // Both direct books and both acquired books are offered — nothing assumes
    // a single portfolio per type.
    expect(result.current.portfolioContexts.filter((c) => c.depth === 2)).toHaveLength(4);
  });

  it("defaults to Total and exposes the backend's Total membership", async () => {
    const { result } = renderHook(() => useWorkspace(makeClient()));
    await waitFor(() => expect(result.current.activeContext).not.toBeNull());
    expect(result.current.selectedContextId).toBe("total");
    expect(result.current.activeContext!.portfolio_ids).toEqual([
      "direct_001", "direct_002", "acquired_001", "acquired_002",
    ]);
  });

  it("sends the SAME context to every scoped fetch and to the chat", async () => {
    const client = makeClient();
    const { result } = renderHook(() => useWorkspace(client));
    await waitFor(() => expect(result.current.portfolioContexts.length).toBe(7));

    act(() => result.current.setSelectedContextId("acquired"));
    await waitFor(() =>
      expect(client.getSnapshot).toHaveBeenCalledWith(expect.any(String), "acquired"));
    expect(client.getForecastSnapshot).toHaveBeenCalledWith(expect.any(String), "acquired");

    act(() => result.current.ask("What is the weighted average LTV?"));
    await waitFor(() => expect(client.ask).toHaveBeenCalled());
    const request = (client.ask as ReturnType<typeof vi.fn>).mock.calls.at(-1)![0];
    expect(request.sourceLens).toBe("acquired");
  });

  it("keeps the selected context when the view changes", async () => {
    const { result } = renderHook(() => useWorkspace(makeClient()));
    await waitFor(() => expect(result.current.portfolioContexts.length).toBe(7));
    act(() => result.current.setSelectedContextId("direct_002"));
    act(() => result.current.setActiveView("funded"));
    act(() => result.current.setActiveView("risk_limits"));
    expect(result.current.selectedContextId).toBe("direct_002");
  });

  it("disables Pipeline for a non-originating acquired book, with the backend reason", async () => {
    const { result } = renderHook(() => useWorkspace(makeClient()));
    await waitFor(() => expect(result.current.portfolioContexts.length).toBe(7));
    act(() => result.current.setSelectedContextId("acquired_001"));
    await waitFor(() => expect(result.current.disabledViews).toContain("pipeline"));
    expect(result.current.capabilityEnabled("pipeline")).toBe(false);
    expect(result.current.capabilityEnabled("origination_forecast")).toBe(false);
    expect(result.current.disabledViewReasons.pipeline)
      .toBe("acquired_001 does not originate new lending.");
  });

  it("keeps funded analytical depth for acquired books", async () => {
    const { result } = renderHook(() => useWorkspace(makeClient()));
    await waitFor(() => expect(result.current.portfolioContexts.length).toBe(7));
    act(() => result.current.setSelectedContextId("acquired_002"));
    await waitFor(() => expect(result.current.activeContext?.context_id).toBe("acquired_002"));
    for (const cap of ["funded", "risk", "evolution", "cohorts", "drill_through", "export"] as const) {
      expect(result.current.capabilityEnabled(cap)).toBe(true);
    }
  });

  it("keeps Pipeline available in Total and carries the originating-only disclosure", async () => {
    const { result } = renderHook(() => useWorkspace(makeClient()));
    await waitFor(() => expect(result.current.portfolioContexts.length).toBe(7));
    expect(result.current.selectedContextId).toBe("total");
    expect(result.current.capabilityEnabled("pipeline")).toBe(true);
    expect(result.current.capabilityReason("pipeline"))
      .toContain("Included: direct_001, direct_002");
  });

  it("NEVER switches the workspace away from a scope whose view is disabled", async () => {
    const { result } = renderHook(() => useWorkspace(makeClient()));
    await waitFor(() => expect(result.current.portfolioContexts.length).toBe(7));
    act(() => result.current.setActiveView("pipeline"));
    act(() => result.current.setSelectedContextId("acquired_001"));
    await waitFor(() => expect(result.current.disabledViews).toContain("pipeline"));
    // The view and the context both stand: the tab is disabled and explained,
    // rather than the app silently moving the user somewhere else.
    expect(result.current.activeView).toBe("pipeline");
    expect(result.current.selectedContextId).toBe("acquired_001");
  });

  it("carries the backend coverage disclosure onto the chat message verbatim", async () => {
    const client = makeClient();
    const { result } = renderHook(() => useWorkspace(client));
    await waitFor(() => expect(result.current.portfolioContexts.length).toBe(7));
    act(() => result.current.ask("Show average borrower age."));
    await waitFor(() => {
      const last = result.current.messages.at(-1)!;
      expect(last.pending).toBeFalsy();
    });
    const coverage = result.current.messages.at(-1)!.portfolioCoverage!;
    expect(coverage.is_fully_consolidated).toBe(false);
    expect(coverage.portfolios_excluded.direct_002.reason_code).toBe("FIELD_NOT_AVAILABLE");
    expect(coverage.disclosure).toContain("NOT a fully consolidated");
  });

  it("falls back to a single Total scope when no governed hierarchy exists", async () => {
    const client = makeClient({
      getPortfolioContext: vi.fn(async () => ({
        available: false, client_id: null, default_context_id: "total",
        contexts: [], portfolios: [], portfolio_types: [], pipeline_portfolios: null,
      })),
    });
    const { result } = renderHook(() => useWorkspace(client));
    await waitFor(() => expect(client.getPortfolioContext).toHaveBeenCalled());
    expect(result.current.portfolioContexts).toEqual([]);
    // No capability entries → nothing is gated off, so a deployment without the
    // contract keeps its existing navigation.
    expect(result.current.disabledViews).toEqual([]);
    expect(result.current.capabilityEnabled("pipeline")).toBe(true);
  });
});
