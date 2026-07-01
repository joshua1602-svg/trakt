import { renderHook, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it } from "vitest";
import type { AgentClient } from "@/api";
import { useWorkspace } from "./useWorkspace";

// The EXACT live /mi/snapshots response for a platform canonical with a single
// funded book — proves the store binds source_portfolio_id + runs[].reporting_date
// and auto-selects direct_001 / latest.
const INDEX = {
  portfolios: [
    {
      client_id: "direct_001",
      label: "direct_001",
      source_portfolio_id: "direct_001",
      runs: [
        {
          run_id: "latest",
          reporting_date: "2025-12-11",
          loan_count: 73,
          current_outstanding_balance: 8903225.07,
        },
      ],
    },
  ],
  source: "platform_canonical_typed.csv",
};

function makeClient(): AgentClient {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const any = async () => ({}) as any;
  return {
    id: "test",
    mock: false,
    ask: any,
    getSnapshots: async () => INDEX as unknown as Awaited<ReturnType<AgentClient["getSnapshots"]>>,
    getSourcePortfolios: async () => ({
      available: true,
      lenses: [{ id: "direct_001", kind: "cohort", label: "direct_001", filters: {}, funded_only: false }],
      source: "platform_canonical_typed.csv",
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
    }) as any,
    getSnapshot: any,
    getForecastSnapshot: any,
    getFundedEvolution: any,
    getPipelineEvolution: any,
    getForecastEvolution: any,
    getFunnelEvolution: any,
    getRiskLimits: any,
    getForecastExtrapolation: any,
  };
}

const KEY = "trakt.mi-agent.workspace.v3";

beforeEach(() => localStorage.clear());

describe("useWorkspace — platform-canonical /mi/snapshots binding", () => {
  it("auto-selects the only portfolio (direct_001) and its latest run", async () => {
    const { result } = renderHook(() => useWorkspace(makeClient()));
    await waitFor(() => expect(result.current.selectedClientId).toBe("direct_001"));
    expect(result.current.selectedRunId).toBe("latest");
    // portfolio dropdown source: client_id / source_portfolio_id from the response
    expect(result.current.portfolios).toHaveLength(1);
    expect(result.current.portfolios[0].client_id).toBe("direct_001");
    // reporting-date dropdown source: runs[].reporting_date
    expect(result.current.runs).toHaveLength(1);
    expect(result.current.runs[0].run_id).toBe("latest");
    expect(result.current.runs[0].reporting_date).toBe("2025-12-11");
  });

  it("selects the sole portfolio even when a stale selection is persisted", async () => {
    localStorage.setItem(
      KEY,
      JSON.stringify({ clientId: "client_001", runId: "mi_2025_11", messages: [], artifacts: [] }),
    );
    const { result } = renderHook(() => useWorkspace(makeClient()));
    await waitFor(() => expect(result.current.selectedClientId).toBe("direct_001"));
    expect(result.current.selectedRunId).toBe("latest");
  });
});
