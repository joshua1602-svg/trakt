import { afterEach, describe, expect, it, vi } from "vitest";
import type { AgentRequest } from "@/domain";
import { AgentError, HttpAgentClient } from "./index";

const request: AgentRequest = {
  question: "Show balance by region",
  portfolio: { id: "erm-uk-master", name: "ERM UK — Master", entity: "Trakt SPV I" },
  reporting: { asOf: "2026-05-31" },
};

const apiBody = {
  ok: true,
  question: "Show balance by region",
  answer: "Bar chart of Balance by Region — 10 group(s).",
  interpreted: "Chart: Bar · Metric: Balance · Dimension: Region",
  spec: { chart_type: "bar", dimension: "geographic_region_obligor" },
  validation: { ok: true, errors: [], warnings: [] },
  warnings: [],
  assumptions: [],
  artifacts: [
    {
      id: "art_1",
      type: "chart",
      title: "Balance by Region",
      createdAt: "2026-05-31T08:00:00Z",
      mock: false,
      source: { engine: "mi_agent.workflow", label: "MI Agent · bar" },
      chartType: "bar",
      xKey: "geographic_region_obligor",
      series: [{ key: "current_outstanding_balance_sum", label: "Balance", color: "#919dd1" }],
      rows: [{ geographic_region_obligor: "London", current_outstanding_balance_sum: 184 }],
    },
  ],
  metadata: { engine: "mi_agent", source: "python", mock: false },
};

afterEach(() => vi.restoreAllMocks());

describe("HttpAgentClient", () => {
  it("maps a successful API response into an AgentResponse", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => new Response(JSON.stringify(apiBody), { status: 200 })),
    );
    const client = new HttpAgentClient("http://localhost:8000");
    const res = await client.ask(request);

    expect(client.mock).toBe(false);
    expect(res.ok).toBe(true);
    expect(res.narrative).toContain("Bar chart");
    expect(res.interpreted).toContain("Metric: Balance");
    expect(res.intent).toBe("concentration_risk");
    expect(res.artifacts).toHaveLength(1);
    expect(res.artifacts[0].type).toBe("chart");
  });

  it("normalises the snake_case wire spec to the camelCase MIQuerySpec shape", async () => {
    const body = {
      ...apiBody,
      spec: { intent: "chart", chart_type: "bar", top_n: 10,
              risk_monitor_mode: "concentration", metric: "current_outstanding_balance",
              dimension: "broker_channel" },
    };
    vi.stubGlobal("fetch", vi.fn(async () => new Response(JSON.stringify(body), { status: 200 })));
    const res = await new HttpAgentClient("http://localhost:8000").ask(request);
    // camelCase fields the MIQuerySpec type promises are populated...
    expect(res.spec?.chartType).toBe("bar");
    expect(res.spec?.topN).toBe(10);
    expect(res.spec?.riskMode).toBe("concentration");
    expect(res.spec?.metric).toBe("current_outstanding_balance");
    // ...and the snake_case keys are not leaked through.
    expect((res.spec as Record<string, unknown>)?.chart_type).toBeUndefined();
    expect((res.spec as Record<string, unknown>)?.top_n).toBeUndefined();
    // Intent detection is unchanged (risk_monitor_mode present -> risk_monitoring).
    expect(res.intent).toBe("risk_monitoring");
  });

  it("passes an already-camelCase (mock-shaped) spec through unchanged", async () => {
    const body = { ...apiBody, spec: { intent: "chart", chartType: "line", topN: 5 } };
    vi.stubGlobal("fetch", vi.fn(async () => new Response(JSON.stringify(body), { status: 200 })));
    const res = await new HttpAgentClient("http://localhost:8000").ask(request);
    expect(res.spec?.chartType).toBe("line");
    expect(res.spec?.topN).toBe(5);
  });

  it("includes the active datasetContext in the query body", async () => {
    const spy = vi.fn(async () => new Response(JSON.stringify(apiBody), { status: 200 }));
    vi.stubGlobal("fetch", spy);
    await new HttpAgentClient("http://localhost:8000").ask({ ...request, datasetContext: "pipeline" });
    const call = spy.mock.calls[0] as unknown as [string, RequestInit];
    const body = JSON.parse(call[1].body as string);
    expect(body.datasetContext).toBe("pipeline");
  });

  it("drops malformed artifacts via the type guard", async () => {
    const body = { ...apiBody, artifacts: [...apiBody.artifacts, { nope: true }] };
    vi.stubGlobal("fetch", vi.fn(async () => new Response(JSON.stringify(body), { status: 200 })));
    const res = await new HttpAgentClient("http://localhost:8000").ask(request);
    expect(res.artifacts).toHaveLength(1);
  });

  it("throws AgentError on network failure", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => {
        throw new TypeError("Failed to fetch");
      }),
    );
    await expect(new HttpAgentClient("http://localhost:8000").ask(request)).rejects.toBeInstanceOf(
      AgentError,
    );
  });

  it("throws AgentError on a non-2xx response", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => new Response("err", { status: 500 })));
    await expect(new HttpAgentClient("http://localhost:8000").ask(request)).rejects.toBeInstanceOf(
      AgentError,
    );
  });

  it("fetches the forecast snapshot from /mi/forecast/snapshot", async () => {
    const fc = {
      ok: true,
      portfolioId: "client_001/mi_2025_11",
      forecastBridge: { forecastFundedBalance: 9_966_249.7, fundedBalance: 8_902_999.7, weightedExpectedFundedAmount: 1_063_250 },
      pipelineSnapshot: { recordType: "pipeline", pipelineRowCount: 10 },
      watchlist: [],
    };
    const spy = vi.fn(async () => new Response(JSON.stringify(fc), { status: 200 }));
    vi.stubGlobal("fetch", spy);
    const res = await new HttpAgentClient("http://localhost:8000").getForecastSnapshot("client_001/mi_2025_11");
    expect(spy).toHaveBeenCalledWith(
      "http://localhost:8000/mi/forecast/snapshot?portfolioId=client_001%2Fmi_2025_11",
      expect.anything(),
    );
    expect(res.forecastBridge?.forecastFundedBalance).toBe(9_966_249.7);
  });

  it("fetches geographic exposure from /mi/geo/exposure", async () => {
    const geo = {
      dataset: "geo_itl3", portfolioId: "client_001/mi_2025_11", available: true,
      areas: [{ itl3_code: "TLK51", itl3_name: "Bristol, City of", balance: 31_000_000, count: 97, sharePct: 11.0 }],
      total: 281_400_000,
    };
    const spy = vi.fn(async () => new Response(JSON.stringify(geo), { status: 200 }));
    vi.stubGlobal("fetch", spy);
    const res = await new HttpAgentClient("http://localhost:8000").getGeoExposure("client_001/mi_2025_11");
    expect(spy).toHaveBeenCalledWith(
      "http://localhost:8000/mi/geo/exposure?portfolioId=client_001%2Fmi_2025_11",
      expect.anything(),
    );
    expect(res.areas[0].itl3_code).toBe("TLK51");
  });

  it("returns ok:false with a validation artifact for an invalid spec", async () => {
    const body = {
      ok: false,
      error: "The proposed query failed validation.",
      answer: "Chart: Bar — could not resolve a dimension.",
      spec: { chart_type: "bar" },
      validation: { ok: false, errors: ["bar chart requires a dimension (or x)"], warnings: [] },
      artifacts: [
        {
          id: "v_1",
          type: "validation",
          title: "Query Validation",
          createdAt: "2026-05-31T08:00:00Z",
          mock: false,
          source: { engine: "mi_agent.workflow", label: "MI Agent · validation" },
          summary: { blockers: 1, warnings: 0, passed: 0, coverage: 0 },
          issues: [],
        },
      ],
      warnings: [],
      assumptions: [],
    };
    vi.stubGlobal("fetch", vi.fn(async () => new Response(JSON.stringify(body), { status: 200 })));
    const res = await new HttpAgentClient("http://localhost:8000").ask(request);
    expect(res.ok).toBe(false);
    expect(res.artifacts[0].type).toBe("validation");
  });
});
