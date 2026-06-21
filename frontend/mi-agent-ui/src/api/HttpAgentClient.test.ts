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
