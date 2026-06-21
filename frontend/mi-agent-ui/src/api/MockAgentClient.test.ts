import { describe, expect, it } from "vitest";
import type { AgentRequest } from "@/domain";
import { AgentError, MockAgentClient } from "./index";

const request: AgentRequest = {
  question: "Explain top concentration risks",
  portfolio: { id: "erm-uk-master", name: "ERM UK — Master", entity: "Trakt SPV I" },
  reporting: { asOf: "2026-05-31" },
};

describe("MockAgentClient", () => {
  it("resolves a structured response", async () => {
    const client = new MockAgentClient({ latencyMs: 0 });
    const res = await client.ask(request);
    expect(res.ok).toBe(true);
    expect(res.intent).toBe("concentration_risk");
    expect(client.mock).toBe(true);
  });

  it("rejects when aborted before resolving", async () => {
    const client = new MockAgentClient({ latencyMs: 50 });
    const controller = new AbortController();
    const promise = client.ask(request, controller.signal);
    controller.abort();
    await expect(promise).rejects.toBeInstanceOf(AgentError);
  });

  it("surfaces simulated transient failures", async () => {
    const client = new MockAgentClient({ latencyMs: 0, failureRate: 1 });
    await expect(client.ask(request)).rejects.toBeInstanceOf(AgentError);
  });
});
