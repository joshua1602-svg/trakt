import { describe, expect, it } from "vitest";
import type { AgentRequest } from "@/domain";
import { isArtifact } from "@/domain";
import { buildAgentResponse, classifyIntent } from "./mockResponses";

const baseRequest = (question: string): AgentRequest => ({
  question,
  portfolio: { id: "erm-uk-master", name: "ERM UK — Master", entity: "Trakt SPV I" },
  reporting: { asOf: "2026-05-31", comparedTo: "2026-04-30" },
});

describe("classifyIntent", () => {
  it.each([
    ["Show portfolio movement since last period", "portfolio_overview"],
    ["Explain top concentration risks", "concentration_risk"],
    ["Generate pipeline bridge to £100MM securitisation size", "pipeline"],
    ["Show static pool performance by vintage", "static_pools"],
    ["Check risk-grade migration and limit breaches", "risk_monitoring"],
    ["Project the book under a base-case scenario", "scenario"],
    ["Summarise validation issues blocking reporting", "validation"],
  ] as const)("routes %s -> %s", (question, intent) => {
    expect(classifyIntent(question)).toBe(intent);
  });

  it("falls back to unknown for unrelated questions", () => {
    expect(classifyIntent("what is the weather today")).toBe("unknown");
  });
});

describe("buildAgentResponse", () => {
  it("returns a well-formed response envelope", () => {
    const res = buildAgentResponse(baseRequest("Explain top concentration risks"));
    expect(res.ok).toBe(true);
    expect(res.intent).toBe("concentration_risk");
    expect(res.narrative).toBeTruthy();
    expect(res.assumptions.length).toBeGreaterThan(0);
    expect(res.interpreted).toContain("Concentration");
    expect(res.artifacts.length).toBeGreaterThan(0);
  });

  it("produces only valid artifacts carrying lineage", () => {
    const res = buildAgentResponse(baseRequest("risk-grade migration and breaches"));
    for (const a of res.artifacts) {
      expect(isArtifact(a)).toBe(true);
      expect(a.source.label).toBeTruthy();
      expect(a.mock).toBe(true);
      expect(a.source.asOf).toBe("2026-05-31");
      expect(a.source.portfolio).toBe("erm-uk-master");
    }
  });

  it("surfaces artifact warnings on the response", () => {
    const res = buildAgentResponse(baseRequest("check limit breaches"));
    expect(res.warnings.length).toBeGreaterThan(0);
  });

  it("returns default dashboard artifacts for unknown intents", () => {
    const res = buildAgentResponse(baseRequest("hello there"));
    expect(res.intent).toBe("unknown");
    expect(res.artifacts.length).toBeGreaterThan(0);
  });

  it("covers every artifact type across the suggested intents", () => {
    const questions = [
      "portfolio movement",
      "concentration by region",
      "pipeline bridge",
      "static pool vintage",
      "risk migration breaches",
      "base-case scenario projection",
      "validation issues",
    ];
    const types = new Set(
      questions.flatMap((q) => buildAgentResponse(baseRequest(q)).artifacts.map((a) => a.type)),
    );
    for (const t of ["kpi", "chart", "table", "validation", "risk", "scenario"]) {
      expect(types.has(t as never)).toBe(true);
    }
  });
});
