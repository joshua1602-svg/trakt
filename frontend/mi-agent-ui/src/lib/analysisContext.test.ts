import { describe, expect, it } from "vitest";
import {
  type AnalysisContext,
  contextSummary,
  deriveContext,
  looksLikeFollowUp,
  resolveFollowUp,
} from "./analysisContext";

function ctx(overrides: Partial<AnalysisContext> = {}): AnalysisContext {
  return {
    lastQuestion: "Show balance by region",
    lastSuccessfulSpec: { metric: "current_outstanding_balance", dimension: "geographic_region_obligor" },
    activeMeasure: "current_outstanding_balance",
    activeDimensions: ["geographic_region_obligor"],
    activeFilters: {},
    ...overrides,
  };
}

describe("deriveContext", () => {
  it("extracts measure / dimensions / filters from a successful spec", () => {
    const c = deriveContext({
      spec: {
        metric: "current_loan_to_value",
        dimensions: ["broker_channel"],
        filters: { geographic_region_obligor: "South East" },
      },
      question: "average ltv by broker",
      portfolioId: "client_001/mi_2025_11",
    });
    expect(c.activeMeasure).toBe("current_loan_to_value");
    expect(c.activeDimensions).toEqual(["broker_channel"]);
    expect(c.activeFilters).toEqual({ geographic_region_obligor: "South East" });
    expect(c.portfolioId).toBe("client_001/mi_2025_11");
  });
});

describe("contextSummary", () => {
  it("is null without context and a clean label with context", () => {
    expect(contextSummary(null)).toBeNull();
    expect(contextSummary(ctx({ activeFilters: { geographic_region_obligor: "South East" } }))).toBe(
      "Balance · Region · South East",
    );
  });
});

describe("looksLikeFollowUp", () => {
  it("is false without context", () => {
    expect(looksLikeFollowUp("split by broker", null)).toBe(false);
  });
  it("detects connective follow-ups", () => {
    const c = ctx();
    expect(looksLikeFollowUp("split by broker", c)).toBe(true);
    expect(looksLikeFollowUp("only South East", c)).toBe(true);
    expect(looksLikeFollowUp("now average LTV", c)).toBe(true);
    expect(looksLikeFollowUp("show as table", c)).toBe(true);
    expect(looksLikeFollowUp("drill into Broker A", c)).toBe(true);
  });
  it("ignores complete standalone questions", () => {
    expect(looksLikeFollowUp("show balance by region", ctx())).toBe(false);
  });
});

describe("resolveFollowUp", () => {
  it("changes the dimension (split by broker)", () => {
    const r = resolveFollowUp("split by broker", ctx());
    expect(r).not.toBeNull();
    expect(r!.question).toBe("Balance by Broker");
    expect(r!.filters).toEqual({});
  });

  it("adds a filter on the active dimension (only South East)", () => {
    const r = resolveFollowUp("only South East", ctx());
    expect(r).not.toBeNull();
    expect(r!.question).toBe("Balance by Region");
    expect(r!.filters).toEqual({ geographic_region_obligor: "South East" });
  });

  it("drills into a value (drill into Broker A) on the active dimension", () => {
    const r = resolveFollowUp("drill into Broker A", ctx({ activeDimensions: ["broker_channel"] }));
    expect(r!.filters).toEqual({ broker_channel: "Broker A" });
  });

  it("changes the measure preserving the dimension (now average ltv)", () => {
    const r = resolveFollowUp("now average ltv", ctx());
    expect(r).not.toBeNull();
    expect(r!.question.toLowerCase()).toContain("average ltv");
    expect(r!.question.toLowerCase()).toContain("by region");
  });

  it("switches output mode (show as table)", () => {
    const r = resolveFollowUp("show as table", ctx());
    expect(r!.question.toLowerCase()).toContain("as a table");
  });

  it("returns null for unsupported follow-ups (exclude / compare)", () => {
    expect(resolveFollowUp("exclude London", ctx())).toBeNull();
    expect(resolveFollowUp("compare to last week", ctx())).toBeNull();
  });
});
