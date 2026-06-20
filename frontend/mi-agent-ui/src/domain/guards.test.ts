import { describe, expect, it } from "vitest";
import type { Artifact } from "@/domain";
import {
  isArtifact,
  isChartArtifact,
  isKPIArtifact,
  isRiskArtifact,
  isScenarioArtifact,
  isTableArtifact,
  isValidationArtifact,
} from "@/domain";

const make = (type: Artifact["type"]): Artifact =>
  ({
    id: "x",
    type,
    title: "t",
    createdAt: new Date().toISOString(),
    mock: true,
    source: { engine: "test", label: "test" },
  }) as Artifact;

describe("type guards", () => {
  it("narrow by discriminant", () => {
    expect(isKPIArtifact(make("kpi"))).toBe(true);
    expect(isChartArtifact(make("chart"))).toBe(true);
    expect(isTableArtifact(make("table"))).toBe(true);
    expect(isValidationArtifact(make("validation"))).toBe(true);
    expect(isRiskArtifact(make("risk"))).toBe(true);
    expect(isScenarioArtifact(make("scenario"))).toBe(true);
  });

  it("are mutually exclusive", () => {
    const kpi = make("kpi");
    expect(isChartArtifact(kpi)).toBe(false);
    expect(isRiskArtifact(kpi)).toBe(false);
  });

  it("isArtifact validates shape", () => {
    expect(isArtifact(make("kpi"))).toBe(true);
    expect(isArtifact(null)).toBe(false);
    expect(isArtifact({})).toBe(false);
    expect(isArtifact({ id: "x", type: "kpi" })).toBe(false);
  });
});
