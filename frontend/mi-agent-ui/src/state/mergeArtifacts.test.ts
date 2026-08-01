import { describe, expect, it } from "vitest";
import type { Artifact } from "@/domain";
import { mergeArtifacts } from "./useWorkspace";

function art(id: string, title: string, over: Partial<Artifact> = {}): Artifact {
  return {
    id,
    type: "chart",
    title,
    source: { engine: "mi_agent.workflow", label: "MI Agent · bar" },
    createdAt: "2026-05-31T08:00:00Z",
    mock: false,
    chartType: "bar",
    xKey: "x",
    series: [],
    rows: [],
    ...over,
  } as Artifact;
}

describe("mergeArtifacts — workspace accumulation", () => {
  it("keeps earlier answers when a new question lands (no wholesale replace)", () => {
    const prev = [art("a1", "Balance by Region")];
    const next = mergeArtifacts(prev, [art("b1", "Balance by Broker")]);
    expect(next.map((a) => a.id)).toEqual(["b1", "a1"]); // newest first
  });

  it("supersedes an older unpinned artifact with the same title + type", () => {
    const prev = [art("a1", "Balance by Region"), art("a2", "Balance by Broker")];
    const next = mergeArtifacts(prev, [art("a3", "Balance by Region")]);
    expect(next.map((a) => a.id)).toEqual(["a3", "a2"]); // refreshed, not stacked
  });

  it("keeps pinned artifacts first and never drops them", () => {
    const prev = [art("p1", "Pinned Chart", { pinned: true }), art("a1", "Balance by Region")];
    const next = mergeArtifacts(prev, [art("b1", "Balance by Broker")]);
    expect(next[0].id).toBe("p1");
    expect(next.map((a) => a.id)).toEqual(["p1", "b1", "a1"]);
  });

  it("caps unpinned accumulation so the workspace cannot grow unbounded", () => {
    const prev = Array.from({ length: 30 }, (_, i) => art(`a${i}`, `Chart ${i}`));
    const next = mergeArtifacts(prev, [art("new", "Fresh Chart")]);
    expect(next.length).toBeLessThanOrEqual(24);
    expect(next[0].id).toBe("new"); // the fresh result always survives the cap
  });
});
