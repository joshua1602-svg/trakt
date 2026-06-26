import { describe, expect, it } from "vitest";
import type { ChartArtifact } from "@/domain";
import { isDebugNarrative, presentAnswer } from "./responsePresenter";

// The exact parser/debug string the live backend used to return as the answer.
const DEBUG = "Chart: Bar · Metric: Balance · Dimension: Region · Aggregation: Sum · Parser: deterministic · Validation: Passed — 6 group(s).";

function regionChart(): ChartArtifact {
  return {
    id: "art",
    type: "chart",
    title: "Balance by Region",
    source: { engine: "mi_agent.workflow", label: "MI Agent · bar", spec: { metric: "current_outstanding_balance", dimension: "geographic_region_obligor" } },
    createdAt: "2026-06-26T08:00:00Z",
    mock: false,
    chartType: "bar",
    xKey: "geographic_region_obligor",
    series: [{ key: "current_outstanding_balance", label: "Balance", color: "#000" }],
    valueFormat: "gbp",
    displayHints: { current_outstanding_balance: { format: "gbp", scale: null } },
    rows: [
      { geographic_region_obligor: "London", current_outstanding_balance: 700 },
      { geographic_region_obligor: "South East", current_outstanding_balance: 200 },
      { geographic_region_obligor: "South West", current_outstanding_balance: 100 },
    ],
  };
}

const spec = { metric: "current_outstanding_balance", dimension: "geographic_region_obligor" };

describe("isDebugNarrative", () => {
  it("flags parser/validation status strings", () => {
    expect(isDebugNarrative(DEBUG)).toBe(true);
    expect(isDebugNarrative("Aggregation: Sum")).toBe(true);
  });
  it("does not flag human prose", () => {
    expect(isDebugNarrative("London holds 41% of the funded book.")).toBe(false);
  });
});

describe("presentAnswer", () => {
  it("discards a parser/debug narrative and answers in plain English", () => {
    const out = presentAnswer({ question: "balance by region", ok: true, spec, artifacts: [regionChart()], narrative: DEBUG });
    expect(out).not.toContain("Parser");
    expect(out).not.toContain("Validation");
    expect(out).not.toContain("Aggregation");
    // grounded, with value + share + an intro to the chart
    expect(out).toContain("London");
    expect(out).toContain("70%");
    expect(out.toLowerCase()).toContain("balance");
    expect(out.toLowerCase()).toContain("below");
  });

  it("keeps a genuinely analytical narrative untouched (e.g. the rich mock prose)", () => {
    const prose =
      "The book is concentrated in London and the South East, which together hold 41.2% of balance; no single region breaches the 25% soft limit.";
    expect(presentAnswer({ question: "q", ok: true, spec, artifacts: [regionChart()], narrative: prose })).toBe(prose);
  });

  it("replaces short generic filler with the grounded sentence", () => {
    const filler = "Here is the bar for your query, covering 3 group(s).";
    const out = presentAnswer({ question: "q", ok: true, spec, artifacts: [regionChart()], narrative: filler });
    expect(out).not.toBe(filler);
    expect(out).toContain("London");
    expect(out).toContain("70%");
  });

  it("answers from the artifact when there is no narrative", () => {
    const out = presentAnswer({ question: "q", ok: true, spec, artifacts: [regionChart()] });
    expect(out).toContain("London");
    expect(isDebugNarrative(out)).toBe(false);
  });

  it("gives a clean error sentence on failure", () => {
    const out = presentAnswer({ question: "q", ok: false, artifacts: [] });
    expect(out.toLowerCase()).toContain("couldn't complete");
    expect(isDebugNarrative(out)).toBe(false);
  });
});
