import { describe, expect, it, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import type { AgentClient } from "@/api";
import { EvolutionPanel, STAGE_ORDER, normaliseStage, orderStages, stageConversion, pipelineXValue } from "./EvolutionPanel";
import { mockFundedEvolution, mockPipelineEvolution, mockForecastEvolution } from "@/data/mockEvolution";
import { mockFunnelEvolution } from "@/data/mockFunnel";
import { mockRiskLimits } from "@/data/mockRiskLimits";
import { mockForecastExtrapolation } from "@/data/mockForecastExtrapolation";

vi.mock("recharts", async () => {
  const actual = await vi.importActual<typeof import("recharts")>("recharts");
  return {
    ...actual,
    ResponsiveContainer: ({ children }: { children: React.ReactNode }) => (
      <div style={{ width: 600, height: 200 }}>{children}</div>
    ),
  };
});

function client(over: Partial<AgentClient> = {}): AgentClient {
  return {
    id: "test", mock: true, ask: vi.fn(), getSnapshots: vi.fn(), getSnapshot: vi.fn(),
    getForecastSnapshot: vi.fn(),
    getFundedEvolution: vi.fn(async () => mockFundedEvolution("client_001")),
    getPipelineEvolution: vi.fn(async () => mockPipelineEvolution("client_001")),
    getForecastEvolution: vi.fn(async () => mockForecastEvolution("client_001")),
    getFunnelEvolution: vi.fn(async () => mockFunnelEvolution("client_001")),
    getRiskLimits: vi.fn(async () => mockRiskLimits("client_001")),
    getForecastExtrapolation: vi.fn(async () => mockForecastExtrapolation("client_001")),
    ...over,
  };
}

// --------------------------------------------------------------------------- //
// A5 — stage process ordering (synonyms + case)
// --------------------------------------------------------------------------- //
describe("stage ordering", () => {
  it("orders by the funnel process order with Withdrawn after, Unknown last", () => {
    expect(orderStages(["OFFER", "COMPLETED", "KFI", "APPLICATION"]))
      .toEqual(["KFI", "APPLICATION", "OFFER", "COMPLETED"]);
    expect(orderStages(["UNKNOWN", "WITHDRAWN", "KFI", "OFFER"]))
      .toEqual(["KFI", "OFFER", "WITHDRAWN", "UNKNOWN"]);
  });

  it("normalises COMPLETION/COMPLETE and APP synonyms and mixed case", () => {
    expect(normaliseStage("completion")).toBe("COMPLETED");
    expect(normaliseStage("Complete")).toBe("COMPLETED");
    expect(normaliseStage("app")).toBe("APPLICATION");
    expect(orderStages(["completion", "kfi", "app", "Offer"]))
      .toEqual(["kfi", "app", "Offer", "completion"]);
  });

  it("STAGE_ORDER is the canonical funnel order", () => {
    expect(STAGE_ORDER.slice(0, 4)).toEqual(["KFI", "APPLICATION", "OFFER", "COMPLETED"]);
  });
});

// --------------------------------------------------------------------------- //
// A4 — conversion rates (divide-by-zero safe)
// --------------------------------------------------------------------------- //
describe("stage conversion vs KFI", () => {
  const kfi = { label: "KFIs", latestValue: 1000, latestCount: 20 } as any;
  it("computes count and value conversion", () => {
    const c = stageConversion({ latestValue: 500, latestCount: 10 } as any, kfi)!;
    expect(c.countPct).toBe(50);
    expect(c.valuePct).toBe(50);
    expect(c.numerCount).toBe(10);
    expect(c.denomCount).toBe(20);
  });
  it("is divide-by-zero safe when KFI is zero", () => {
    const c = stageConversion({ latestValue: 100, latestCount: 5 } as any,
      { latestValue: 0, latestCount: 0 } as any)!;
    expect(c.countPct).toBeNull();
    expect(c.valuePct).toBeNull();
  });
});

describe("EvolutionPanel origination conversion footers", () => {
  it("shows a conversion-vs-KFI footer on Application/Offer/Completion (not KFI)", async () => {
    render(<EvolutionPanel client={client()} portfolioId="client_001" />);
    await screen.findByText("Funded balance by month");
    fireEvent.click(screen.getByRole("tab", { name: "origination" }));
    await screen.findByTestId("origination-funnel");
    expect(screen.getByTestId("funnel-conversion-APPLICATION")).toBeInTheDocument();
    expect(screen.getByTestId("funnel-conversion-OFFER")).toBeInTheDocument();
    expect(screen.getByTestId("funnel-conversion-COMPLETED")).toBeInTheDocument();
    // KFI is the denominator — no conversion footer on it.
    expect(screen.queryByTestId("funnel-conversion-KFI")).toBeNull();
    expect(screen.getAllByText(/Conversion vs KFI/).length).toBeGreaterThan(0);
  });
});

// --------------------------------------------------------------------------- //
// A1 — weekly pipeline labels are day-level (not repeated month labels)
// --------------------------------------------------------------------------- //
describe("weekly pipeline labels (A1)", () => {
  it("uses the day-level extract date, not the YYYY-MM month", () => {
    // Day-level week wins over the month, so weekly points are distinguishable.
    expect(pipelineXValue({ period: "2025-10", week: "2025-10-06", extract_date: "2025-10-06" }))
      .toBe("2025-10-06");
    expect(pipelineXValue({ period: "2025-10", extract_date: "2025-10-13" }))
      .toBe("2025-10-13");
    // Falls back to the month only when no day-level date exists.
    expect(pipelineXValue({ period: "2025-10" })).toBe("2025-10");
  });
});

// --------------------------------------------------------------------------- //
// A2 — pipeline data-quality annotation (sharp week-on-week movement)
// --------------------------------------------------------------------------- //
import { pipelineDataQuality } from "./EvolutionPanel";

describe("pipeline data quality (A2)", () => {
  it("flags the worst sharp week-on-week movement above the threshold", () => {
    const dq = pipelineDataQuality([
      { period: "2025-10-06", pipeline_amount: 300 },
      { period: "2025-10-13", pipeline_amount: 120 },  // -60%
      { period: "2025-10-20", pipeline_amount: 130 },
    ]);
    expect(dq).toEqual({ period: "2025-10-13", changePct: -60 });
  });

  it("returns null when movements are within the threshold", () => {
    expect(pipelineDataQuality([
      { period: "a", pipeline_amount: 100 }, { period: "b", pipeline_amount: 110 },
    ])).toBeNull();
  });

  it("ignores missing (null) weeks rather than treating them as zero", () => {
    const dq = pipelineDataQuality([
      { period: "a", pipeline_amount: 100 },
      { period: "b", pipeline_amount: null },
      { period: "c", pipeline_amount: 105 },
    ]);
    expect(dq).toBeNull();  // no fabricated zero-drop
  });
});
