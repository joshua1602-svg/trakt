import { describe, expect, it, vi } from "vitest";
import { render, screen, fireEvent, within } from "@testing-library/react";
import type { AgentClient } from "@/api";
import { EvolutionPanel, STAGE_ORDER, normaliseStage, orderStages, pipelineXValue } from "./EvolutionPanel";
import { mockFundedEvolution, mockPipelineEvolution, mockForecastEvolution } from "@/data/mockEvolution";
import { mockFunnelEvolution } from "@/data/mockFunnel";
import { mockCohorts } from "@/data/mockCohorts";
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
    id: "test", mock: true, ask: vi.fn(), getSnapshots: vi.fn(), getSourcePortfolios: vi.fn(), getSnapshot: vi.fn(),
    getForecastSnapshot: vi.fn(),
    getFundedEvolution: vi.fn(async () => mockFundedEvolution("client_001")),
    getPipelineEvolution: vi.fn(async () => mockPipelineEvolution("client_001")),
    getForecastEvolution: vi.fn(async () => mockForecastEvolution("client_001")),
    getFunnelEvolution: vi.fn(async () => mockFunnelEvolution("client_001")),
    getRiskLimits: vi.fn(async () => mockRiskLimits("client_001")),
    getForecastExtrapolation: vi.fn(async () => mockForecastExtrapolation("client_001")),
    getMe: vi.fn(async () => ({ authenticated: true, isOperator: true })),
    getDecks: vi.fn(async () => ({ available: false, latest: null, decks: [], client_id: "client_001" })),
    deckDownloadUrl: vi.fn(() => null),
    getCohorts: vi.fn(async () => mockCohorts("client_001")),
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
// Task 4 & 6 — origination funnel: flow-first summary + conversion basis labels
// --------------------------------------------------------------------------- //
describe("EvolutionPanel origination conversion footers", () => {
  it("shows a conversion-vs-KFI footer on Application/Offer/Completion (not KFI)", async () => {
    render(<EvolutionPanel client={client()} portfolioId="client_001" />);
    await screen.findByText("Funded balance by month");
    fireEvent.click(screen.getByRole("tab", { name: "Origination" }));
    await screen.findByTestId("origination-funnel");
    expect(screen.getByTestId("funnel-conversion-APPLICATION")).toBeInTheDocument();
    expect(screen.getByTestId("funnel-conversion-OFFER")).toBeInTheDocument();
    expect(screen.getByTestId("funnel-conversion-COMPLETED")).toBeInTheDocument();
    // KFI is the denominator — no conversion footer on it.
    expect(screen.queryByTestId("funnel-conversion-KFI")).toBeNull();
    expect(screen.getAllByText(/Conversion vs KFI/).length).toBeGreaterThan(0);
  });

  it("keeps conversion stats collapsed by default and reveals them on expand", async () => {
    render(<EvolutionPanel client={client()} portfolioId="client_001" />);
    await screen.findByText("Funded balance by month");
    fireEvent.click(screen.getByRole("tab", { name: "Origination" }));
    const disclosure = await screen.findByTestId("funnel-conversion-APPLICATION");
    // Collapsed by default: the label shows but the detailed stats do not.
    expect(disclosure.textContent).toMatch(/Conversion vs KFI/);
    expect(screen.queryByTestId("funnel-conversion-body-APPLICATION")).toBeNull();
    // Expanding reveals the existing stats, labelled explicitly.
    fireEvent.click(within(disclosure).getByRole("button", { name: /Conversion vs KFI/ }));
    const body = screen.getByTestId("funnel-conversion-body-APPLICATION");
    expect(body.textContent).toMatch(/Weekly rate/);
    expect(body.textContent).toMatch(/by count/);
    expect(body.textContent).toMatch(/by value/);
    // Transparent about the lagged KFI denominator, not a same-period share.
    expect(body.textContent).toMatch(/Avg weekly flow/);
    expect(body.textContent).toMatch(/KFI stock/);
  });

  it("shows weekly-flow summary, a 'Show stock line' toggle, and no stock-level text row", async () => {
    render(<EvolutionPanel client={client()} portfolioId="client_001" />);
    await screen.findByText("Funded balance by month");
    fireEvent.click(screen.getByRole("tab", { name: "Origination" }));
    const card = await screen.findByTestId("funnel-stage-KFI");
    expect(card.textContent).toMatch(/weekly flow/i);
    expect(card.textContent).toMatch(/5-wk avg flow/i);
    // The explicit "Current stock level:" text row is removed from the card.
    expect(card.textContent).not.toMatch(/Current stock level/i);
    expect(screen.queryByTestId("funnel-stock-KFI")).toBeNull();
    // The stock-line overlay toggle is opt-in and renamed to "Show stock line".
    expect(screen.getByLabelText("Show stock line")).toBeInTheDocument();
    expect(screen.queryByLabelText("Show cumulative stock line")).toBeNull();
  });

  it("enlarges a single chart in a focus modal that closes cleanly", async () => {
    render(<EvolutionPanel client={client()} portfolioId="client_001" />);
    await screen.findByText("Funded balance by month");
    fireEvent.click(screen.getByRole("tab", { name: "Origination" }));
    await screen.findByTestId("origination-funnel");
    // No modal until a chart is enlarged.
    expect(screen.queryByTestId("funnel-focus-modal")).toBeNull();
    fireEvent.click(screen.getByTestId("funnel-expand-KFI"));
    const modal = await screen.findByTestId("funnel-focus-modal");
    expect(modal.textContent).toMatch(/weekly origination flow/i);
    // Easy to close.
    fireEvent.click(screen.getByTestId("funnel-focus-close"));
    expect(screen.queryByTestId("funnel-focus-modal")).toBeNull();
  });

  it("no longer shows the long explanatory paragraph by default", async () => {
    render(<EvolutionPanel client={client()} portfolioId="client_001" />);
    await screen.findByText("Funded balance by month");
    fireEvent.click(screen.getByRole("tab", { name: "Origination" }));
    await screen.findByTestId("origination-funnel");
    expect(screen.queryByText(/the new\s+origination each week/i)).toBeNull();
    // A short methodology disclosure is available instead (collapsed).
    expect(screen.getByTestId("origination-methodology-toggle")).toBeInTheDocument();
    expect(screen.queryByTestId("origination-methodology-body")).toBeNull();
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

// --------------------------------------------------------------------------- //
// C — by-stage chart: amount / count pivots, KFI-optional, conversion excludes KFI
// --------------------------------------------------------------------------- //
import { pivotStage, funnelLineStages, stageConversionSeries } from "./EvolutionPanel";
import type { StagePoint } from "@/domain";

const STAGE_ROWS: StagePoint[] = [
  { period: "2025-10-01", week: "2025-10-01", stage: "KFI", value: 1000, count: 20 },
  { period: "2025-10-01", week: "2025-10-01", stage: "APPLICATION", value: 500, count: 10 },
  { period: "2025-10-01", week: "2025-10-01", stage: "OFFER", value: 200, count: 6 },
  { period: "2025-10-01", week: "2025-10-01", stage: "COMPLETED", value: 90, count: 4 },
  { period: "2025-10-01", week: "2025-10-01", stage: "WITHDRAWN", value: 30, count: 2 },
  { period: "2025-11-01", week: "2025-11-01", stage: "KFI", value: 1200, count: 24 },
  { period: "2025-11-01", week: "2025-11-01", stage: "APPLICATION", value: 600, count: 12 },
  { period: "2025-11-01", week: "2025-11-01", stage: "OFFER", value: 300, count: 6 },
];

describe("by-stage pivots (C)", () => {
  it("pivots amount and count over day-level extract dates, stage-ordered", () => {
    const amt = pivotStage(STAGE_ROWS, "value");
    expect(amt.data.map((r) => r.period)).toEqual(["2025-10-01", "2025-11-01"]);
    expect(amt.stages).toEqual(["KFI", "APPLICATION", "OFFER", "COMPLETED", "WITHDRAWN"]);
    expect(amt.data[0]).toMatchObject({ KFI: 1000, APPLICATION: 500, COMPLETED: 90 });
    const cnt = pivotStage(STAGE_ROWS, "count");
    expect(cnt.data[0]).toMatchObject({ KFI: 20, APPLICATION: 10, OFFER: 6 });
  });

  it("funnel lines exclude Withdrawn/Unknown and can drop KFI", () => {
    const stages = ["KFI", "APPLICATION", "OFFER", "COMPLETED", "WITHDRAWN"];
    expect(funnelLineStages(stages, true)).toEqual(["KFI", "APPLICATION", "OFFER", "COMPLETED"]);
    expect(funnelLineStages(stages, false)).toEqual(["APPLICATION", "OFFER", "COMPLETED"]);
  });

  it("conversion excludes KFI (no KFI/KFI=100%): Application/Offer/Completion vs KFI", () => {
    const conv = stageConversionSeries(STAGE_ROWS);
    expect(conv.stages).toEqual(["APPLICATION", "OFFER", "COMPLETED"]);
    expect(conv.stages).not.toContain("KFI");
    // count ratios vs KFI in the same week: App 10/20=50%, Offer 6/20=30%, Completed 4/20=20%.
    expect(conv.data[0]).toMatchObject({ APPLICATION: 50, OFFER: 30, COMPLETED: 20 });
    expect((conv.data[0] as Record<string, unknown>).KFI).toBeUndefined();
  });

  it("conversion is divide-by-zero safe when a week has no KFI", () => {
    const rows: StagePoint[] = [
      { period: "w", stage: "APPLICATION", value: 5, count: 5 },
      { period: "w", stage: "OFFER", value: 2, count: 2 },
    ];
    expect(stageConversionSeries(rows).data[0]).toMatchObject({ APPLICATION: 0, OFFER: 0 });
  });
});

describe("EvolutionPanel stage mode toggle (C)", () => {
  it("switches the by-stage chart between Amount, Count and Conversion", async () => {
    render(<EvolutionPanel client={client()} portfolioId="client_001" />);
    await screen.findByText("Funded balance by month");
    fireEvent.click(screen.getByRole("tab", { name: "Pipeline" }));
    await screen.findByText("Pipeline by stage over time");
    fireEvent.click(screen.getByRole("tab", { name: "Count" }));
    expect(screen.getByTestId("stage-mode-note").textContent).toMatch(/Case count/);
    fireEvent.click(screen.getByRole("tab", { name: "Conversion" }));
    expect(screen.getByTestId("stage-mode-note").textContent).toMatch(/% of KFIs/);
  });
});
