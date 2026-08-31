/**
 * Funded → Risk Limits workspace: renders the governed evaluation verbatim —
 * summary, filters, accessible statuses, detail panel with provenance and
 * drill-through, period change, and the honest unavailable / legacy / empty
 * states. No number is computed here; everything asserts against the mock
 * governed envelope.
 */

import { describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import type { AgentClient } from "@/api/AgentClient";
import {
  mockConcentrationDrillthrough,
  mockConcentrationDrivers,
  mockConcentrationHistory,
  mockConcentrationTests,
} from "@/data/mockConcentrationTests";
import { RiskLimitsWorkspace } from "./RiskLimitsWorkspace";

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
    id: "test",
    mock: true,
    ask: vi.fn(),
    getSnapshots: vi.fn(),
    getSourcePortfolios: vi.fn(),
    getPortfolioContext: vi.fn(),
    getSnapshot: vi.fn(),
    getForecastSnapshot: vi.fn(),
    getFundedEvolution: vi.fn(),
    getPipelineEvolution: vi.fn(),
    getPipelineMovement: vi.fn(async () => ({ dataset: "pipeline_movement" as const, portfolioId: "p", available: false, reason: "no extracts", stages: [] })),
    getForecastEvolution: vi.fn(),
    getFunnelEvolution: vi.fn(),
    getRiskLimits: vi.fn(),
    getConcentrationTests: vi.fn(async () => mockConcentrationTests("client_001")),
    getConcentrationDrillthrough: vi.fn(async (_p: string, testId: string) =>
      mockConcentrationDrillthrough("client_001", testId)),
    getConcentrationHistory: vi.fn(async (_p: string, testId?: string) =>
      mockConcentrationHistory("client_001", testId)),
    getConcentrationDrivers: vi.fn(async (_p: string, testId: string) =>
      mockConcentrationDrivers("client_001", testId)),
    getForecastExtrapolation: vi.fn(),
    getCohortProgression: vi.fn(),
    getMe: vi.fn(),
    getDecks: vi.fn(),
    deckDownloadUrl: vi.fn(() => null),
    getCohorts: vi.fn(),
    getGeoExposure: vi.fn(),
    ...over,
  } as AgentClient;
}

describe("RiskLimitsWorkspace", () => {
  it("renders the state-grouped summary with counts and reporting dates", async () => {
    render(<RiskLimitsWorkspace client={client()} portfolioId="client_001/mi_2025_11" />);
    await screen.findByTestId("concentration-summary");
    expect(screen.getByTestId("concentration-breaches")).toHaveTextContent("1");
    expect(screen.getByTestId("concentration-breaches")).toHaveTextContent("contractual");
    expect(screen.getByTestId("concentration-expected-breaches")).toHaveTextContent("2");
    expect(screen.getByTestId("concentration-expected-breaches")).toHaveTextContent(
      "prediction",
    );
    expect(screen.getByTestId("concentration-stress-breaches")).toHaveTextContent(
      "stress — max exposure",
    );
    expect(screen.getByTestId("concentration-deteriorations")).toHaveTextContent("2");
    expect(screen.getByText(/Reporting date 2025-11-30/)).toBeInTheDocument();
    expect(screen.getByText(/prior 2025-10-31/)).toBeInTheDocument();
  });

  it("discloses the approved configuration source (version + operator)", async () => {
    render(<RiskLimitsWorkspace client={client()} portfolioId="p" />);
    const banner = await screen.findByTestId("concentration-source-banner");
    expect(banner).toHaveTextContent("Approved configuration v2");
    expect(banner).toHaveTextContent("Demo Operator");
  });

  it("conveys status by label and glyph, not colour alone", async () => {
    render(<RiskLimitsWorkspace client={client()} portfolioId="p" />);
    const table = await screen.findByTestId("concentration-table");
    // The breached row carries a text label and its row is reachable by name.
    const row = within(table).getByRole("button", {
      name: /High-value property exposure.*Breach/,
    });
    expect(row).toHaveTextContent("Breach");
    expect(row).toHaveTextContent("✕");
    // Unavailable is never presented as passing or zero.
    const hpi = within(table).getByRole("button", { name: /House-price-index/ });
    expect(hpi).toHaveTextContent("Unavailable");
    expect(hpi).toHaveTextContent("—");
  });

  it("shows all three states with funded-to-expected movement per row", async () => {
    render(<RiskLimitsWorkspace client={client()} portfolioId="p" />);
    const table = await screen.findByTestId("concentration-table");
    const row = within(table).getByRole("button", {
      name: /London \+ South East/,
    });
    expect(row).toHaveTextContent("47.90%"); // Funded (contractual)
    expect(row).toHaveTextContent("49.20%"); // Expected Forecast
    expect(row).toHaveTextContent("hd 0.8pp"); // expected headroom
    expect(row).toHaveTextContent("52.10%"); // Full Pipeline (stress)
    expect(row).toHaveTextContent("+1.30pp"); // Move F→E
    expect(row).toHaveTextContent("LOW HEADROOM"); // service risk classification
  });

  it("renders the service-ranked emerging risks verbatim", async () => {
    render(<RiskLimitsWorkspace client={client()} portfolioId="p" />);
    const risks = await screen.findByTestId("emerging-risks");
    const items = within(risks).getAllByRole("listitem");
    expect(items[0]).toHaveTextContent("BREACH");
    expect(items[1]).toHaveTextContent("EXPECTED BREACH");
    expect(items[1]).toHaveTextContent("expected to breach");
    expect(risks).toHaveTextContent(
      "maximum-exposure scenario, not an expected outcome",
    );
  });

  it("distinguishes expected breach and stress-only breach without colour", async () => {
    render(<RiskLimitsWorkspace client={client()} portfolioId="p" />);
    const table = await screen.findByTestId("concentration-table");
    const ea = within(table).getByRole("button", { name: /East Anglia.*breach expected/ });
    expect(ea).toHaveTextContent("EXPECTED BREACH");
    // London + SE breaches only under stress; its accessible name says so.
    within(table).getByRole("button", { name: /London.*stress-only breach/ });
  });

  it("quick-filters to expected breaches only", async () => {
    render(<RiskLimitsWorkspace client={client()} portfolioId="p" />);
    const table = await screen.findByTestId("concentration-table");
    fireEvent.click(screen.getByRole("button", { name: "Expected breaches only" }));
    const rows = within(table).getAllByRole("button");
    expect(rows).toHaveLength(2); // East Anglia + high-value property
    for (const row of rows) expect(row).not.toHaveTextContent("Net WAC");
  });

  it("discloses the forecast methodology from the governed payload", async () => {
    render(<RiskLimitsWorkspace client={client()} portfolioId="p" />);
    const banner = await screen.findByTestId("forecast-provenance-banner");
    expect(banner).toHaveTextContent("2025-10-27 → 2025-11-24");
    expect(banner).toHaveTextContent("sufficiency floor"); // KFI fallback caveat
    fireEvent.click(within(banner).getByRole("button", { name: "View methodology" }));
    const block = await screen.findByTestId("methodology-block");
    expect(block).toHaveTextContent("No machine learning");
    expect(block).toHaveTextContent("maximum-exposure stress, not a prediction");
    expect(block).toHaveTextContent("≥ 12 observed cases");
  });

  it("drills through to pipeline drivers that reconcile", async () => {
    const c = client();
    render(<RiskLimitsWorkspace client={c} portfolioId="p" />);
    const table = await screen.findByTestId("concentration-table");
    fireEvent.click(within(table).getByRole("button", { name: /East Anglia/ }));
    await screen.findByTestId("state-comparison");
    expect(screen.getByTestId("breach-horizon")).toHaveTextContent("2026-02");
    fireEvent.click(screen.getByRole("button", { name: "Show pipeline drivers" }));
    await waitFor(() =>
      expect(c.getConcentrationDrivers).toHaveBeenCalledWith(
        "p", "ct_east_anglia", undefined),
    );
    expect(await screen.findByText("P-1042")).toBeInTheDocument();
    expect(screen.getByText("tips breach")).toBeInTheDocument();
    expect(screen.getByTestId("pipeline-drivers")).toHaveTextContent(
      "reconcile to the Expected Forecast numerator",
    );
  });

  it("shows the honest forecast-unavailable state (funded unaffected)", async () => {
    const snapshot = mockConcentrationTests("p");
    render(
      <RiskLimitsWorkspace
        client={client({
          getConcentrationTests: vi.fn(async () => ({
            ...snapshot,
            states: { available: false, reason: "No governed pipeline source found." },
            emergingRisks: [],
          })),
        })}
        portfolioId="p"
      />,
    );
    const note = await screen.findByTestId("forecast-unavailable-banner");
    expect(note).toHaveTextContent("Funded results are unaffected");
    // Funded column still renders; expected tile shows a dash, never zero.
    expect(screen.getByTestId("concentration-expected-breaches")).toHaveTextContent("—");
  });

  it("filters by status and searches by name", async () => {
    render(<RiskLimitsWorkspace client={client()} portfolioId="p" />);
    const table = await screen.findByTestId("concentration-table");
    fireEvent.change(screen.getByLabelText("Filter by status"), {
      target: { value: "breach" },
    });
    expect(within(table).getAllByRole("button")).toHaveLength(1);
    fireEvent.change(screen.getByLabelText("Filter by status"), {
      target: { value: "all" },
    });
    fireEvent.change(screen.getByLabelText("Search tests"), {
      target: { value: "east anglia" },
    });
    expect(within(table).getAllByRole("button")).toHaveLength(1);
    fireEvent.change(screen.getByLabelText("Search tests"), {
      target: { value: "zzz" },
    });
    expect(screen.getByTestId("concentration-no-match")).toBeInTheDocument();
  });

  it("opens the detail panel with definition, provenance and approval", async () => {
    render(<RiskLimitsWorkspace client={client()} portfolioId="p" />);
    const table = await screen.findByTestId("concentration-table");
    fireEvent.click(
      within(table).getByRole("button", { name: /East Anglia/ }),
    );
    const detail = await screen.findByTestId("concentration-detail");
    expect(detail).toHaveTextContent("must not exceed 25% of the Portfolio");
    expect(detail).toHaveTextContent("clause 1.1");
    expect(detail).toHaveTextContent("Demo Operator on 2026-01-10");
    expect(detail).toHaveTextContent("version 2");
    expect(detail).toHaveTextContent("18.40%");
  });

  it("drills through to the contributing loans, which reconcile", async () => {
    const c = client();
    render(<RiskLimitsWorkspace client={c} portfolioId="p" />);
    const table = await screen.findByTestId("concentration-table");
    fireEvent.click(within(table).getByRole("button", { name: /East Anglia/ }));
    await screen.findByTestId("concentration-detail");
    fireEvent.click(screen.getByRole("button", { name: "Show contributing loans" }));
    await waitFor(() =>
      expect(c.getConcentrationDrillthrough).toHaveBeenCalledWith(
        "p", "ct_east_anglia", undefined),
    );
    expect(await screen.findByText(/231 loan\(s\) in the numerator/)).toBeInTheDocument();
  });

  it("toggles to the prior period without recalculating anything", async () => {
    render(<RiskLimitsWorkspace client={client()} portfolioId="p" />);
    const table = await screen.findByTestId("concentration-table");
    fireEvent.click(screen.getByRole("button", { name: "Show prior period" }));
    const row = within(table).getByRole("button", { name: /East Anglia/ });
    expect(row).toHaveTextContent("17.10%"); // priorValue verbatim
  });

  it("marks the legacy extracted source as not operator-approved", async () => {
    const snapshot = mockConcentrationTests("p");
    render(
      <RiskLimitsWorkspace
        client={client({
          getConcentrationTests: vi.fn(async () => ({
            ...snapshot,
            source: "legacy_extracted" as const,
            approvalStatus: "unapproved_legacy" as const,
          })),
        })}
        portfolioId="p"
      />,
    );
    const banner = await screen.findByTestId("concentration-source-banner");
    expect(banner).toHaveTextContent("NOT operator-approved");
  });

  it("renders the honest empty state when nothing is active", async () => {
    const snapshot = mockConcentrationTests("p");
    render(
      <RiskLimitsWorkspace
        client={client({
          getConcentrationTests: vi.fn(async () => ({
            ...snapshot,
            available: false,
            tests: [],
            openProposals: 3,
            source: "none" as const,
          })),
        })}
        portfolioId="p"
      />,
    );
    const empty = await screen.findByTestId("concentration-empty");
    expect(empty).toHaveTextContent("No active concentration tests yet");
    expect(empty).toHaveTextContent("3 proposal(s) currently await review");
  });

  it("shows a controlled error state when the service is unreachable", async () => {
    render(
      <RiskLimitsWorkspace
        client={client({
          getConcentrationTests: vi.fn(async () => {
            throw new Error("down");
          }),
        })}
        portfolioId="p"
      />,
    );
    expect(await screen.findByTestId("concentration-error")).toHaveTextContent(
      "nothing is estimated",
    );
  });
});
