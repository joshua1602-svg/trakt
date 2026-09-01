/**
 * Sprint 2 — the React stage-movement panel.
 *
 * The panel's job is to RENDER the governed transition payload, so the tests
 * that matter most are the ones proving it adds nothing of its own:
 *
 *   * every number on screen is the engine's, asserted against the engine's
 *     actual output rather than a hand-written stand-in;
 *   * a new arrival is never drawn as if it came from a real stage;
 *   * an amount amendment never appears as an exit plus an arrival;
 *   * an unclassified departure is never resolved into a withdrawal;
 *   * the engine's typed unavailability is shown in the engine's words,
 *     never as an empty matrix that reads as "nothing moved";
 *   * the panel is invisible, and issues no request, when the layer is off.
 *
 * The fixture is the REAL payload from
 * `movement_detail.resolve_stage_transition_detail` over the committed
 * two-snapshot pack (see scripts/generate_stage_transition_fixture.py). A
 * Python test re-runs the engine and compares, so it cannot drift.
 */

import { render, screen, waitFor, within } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { StageTransitionPanel, departureLabel } from
  "@/components/pipeline/StageTransitionPanel";
import type { AgentClient } from "@/api";
import type { StageTransitionDetail } from "@/domain";
import { UNCLASSIFIED_DEPARTURE } from "@/domain";

import engineDetail from "@/test/fixtures/stageTransitionDetail.json";

const DETAIL = engineDetail as unknown as StageTransitionDetail;

function clientReturning(detail: unknown, spy = vi.fn()): AgentClient {
  return {
    id: "test", mock: false,
    getStageTransitionDetail: spy.mockResolvedValue(detail),
  } as unknown as AgentClient;
}

function renderPanel(detail: unknown = DETAIL, enabled = true, spy = vi.fn()) {
  const client = clientReturning(detail, spy);
  render(<StageTransitionPanel client={client} portfolioId="client_001"
    enabled={enabled} />);
  return { client, spy };
}

async function panel() {
  return await screen.findByTestId("stage-transitions");
}

// --------------------------------------------------------------------------- //
// Containment — off means absent, and silent.
// --------------------------------------------------------------------------- //
describe("containment", () => {
  it("renders nothing and issues no request when the layer is off", async () => {
    const { spy } = renderPanel(DETAIL, false);
    expect(screen.queryByTestId("stage-transitions")).toBeNull();
    expect(screen.queryByTestId("stage-transitions-unavailable")).toBeNull();
    expect(spy).not.toHaveBeenCalled();
  });

  it("asks the SAME governed capability the deck reads", async () => {
    const { spy } = renderPanel();
    await panel();
    expect(spy).toHaveBeenCalledWith("client_001", undefined, undefined);
  });
});

// --------------------------------------------------------------------------- //
// The engine's numbers, unmodified.
// --------------------------------------------------------------------------- //
describe("it renders the engine's numbers", () => {
  it("shows the reporting window the engine resolved", async () => {
    renderPanel();
    const el = await panel();
    expect(within(el).getByText(/2026-06-05\s*→\s*2026-06-12/)).toBeInTheDocument();
  });

  it("shows every stage-to-stage transition the engine published", async () => {
    renderPanel();
    const el = await panel();
    // Truth from the fixture: KFI→Application 2, Application→Offer 2,
    // Offer→Completed 1.
    expect(DETAIL.transitions).toHaveLength(3);
    expect(within(el).getByText("KFI → Application")).toBeInTheDocument();
    expect(within(el).getByText("Application → Offer")).toBeInTheDocument();
    expect(within(el).getByText("Offer → Completed")).toBeInTheDocument();
  });

  it("shows the engine's KFI→Application case count, not a recomputed one", async () => {
    renderPanel();
    const el = await panel();
    const row = within(el).getByText("KFI → Application").parentElement!;
    const move = DETAIL.transitions.find(
      (t) => t.source_stage === "KFI" && t.destination_stage === "APPLICATION")!;
    expect(move.case_count).toBe(2);
    expect(row.textContent).toContain(`${move.case_count} cases`);
  });

  it("shows the engine's Application→Offer case count", async () => {
    renderPanel();
    const el = await panel();
    const row = within(el).getByText("Application → Offer").parentElement!;
    const move = DETAIL.transitions.find(
      (t) => t.source_stage === "APPLICATION" && t.destination_stage === "OFFER")!;
    expect(move.case_count).toBe(2);
    expect(row.textContent).toContain(`${move.case_count} cases`);
  });

  it("shows the terminal Offer→Completed flow as a transition, not a departure",
    async () => {
      renderPanel();
      const el = await panel();
      const row = within(el).getByText("Offer → Completed").parentElement!;
      expect(row.textContent).toContain("1 case");
    });

  it("reproduces the engine's per-stage reconciliation exactly", async () => {
    renderPanel();
    const table = await screen.findByTestId("stage-transitions-reconciliation");
    for (const r of DETAIL.reconciliation!.by_stage) {
      const label = r.stage[0] + r.stage.slice(1).toLowerCase();
      const row = within(table).getByRole("row", { name: new RegExp(`^${label}`, "i") });
      const cells = within(row).getAllByRole("cell").map((c) => c.textContent);
      expect(cells).toEqual([
        String(r.opening_case_count), String(r.new_arrivals),
        String(r.transitions_in), String(r.transitions_out),
        String(r.departures), String(r.closing_case_count),
      ]);
    }
  });

  it("discloses the engine's residuals rather than hiding them", async () => {
    renderPanel();
    const el = await panel();
    expect(DETAIL.reconciliation!.count_reconciliation_residual).toBe(0);
    expect(el.textContent).toContain("Reconciliation residual");
    expect(el.textContent).toContain("pipeline_case_identifier");
  });
});

// --------------------------------------------------------------------------- //
// Semantics the engine sprint established, which presentation must not undo.
// --------------------------------------------------------------------------- //
describe("engine semantics survive presentation", () => {
  it("never draws a new arrival as coming from a real stage", async () => {
    renderPanel();
    const el = await panel();
    // Truth from the fixture: arrivals into KFI and into Application.
    expect(DETAIL.new_arrivals.map((a) => a.destination_stage).sort())
      .toEqual(["APPLICATION", "KFI"]);
    expect(within(el).getByText("New into KFI")).toBeInTheDocument();
    expect(within(el).getByText("New into Application")).toBeInTheDocument();
    // The synthetic self-loop the engine sprint forbade.
    expect(el.textContent).not.toContain("KFI → KFI");
    expect(el.textContent).not.toContain("Application → Application");
    expect(el.textContent).not.toContain("NEW →");
  });

  it("shows an amendment on a stayer as an amount change, never as churn",
    async () => {
      renderPanel();
      const el = await panel();
      const kfi = DETAIL.stayers.find((s) => s.stage === "KFI")!;
      expect(kfi.amount_change).toBe(20_000);
      // Two stayers at KFI, ONE row — not an exit plus an arrival.
      const stayers = within(el).getByTestId("stx-stayers");
      expect(within(stayers).getByText("KFI").parentElement!.textContent)
        .toContain("2 cases");
      // No departure or arrival was invented for the amended case.
      expect(DETAIL.event_totals!.stayer.case_count).toBe(3);
    });

  it("leaves an unclassified departure unclassified", async () => {
    renderPanel();
    const el = await panel();
    const unresolved = DETAIL.departures.filter(
      (d) => d.governed_outcome === UNCLASSIFIED_DEPARTURE);
    expect(unresolved).toHaveLength(2);
    expect(within(el).getByText("Left from Offer — unclassified"))
      .toBeInTheDocument();
    expect(within(el).getByText("Left from Application — unclassified"))
      .toBeInTheDocument();
  });

  it("shows a governed terminal outcome where the engine evidenced one", async () => {
    renderPanel();
    const el = await panel();
    expect(within(el).getByText("Left after Completed")).toBeInTheDocument();
    expect(within(el).getByText("Left after Withdrawn")).toBeInTheDocument();
  });

  it("never labels an unevidenced departure a withdrawal", () => {
    expect(departureLabel({
      source_stage: "OFFER", governed_outcome: UNCLASSIFIED_DEPARTURE,
      outcome_evidence: "none", case_count: 1, prior_amount: 1,
    })).toBe("Left from Offer — unclassified");
    expect(departureLabel({
      source_stage: "OFFER", governed_outcome: UNCLASSIFIED_DEPARTURE,
      outcome_evidence: "none", case_count: 1, prior_amount: 1,
    })).not.toMatch(/withdraw/i);
  });
});

// --------------------------------------------------------------------------- //
// Count AND amount from the one payload.
// --------------------------------------------------------------------------- //
describe("the same payload answers both measures", () => {
  it("switches to the engine's amounts without refetching", async () => {
    const { spy } = renderPanel();
    const el = await panel();
    (await within(el).findByRole("tab", { name: "Value" })).click();
    const move = DETAIL.transitions.find(
      (t) => t.source_stage === "KFI" && t.destination_stage === "APPLICATION")!;
    expect(move.latest_amount).toBe(920_000);
    await waitFor(() => {
      expect(within(el).getByText("KFI → Application").parentElement!.textContent)
        .toContain("920");
    });
    // One payload, both measures — the toggle is presentation, not a new query.
    expect(spy).toHaveBeenCalledTimes(1);
  });
});

// --------------------------------------------------------------------------- //
// Typed unavailability propagates; the panel never decides it.
// --------------------------------------------------------------------------- //
describe("unavailability", () => {
  const refusal = {
    ...DETAIL, available: false,
    reason_code: "duplicate_case_identifiers",
    reason: "2 duplicate pipeline_case_identifier value(s) in the latest "
      + "snapshot prevent deterministic case matching.",
    transitions: [], new_arrivals: [], stayers: [], departures: [],
    reconciliation: null, event_totals: null, counts: null,
  };

  it("shows the engine's reason rather than an empty matrix", async () => {
    render(<StageTransitionPanel client={clientReturning(refusal)}
      portfolioId="client_001" enabled />);
    const el = await screen.findByTestId("stage-transitions-unavailable");
    expect(el.textContent).toContain("duplicate pipeline_case_identifier");
    // Critically NOT the available panel, which would read as "nothing moved".
    expect(screen.queryByTestId("stage-transitions")).toBeNull();
  });

  it("treats a rejected request as no detail, never a visible error", async () => {
    const client = {
      id: "test", mock: false,
      getStageTransitionDetail: vi.fn().mockRejectedValue(new Error("404")),
    } as unknown as AgentClient;
    render(<StageTransitionPanel client={client} portfolioId="client_001" enabled />);
    expect(await screen.findByTestId("stage-transitions-unavailable"))
      .toBeInTheDocument();
  });
});
