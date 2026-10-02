import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { StageMovementPanel } from "./StageMovementPanel";
import type { PipelineMovement } from "@/domain/evolution";

/**
 * The payload below is the shape `/mi/evolution/pipeline-movement` returns —
 * the SAME computation the investor pack renders. If the engine's contract
 * changes, this fixture and the deck's reader change together.
 */
const MOVEMENT: PipelineMovement = {
  dataset: "pipeline_movement",
  portfolioId: "client/run",
  available: true,
  openingWeek: "2026-06-12",
  closingWeek: "2026-06-26",
  identifierField: "pipeline_case_identifier",
  openingCaseCount: 20,
  closingCaseCount: 19,
  persistingCaseCount: 17,
  reconciles: true,
  stages: [
    {
      stage: "KFI",
      openingCaseCount: 10, openingAmount: 2_000_000,
      arrivalCaseCount: 2, arrivalAmount: 400_000,
      departureCaseCount: 3, departureAmount: 600_000,
      persistingCaseCount: 7, amountChangeOnPersisting: 50_000,
      closingCaseCount: 9, closingAmount: 1_850_000,
      departuresByDestination: [
        { stage: "APPLICATION", caseCount: 2, amount: 400_000 },
        { stage: "WITHDRAWN", caseCount: 1, amount: 200_000 },
      ],
      residual: 0, reconciles: true,
    },
    {
      stage: "OFFER",
      openingCaseCount: 10, openingAmount: 3_000_000,
      arrivalCaseCount: 0, arrivalAmount: 0,
      departureCaseCount: 0, departureAmount: 0,
      persistingCaseCount: 10, amountChangeOnPersisting: 0,
      closingCaseCount: 10, closingAmount: 3_000_000,
      departuresByDestination: [],
      residual: 0, reconciles: true,
    },
  ],
};

describe("StageMovementPanel", () => {
  it("renders one row per live stage", () => {
    render(<StageMovementPanel movement={MOVEMENT} />);
    expect(screen.getByText("KFI")).toBeInTheDocument();
    expect(screen.getByText("Offer")).toBeInTheDocument();
  });

  it("splits departures by where the case actually went", () => {
    // "Left the stage" and "left the pipeline" are different events, and a
    // completion is not attrition.
    render(<StageMovementPanel movement={MOVEMENT} />);
    const destinations = screen.getByTestId("stage-movement-destinations");
    expect(destinations).toHaveTextContent("Application");
    expect(destinations).toHaveTextContent("Withdrawn");
  });

  it("draws a zero leg as a dash", () => {
    // "−0  £0" reads as a rendering fault rather than as nothing happening.
    render(<StageMovementPanel movement={MOVEMENT} />);
    expect(screen.getAllByText("—").length).toBeGreaterThan(0);
  });

  it("states the engine's own reason when identity cannot be governed", () => {
    // There is deliberately no fallback: without a stable case key the only
    // honest answer is that this cannot be reported.
    render(<StageMovementPanel movement={{
      dataset: "pipeline_movement", portfolioId: "p", available: false,
      reason: "case-level movement needs a stable pipeline_case_identifier",
      stages: [],
    }} />);
    expect(screen.getByTestId("stage-movement-unavailable"))
      .toHaveTextContent("stable pipeline_case_identifier");
  });

  it("shows the window the movement was measured over", () => {
    render(<StageMovementPanel movement={MOVEMENT} />);
    expect(screen.getByTestId("stage-movement"))
      .toHaveTextContent("2026-06-12");
  });

  it("names the identifier the reconciliation is joined on", () => {
    render(<StageMovementPanel movement={MOVEMENT} />);
    expect(screen.getByTestId("stage-movement"))
      .toHaveTextContent("pipeline_case_identifier");
  });

  it("says so plainly when nothing left a stage", () => {
    render(<StageMovementPanel movement={{
      ...MOVEMENT,
      stages: [{ ...MOVEMENT.stages[1] }],
    }} />);
    expect(screen.getByTestId("stage-movement"))
      .toHaveTextContent("No case left a live stage");
  });
});
