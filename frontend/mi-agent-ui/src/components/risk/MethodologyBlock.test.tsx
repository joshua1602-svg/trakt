import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { MethodologyBlock } from "./MethodologyBlock";

describe("MethodologyBlock", () => {
  it("describes the stage run-off when the expected state uses it", () => {
    render(
      <MethodologyBlock
        forecast={{
          available: true,
          methodology: "stage_runoff",
          observationWindowStart: "2025-09-08",
          observationWindowEnd: "2026-09-24",
          weeklyExtractsUsed: 90,
          stageRates: { KFI: { rate: 0.05, observed: 5486, sufficient: true } },
          runoff: {
            available: true,
            appToOfferPullThrough: 0.62,
            offerToCompletionPullThrough: 0.71,
            stages: { APPLICATION: { windowDays: 28, windowBasis: "measured" } },
          },
          notForecast: { kfiCount: 4750, lapsedCount: 312 },
        }}
      />,
    );
    const block = screen.getByTestId("methodology-block");
    expect(block).toHaveTextContent("Stage run-off");
    expect(block).toHaveTextContent("Application → Offer 62%");
    expect(block).toHaveTextContent("4750 KFIs");
    expect(block).toHaveTextContent("Every live Application and Offer at 100%");
    // The superseded flat stage rates are not shown beside it.
    expect(block).not.toHaveTextContent("Kfi 0.05");
  });
});
