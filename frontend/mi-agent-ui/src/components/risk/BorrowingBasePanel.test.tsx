/**
 * The borrowing-base section of Eligibility & Concentrations.
 *
 * What these tests protect is the honesty of the panel, not its layout: a
 * measure the service could not produce must READ as not calculable and name
 * what is missing; a negative headroom must surface as a deficiency rather
 * than disappear; and a prototype eligibility assumption must be visible on
 * the page, not only in the receipt.
 */

import { describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import type { BorrowingBaseSnapshot } from "@/domain";
import { NOT_CALCULABLE } from "@/domain";
import { mockConcentrationTests } from "@/data/mockConcentrationTests";
import { BorrowingBasePanel } from "./BorrowingBasePanel";

const BASE = mockConcentrationTests("client_001").borrowingBase as BorrowingBaseSnapshot;

function snapshot(over: Partial<BorrowingBaseSnapshot> = {}): BorrowingBaseSnapshot {
  return { ...BASE, ...over };
}

describe("BorrowingBasePanel — the governed figures", () => {
  it("shows eligible collateral and the borrowing base from the envelope", () => {
    render(<BorrowingBasePanel snapshot={snapshot()} />);
    expect(screen.getByTestId("bb-eligible-collateral")).toHaveTextContent("£141.1MM");
    expect(screen.getByTestId("bb-borrowing-base")).toHaveTextContent("£145.3MM");
  });

  it("names the facility cap as binding only when it actually binds", () => {
    render(<BorrowingBasePanel snapshot={snapshot({ facilityCapBinding: true })} />);
    expect(screen.getByTestId("bb-borrowing-base")).toHaveTextContent(
      "capped at facility commitment",
    );
  });

  it("shows the Concentration Limit Denominator and whether the floor binds", () => {
    render(
      <BorrowingBasePanel
        snapshot={snapshot({
          concentrationLimitDenominator: 33_000_000,
          concentrationDenominatorFloorBinding: true,
        })}
      />,
    );
    expect(screen.getByTestId("borrowing-base-panel")).toHaveTextContent(
      "contractual floor binding",
    );
  });
});

describe("BorrowingBasePanel — what it refuses to invent", () => {
  it("renders a missing drawn balance as Not calculable, never as zero", () => {
    render(<BorrowingBasePanel snapshot={snapshot()} />);
    const drawn = screen.getByTestId("bb-facility-drawn");
    expect(drawn).toHaveTextContent("Not calculable");
    expect(drawn).not.toHaveTextContent("£0");
  });

  it("says WHICH input is missing rather than only that something is", () => {
    render(<BorrowingBasePanel snapshot={snapshot()} />);
    expect(screen.getByTestId("bb-headroom")).toHaveTextContent(
      "facility drawings not supplied",
    );
  });

  it("shows the prototype eligibility assumption on the page", () => {
    render(<BorrowingBasePanel snapshot={snapshot()} />);
    expect(screen.getByTestId("borrowing-base-prototype-banner")).toHaveTextContent(
      /PROTOTYPE ASSUMPTION/,
    );
  });

  it("carries the service's own note that breaches are not deducted", () => {
    render(<BorrowingBasePanel snapshot={snapshot()} />);
    expect(screen.getByTestId("borrowing-base-panel")).toHaveTextContent(
      /not the contractual consequence of a breach/,
    );
  });

  it("refuses to present unreconciled figures as governed", () => {
    render(<BorrowingBasePanel snapshot={snapshot({ reconciles: false })} />);
    expect(
      screen.getByTestId("borrowing-base-reconciliation-failed"),
    ).toHaveTextContent(/NOT a governed borrowing base/);
  });

  it("shows the empty state, with the reason, when no facility is configured", () => {
    render(
      <BorrowingBasePanel
        snapshot={{
          available: false,
          facility: null,
          reason: "No funding facility is configured for this portfolio.",
        }}
      />,
    );
    expect(screen.getByTestId("borrowing-base-unavailable")).toHaveTextContent(
      "No funding facility is configured",
    );
  });
});

describe("BorrowingBasePanel — an over-drawn facility", () => {
  const overDrawn = snapshot({
    currentDrawnAmount: 160_000_000,
    borrowingBaseHeadroom: -14_718_500,
    borrowingBaseDeficiency: 14_718_500,
    borrowingBaseUtilisationPct: 110.13,
    facilityUtilisationPct: 64,
    missingInputs: [],
  });

  it("shows £0 headroom in the tile but surfaces the deficiency beside it", () => {
    render(<BorrowingBasePanel snapshot={overDrawn} />);
    expect(screen.getByTestId("bb-headroom")).toHaveTextContent("£0");
    expect(screen.getByTestId("borrowing-base-deficiency")).toHaveTextContent("£14.7MM");
  });

  it("does not hide the deficiency anywhere in the panel", () => {
    render(<BorrowingBasePanel snapshot={overDrawn} />);
    expect(screen.getByTestId("borrowing-base-panel")).toHaveTextContent(
      /drawings exceed the available borrowing base/,
    );
  });
});

describe("BorrowingBasePanel — the eligibility split", () => {
  it("shows count, balance and share for each of the three statuses", () => {
    render(<BorrowingBasePanel snapshot={snapshot()} />);
    expect(screen.getByTestId("bb-split-eligible")).toHaveTextContent("1,180");
    expect(screen.getByTestId("bb-split-eligible")).toHaveTextContent("£141.1MM");
    expect(screen.getByTestId("bb-split-eligible")).toHaveTextContent("95.2%");
    expect(screen.getByTestId("bb-split-undetermined")).toHaveTextContent("60");
    expect(screen.getByTestId("bb-split-ineligible")).toHaveTextContent("0");
  });

  it("offers a drill-down to the loans behind each status", () => {
    const onShowLoans = vi.fn();
    render(<BorrowingBasePanel snapshot={snapshot()} onShowLoans={onShowLoans} />);
    fireEvent.click(screen.getByText("Show undetermined loans"));
    expect(onShowLoans).toHaveBeenCalledWith("UNDETERMINED");
  });

  it("discloses a stand-in population rather than presenting it as contractual", () => {
    render(
      <BorrowingBasePanel
        snapshot={snapshot()}
        population={{
          basis: "whole_book_no_facility_configured",
          facilityId: null,
          eligibilityGoverned: false,
          prototypeAssumptionActive: false,
          eligibleLoanCount: null,
          fundedLoanCount: 1_240,
          priorBasis: "whole_book_no_facility_configured",
          note: "That is a STAND-IN, not the contractual population.",
        }}
      />,
    );
    expect(screen.getByTestId("borrowing-base-panel")).toHaveTextContent(
      /STAND-IN, not the contractual population/,
    );
  });
});

describe("the NOT_CALCULABLE contract", () => {
  it("is a string sentinel, so a forgotten check is loud rather than plausible", () => {
    expect(NOT_CALCULABLE).toBe("NOT_CALCULABLE");
  });
});
