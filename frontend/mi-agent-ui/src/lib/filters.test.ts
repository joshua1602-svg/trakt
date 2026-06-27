import { describe, expect, it } from "vitest";
import { formatPredicate, formatFilters } from "./filters";

describe("formatPredicate — no [object Object]", () => {
  it("formats a numeric comparison", () => {
    expect(formatPredicate("current_loan_to_value", { op: "gt", value: 40 }))
      .toBe("Current Loan To Value > 40");
  });

  it("formats a currency threshold with thousands separators", () => {
    expect(formatPredicate("current_outstanding_balance", { op: "gt", value: 200000 }))
      .toBe("Current Outstanding Balance > 200,000");
  });

  it("formats equality and ≤ / ≥", () => {
    expect(formatPredicate("youngest_borrower_age", { op: "eq", value: 60 }))
      .toBe("Youngest Borrower Age = 60");
    expect(formatPredicate("current_loan_to_value", { op: "le", value: 50 }))
      .toBe("Current Loan To Value ≤ 50");
    expect(formatPredicate("current_loan_to_value", { op: "ge", value: 50 }))
      .toBe("Current Loan To Value ≥ 50");
  });

  it("formats between", () => {
    expect(formatPredicate("current_loan_to_value", { op: "between", value: [20, 40] }))
      .toBe("Current Loan To Value between 20 and 40");
  });

  it("formats a categorical equality", () => {
    expect(formatPredicate("borrower_structure", "Joint")).toBe("Borrower Structure = Joint");
  });

  it("formats not_in with truncation (the Other drill)", () => {
    const out = formatPredicate("broker_channel",
      { op: "not_in", value: ["Alpha", "Beta", "Gamma", "Delta", "Eps"] });
    expect(out).toContain("not in [Alpha, Beta, Gamma, +2 more]");
  });

  it("never renders [object Object]", () => {
    const out = formatPredicate("x", { op: "gt", value: 1 });
    expect(out).not.toContain("[object Object]");
    expect(formatFilters({ a: { op: "gt", value: 1 }, b: "Y" }))
      .toBe("A > 1 · B = Y");
  });
});
