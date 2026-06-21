import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import type { ChartArtifact } from "@/domain";
import { formatValue } from "@/lib/utils";
import { ChartArtifactView } from "./ChartArtifactView";

function bubble(overrides: Partial<ChartArtifact> = {}): ChartArtifact {
  return {
    id: "c1",
    type: "chart",
    title: "Balance by LTV by age",
    createdAt: new Date().toISOString(),
    mock: false,
    chartType: "bubble",
    xKey: "youngest_borrower_age",
    yKey: "current_loan_to_value",
    sizeKey: "current_outstanding_balance",
    xLabel: "Youngest Borrower Age",
    yLabel: "Current Loan To Value",
    sizeLabel: "Current Outstanding Balance",
    xFormat: "number",
    yFormat: "pct",
    sizeFormat: "gbp",
    xScale: 1,
    yScale: 100,
    sizeScale: 1,
    series: [
      { key: "youngest_borrower_age", label: "Youngest Borrower Age", color: "#1" },
      { key: "current_loan_to_value", label: "Current Loan To Value", color: "#2" },
      { key: "current_outstanding_balance", label: "Current Outstanding Balance", color: "#3" },
    ],
    rows: [
      { youngest_borrower_age: 71, current_loan_to_value: 0.314, current_outstanding_balance: 184000 },
      { youngest_borrower_age: 68, current_loan_to_value: 0.51, current_outstanding_balance: 220000 },
    ],
    ...overrides,
  } as ChartArtifact;
}

describe("ChartArtifactView (bubble)", () => {
  it("renders a bubble chart from explicit role keys", () => {
    const { container } = render(<ChartArtifactView artifact={bubble()} />);
    expect(container.querySelector(".recharts-responsive-container")).toBeTruthy();
  });

  it("shows a controlled error (not a blank chart) when a role key is missing", () => {
    render(<ChartArtifactView artifact={bubble({ yKey: undefined })} />);
    expect(screen.getByRole("alert")).toHaveTextContent(/missing required key/i);
    expect(screen.getByRole("alert")).toHaveTextContent(/yKey/);
  });

  it("requires sizeKey specifically for bubble", () => {
    render(<ChartArtifactView artifact={bubble({ sizeKey: undefined })} />);
    expect(screen.getByRole("alert")).toHaveTextContent(/sizeKey/);
  });
});

describe("formatValue pct scaling", () => {
  it("renders a 0..1 fraction as a percentage with scale 100", () => {
    expect(formatValue(0.51, "pct", 100)).toBe("51.0%");
    expect(formatValue(0.314, "pct", 100)).toBe("31.4%");
  });

  it("leaves an already-percentage value unscaled (scale 1)", () => {
    expect(formatValue(51, "pct", 1)).toBe("51.0%");
    expect(formatValue(51, "pct")).toBe("51.0%");
  });
});
