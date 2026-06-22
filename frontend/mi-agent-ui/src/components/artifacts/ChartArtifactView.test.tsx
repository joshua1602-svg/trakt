import { render } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import type { ChartArtifact } from "@/domain";
import { ChartArtifactView } from "./ChartArtifactView";

function bubble(): ChartArtifact {
  return {
    id: "art_1",
    type: "chart",
    title: "LTV by age sized by balance",
    source: { engine: "mi_agent.workflow", label: "MI Agent · bubble" },
    createdAt: new Date().toISOString(),
    mock: false,
    chartType: "bubble",
    xKey: "youngest_borrower_age",
    yKey: "current_loan_to_value",
    sizeKey: "current_outstanding_balance",
    xLabel: "Borrower Age",
    yLabel: "Current LTV",
    sizeLabel: "Balance",
    valueFormat: "pct",
    displayHints: {
      youngest_borrower_age: { format: "number", scale: null },
      current_loan_to_value: { format: "pct", scale: "percent_fraction" },
      current_outstanding_balance: { format: "gbp", scale: null },
    },
    series: [
      { key: "youngest_borrower_age", label: "Borrower Age", color: "#919dd1" },
      { key: "current_loan_to_value", label: "Current LTV", color: "#232d55" },
      { key: "current_outstanding_balance", label: "Balance", color: "#3d4a82" },
    ],
    rows: [
      { youngest_borrower_age: 67, current_loan_to_value: 0.29, current_outstanding_balance: 250000 },
      { youngest_borrower_age: 72, current_loan_to_value: 0.56, current_outstanding_balance: 310000 },
    ],
  };
}

describe("ChartArtifactView bubble", () => {
  it("renders a bubble chart from explicit x/y/size keys (yKey not null)", () => {
    const art = bubble();
    expect(art.yKey).toBe("current_loan_to_value");
    expect(art.sizeKey).toBe("current_outstanding_balance");
    // Renders without throwing (the renderer consumes xKey/yKey/sizeKey directly).
    const { container } = render(<ChartArtifactView artifact={art} />);
    expect(container.querySelector(".recharts-responsive-container")).not.toBeNull();
  });
});
