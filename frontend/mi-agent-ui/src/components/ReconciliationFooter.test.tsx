import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import type { TableArtifact } from "@/domain";
import { ReconciliationFooter } from "./ReconciliationFooter";

function tableArtifact(extra: Partial<TableArtifact>): TableArtifact {
  return {
    id: "t1",
    type: "table",
    title: "Result",
    source: { kind: "live", label: "MI Agent" } as never,
    createdAt: "2025-01-01T00:00:00Z",
    mock: false,
    columns: [],
    rows: [],
    ...extra,
  };
}

describe("ReconciliationFooter", () => {
  it("renders coverage figures tying back to the funded book", () => {
    render(
      <ReconciliationFooter
        artifact={tableArtifact({
          reconciliation: {
            total_balance: 8_900_000,
            balance_included: 8_900_000,
            coverage_by_balance_pct: 100,
            total_records: 100,
            records_included: 100,
            balance_excluded_missing: 0,
            missing_dimension_policy: "bucket",
          },
        })}
      />,
    );
    expect(screen.getByText("Reconciliation & coverage")).toBeInTheDocument();
    expect(screen.getByText(/Coverage by balance: 100%/)).toBeInTheDocument();
  });

  it("states the excluded balance when coverage is partial", () => {
    render(
      <ReconciliationFooter
        artifact={tableArtifact({
          reconciliation: {
            total_balance: 8_900_000,
            balance_included: 5_000_000,
            balance_excluded_missing: 3_900_000,
            coverage_by_balance_pct: 56.2,
            missing_dimension_policy: "exclude",
            missing_dimension_fields: ["age_bucket", "ltv_bucket"],
          },
        })}
      />,
    );
    expect(screen.getByText(/was excluded because age_bucket and\/or ltv_bucket was missing/)).toBeInTheDocument();
  });

  it("surfaces field provenance source notes", () => {
    render(
      <ReconciliationFooter
        artifact={tableArtifact({
          sourceNotes: [
            { field: "current_interest_rate", note: "Interest rate sourced from Product Rate in pipeline/KFI file. Confirm whether this is authoritative for funded-book MI." },
          ],
        })}
      />,
    );
    expect(screen.getByText(/Source note \(current_interest_rate\):/)).toBeInTheDocument();
    expect(screen.getByText(/Confirm whether this is authoritative/)).toBeInTheDocument();
  });

  it("renders nothing when there is no reconciliation or notes", () => {
    const { container } = render(<ReconciliationFooter artifact={tableArtifact({})} />);
    expect(container).toBeEmptyDOMElement();
  });
});
