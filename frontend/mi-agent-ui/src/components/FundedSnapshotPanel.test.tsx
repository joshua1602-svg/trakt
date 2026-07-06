import { describe, expect, it } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import type { FundedSnapshot } from "@/domain";
import { FundedSnapshotPanel } from "./FundedSnapshotPanel";
import { mockSnapshot } from "@/data/mockSnapshots";

describe("FundedSnapshotPanel", () => {
  it("renders the funded snapshot (label, reporting date, KPIs) before any query", () => {
    render(<FundedSnapshotPanel snapshot={mockSnapshot("client_001/mi_2025_11")} />);
    expect(screen.getByText("Funded Book Snapshot")).toBeInTheDocument();
    expect(screen.getByText("CLIENT_001")).toBeInTheDocument();
    expect(screen.getByText(/Reporting Date · 30 Nov 2025/)).toBeInTheDocument();
    expect(screen.getByText("Current funded balance")).toBeInTheDocument();
    expect(screen.getByText("73")).toBeInTheDocument();
  });

  it("shows month-on-month change versus the prior run", () => {
    render(<FundedSnapshotPanel snapshot={mockSnapshot("client_001/mi_2025_11")} />);
    expect(screen.getByText("Monthly change · loans")).toBeInTheDocument();
    // "+40" appears as both the loans-funded delta and the dedicated tile value.
    expect(screen.getAllByText("+40").length).toBeGreaterThanOrEqual(1);
    // The duplicate "New loans since prior run" tile is replaced by a
    // portfolio-aware risk tile (NNEG for ERM).
    expect(screen.queryByText("New loans since prior run")).toBeNull();
    expect(screen.getByText("NNEG exposure (current)")).toBeInTheDocument();
    expect(screen.getByText(/vs prior run · 2025-10-31/)).toBeInTheDocument();
  });

  it("renders point-in-time stratifications (balance by dimension)", () => {
    render(<FundedSnapshotPanel snapshot={mockSnapshot("client_001/mi_2025_11")} />);
    expect(screen.getByText(/Stratifications · balance by dimension/)).toBeInTheDocument();
    expect(screen.getByText("By LTV band")).toBeInTheDocument();
    expect(screen.getByText("By borrower age")).toBeInTheDocument();
    expect(screen.getByText("By region")).toBeInTheDocument();
    expect(screen.getByText("By rate band")).toBeInTheDocument();
    // A band label from the LTV breakdown is present.
    expect(screen.getByText("40–50%")).toBeInTheDocument();
  });

  it("omits the stratifications section when the tape carries none", () => {
    const snap: FundedSnapshot = { ...mockSnapshot("client_001/mi_2025_11"), stratifications: [] };
    render(<FundedSnapshotPanel snapshot={snap} />);
    expect(screen.queryByText(/Stratifications · balance by dimension/)).toBeNull();
  });

  it("shows 'No prior reporting date available' for the earliest run", () => {
    render(<FundedSnapshotPanel snapshot={mockSnapshot("client_001/mi_2025_10")} />);
    expect(screen.getByText(/No prior reporting date available/)).toBeInTheDocument();
  });

  it("hides technical diagnostics behind an expandable section", () => {
    const snap: FundedSnapshot = {
      ...mockSnapshot("client_001/mi_2025_10"),
      diagnostics: ["percent-scale heuristically detected as 'fraction'"],
    };
    render(<FundedSnapshotPanel snapshot={snap} />);
    // The technical note is not visible until expanded.
    expect(screen.queryByText(/percent-scale heuristically detected/)).not.toBeInTheDocument();
    fireEvent.click(screen.getByText(/Technical details/));
    expect(screen.getByText(/percent-scale heuristically detected/)).toBeInTheDocument();
  });
});
