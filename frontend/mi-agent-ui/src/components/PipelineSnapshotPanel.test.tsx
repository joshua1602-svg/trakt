import { describe, expect, it } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import type { PipelineSnapshot } from "@/domain";
import { PipelineSnapshotPanel } from "./PipelineSnapshotPanel";
import { mockForecastSnapshot } from "@/data/mockForecast";

const NOV = mockForecastSnapshot("client_001/mi_2025_11").pipelineSnapshot!;

describe("PipelineSnapshotPanel", () => {
  it("renders the pipeline SSoT KPIs (separate from the funded book)", () => {
    render(<PipelineSnapshotPanel snapshot={NOV} />);
    expect(screen.getByText("Pipeline Snapshot")).toBeInTheDocument();
    expect(screen.getByText(/Origination pipeline \(pre-funded\)/)).toBeInTheDocument();
    expect(screen.getByText("Pipeline cases")).toBeInTheDocument();
    expect(screen.getByText("10")).toBeInTheDocument();
    expect(screen.getByText("Weighted expected funded")).toBeInTheDocument();
  });

  it("renders the pipeline stage breakdown (amount and count)", () => {
    render(<PipelineSnapshotPanel snapshot={NOV} />);
    expect(screen.getByText("Pipeline amount by stage")).toBeInTheDocument();
    expect(screen.getByText("Pipeline count by stage")).toBeInTheDocument();
    // Stage labels appear in the breakdown bars.
    expect(screen.getAllByText("OFFER").length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText("COMPLETED").length).toBeGreaterThanOrEqual(1);
  });

  it("renders the expected completion breakdown when months exist", () => {
    render(<PipelineSnapshotPanel snapshot={NOV} />);
    expect(screen.getByText("Weighted expected funded by completion month")).toBeInTheDocument();
    // 2025-12 appears in both the "next completions" tile and the month bar.
    expect(screen.getAllByText("2025-12").length).toBeGreaterThanOrEqual(1);
  });

  it("hides technical diagnostics until expanded", () => {
    const snap: PipelineSnapshot = {
      ...NOV,
      dataQuality: [{ check: "amount_parse_partial", severity: "warning", detail: "economic amount parsed for 9/10 rows" }],
    };
    render(<PipelineSnapshotPanel snapshot={snap} />);
    expect(screen.queryByText(/economic amount parsed for 9\/10 rows/)).not.toBeInTheDocument();
    fireEvent.click(screen.getByText(/Technical details/));
    expect(screen.getByText(/economic amount parsed for 9\/10 rows/)).toBeInTheDocument();
  });

  it("renders an unavailable state gracefully", () => {
    const snap = { ...NOV, ok: false, error: "No pipeline data for this reporting date." };
    render(<PipelineSnapshotPanel snapshot={snap} />);
    expect(screen.getByText("Pipeline Snapshot unavailable")).toBeInTheDocument();
    expect(screen.getByText(/No pipeline data for this reporting date/)).toBeInTheDocument();
  });
});
