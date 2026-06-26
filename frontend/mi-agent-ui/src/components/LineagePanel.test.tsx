import { describe, expect, it } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import type { ViewLineage } from "@/domain";
import { LineagePanel } from "./LineagePanel";

const lineage: ViewLineage = {
  view: "pipeline",
  source: "M2L KFI and Pipeline 2025_12_01_115711.xlsx",
  metric: "expected_funded_amount",
  weightedMetric: "expected_funded_amount × completion_probability",
  pipelineAsOfDate: "2025-12-01",
  pipelineSourceFolderDate: "2025-11-01",
  currentPipelineSnapshotDate: "2025-12-01",
  currentPipelineSourceFile: "M2L KFI and Pipeline 2025_12_01_115711.xlsx",
  historicalObservationWindowStart: "2025-10-06",
  historicalObservationWindowEnd: "2025-12-01",
  uniqueWeeklyExtractsUsed: 10,
  sourceFilesScanned: 26,
  observationWindowStart: "2025-10-06",
  observationWindowEnd: "2025-12-01",
  completionProbabilityBasis: "mixed_historical_and_config",
  historicalModelEvidence: {
    weeklyFilesUsed: 10,
    weeklyFileNames: [],
    observationWindowStart: "2025-10-06",
    observationWindowEnd: "2025-12-01",
    historicalRowsUsed: 15256,
    trackedCaseCount: 1526,
    observedCompletionCount: 412,
    stableIdentifierUsed: "pipeline_case_identifier (Account Number)",
    stagesUsingHistoricalRates: ["OFFER"],
    stagesUsingConfigFallback: ["KFI"],
    excludedStageCounts: { WITHDRAWN: 106 },
    completionProbabilityBasis: "mixed_historical_and_config",
    sourceFilesScanned: 26,
    uniqueWeeklyExtractsUsed: 10,
    duplicatesExcluded: 16,
    primarySourcePreference: "xlsx_over_csv",
    available: true,
  },
  explanation: "Origination pipeline.",
};

describe("LineagePanel — completion model evidence", () => {
  it("shows UNIQUE weekly extracts (not raw files) and the historical window", () => {
    render(<LineagePanel lineage={lineage} />);
    expect(screen.getByText("Completion model evidence")).toBeInTheDocument();
    // 10 unique weekly extracts · 15,256 historical rows · 1,526 tracked cases · ...
    expect(screen.getByText(/10 unique weekly extracts/)).toBeInTheDocument();
    expect(screen.getByText(/15,256 historical rows/)).toBeInTheDocument();
    expect(screen.getByText(/1,526 tracked cases/)).toBeInTheDocument();
    expect(screen.getByText(/Historical window: 2025-10-06 to 2025-12-01/)).toBeInTheDocument();
  });

  it("reports both files scanned and duplicates excluded", () => {
    render(<LineagePanel lineage={lineage} />);
    // 26 scanned, 16 duplicates removed -> 10 unique used.
    expect(screen.getByText(/16 duplicates excluded/)).toBeInTheDocument();
    expect(screen.getByText(/26 files scanned/)).toBeInTheDocument();
  });

  it("keeps the current snapshot as-of date distinct from the observation window", () => {
    render(<LineagePanel lineage={lineage} />);
    fireEvent.click(screen.getByText(/How calculated/));
    // The as-of (2025-12-01) and the window start (2025-10-06) are shown separately.
    expect(screen.getByText("Pipeline snapshot as-of")).toBeInTheDocument();
    expect(screen.getByText("Historical window")).toBeInTheDocument();
    expect(screen.getByText("2025-10-06 to 2025-12-01")).toBeInTheDocument();
    // The current source file and unique-vs-scanned count are surfaced.
    expect(screen.getByText("Current source file")).toBeInTheDocument();
    expect(screen.getByText("Unique weekly extracts used")).toBeInTheDocument();
    expect(screen.getByText("10 of 26 scanned")).toBeInTheDocument();
  });

  it("renders nothing for a missing lineage", () => {
    const { container } = render(<LineagePanel lineage={null} />);
    expect(container).toBeEmptyDOMElement();
  });
});
