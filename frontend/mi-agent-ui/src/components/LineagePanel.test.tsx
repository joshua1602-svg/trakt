import { describe, expect, it } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import type { ViewLineage } from "@/domain";
import { LineagePanel } from "./LineagePanel";

const lineage: ViewLineage = {
  view: "pipeline",
  source: "M2L_KFI_and_Pipeline_2025_12_01.csv",
  metric: "expected_funded_amount",
  weightedMetric: "expected_funded_amount × completion_probability",
  pipelineAsOfDate: "2025-12-01",
  pipelineSourceFolderDate: "2025-10-01",
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
    available: true,
  },
  explanation: "Origination pipeline.",
};

describe("LineagePanel — completion model evidence", () => {
  it("shows weekly files used and the observation window", () => {
    render(<LineagePanel lineage={lineage} />);
    expect(screen.getByText("Completion model evidence")).toBeInTheDocument();
    // 10 weekly files · 15,256 historical rows · 1,526 tracked cases · 412 observed completions
    expect(screen.getByText(/10 weekly files/)).toBeInTheDocument();
    expect(screen.getByText(/15,256 historical rows/)).toBeInTheDocument();
    expect(screen.getByText(/1,526 tracked cases/)).toBeInTheDocument();
    expect(screen.getByText(/Observation window: 2025-10-06 to 2025-12-01/)).toBeInTheDocument();
  });

  it("keeps the current snapshot as-of date distinct from the observation window", () => {
    render(<LineagePanel lineage={lineage} />);
    fireEvent.click(screen.getByText(/How calculated/));
    // The as-of (2025-12-01) and the window start (2025-10-06) are shown separately.
    expect(screen.getByText("Pipeline snapshot as-of")).toBeInTheDocument();
    expect(screen.getByText("Observation window")).toBeInTheDocument();
    expect(screen.getByText("2025-10-06 to 2025-12-01")).toBeInTheDocument();
  });

  it("renders nothing for a missing lineage", () => {
    const { container } = render(<LineagePanel lineage={null} />);
    expect(container).toBeEmptyDOMElement();
  });
});
