/**
 * Concentration-test shapes — mirror `mi_agent_api.concentration_tests_api`.
 *
 * One governed evaluation service: the operator-APPROVED, versioned
 * concentration-test configuration evaluated against the funded book, with the
 * legacy Schedule 8 extracted limits presented in the same shape (explicitly
 * marked unapproved) while no approved configuration exists. The UI renders
 * these results verbatim; it never recalculates, never treats unavailable as
 * zero and never restyles an unapproved source as approved.
 */

export type ConcentrationTestStatus =
  | "pass"
  | "warning"
  | "breach"
  | "unavailable"
  | "insufficient_data"
  | "pending_effective_date"
  | "expired";

export type ConcentrationSource =
  | "approved_configuration"
  | "legacy_extracted"
  | "none";

export interface ConcentrationTestDefinition {
  numerator: string;
  aggregation: string;
  description: string;
  definitionNotes: string;
}

export interface ConcentrationTestProvenance {
  sourceReference: string;
  sourceText: string;
  locator: string;
  approvedBy: string;
  approvedAt: string;
  approvalComments: string;
  proposalId: string;
}

export interface ConcentrationTest {
  testId: string;
  metricId: string | null;
  displayName: string;
  category: string;
  reportingDate: string | null;
  currentValue: number | null;
  priorValue: number | null;
  priorReportingDate: string | null;
  priorAvailable: boolean;
  absoluteChange: number | null;
  percentagePointChange: number | null;
  relativeChange: number | null;
  threshold: number | null;
  operator: string;
  warningFraction: number;
  unit: string | null;
  utilization: number | null;
  headroom: number | null;
  status: ConcentrationTestStatus;
  priorStatus: ConcentrationTestStatus | null;
  statusTransition: string | null;
  deteriorated: boolean;
  breachAmount: number | null;
  dataStatus: string;
  missingFields: string[];
  numeratorValue: number | null;
  denominatorValue: number | null;
  denominatorBasis: string;
  loansInNumerator: number;
  totalLoans: number;
  severity: string;
  effectiveDate: string | null;
  expiryDate: string | null;
  notes: string;
  resolvedColumns: Record<string, string>;
  parameters: Record<string, unknown>;
  definition: ConcentrationTestDefinition;
  provenance: ConcentrationTestProvenance;
  configurationVersion: number | null;
  evaluatedAt: string | null;
  legacy?: boolean;
}

export interface ConcentrationSummary {
  overallStatus: ConcentrationTestStatus;
  activeTests: number;
  breaches: number;
  warnings: number;
  passes: number;
  unavailable: number;
  pendingEffectiveDate?: number;
  expired?: number;
  reportingDate: string | null;
  priorReportingDate: string | null;
  priorAvailable: boolean;
  deteriorations: number;
  closestToLimit: {
    testId: string;
    displayName: string;
    headroom: number;
    unit: string | null;
  } | null;
}

export interface ConcentrationTestsSnapshot {
  portfolioId: string;
  toRunId: string | null;
  reportingDate: string | null;
  priorReportingDate?: string | null;
  priorAvailable?: boolean;
  available: boolean;
  source: ConcentrationSource;
  approvalStatus: "approved" | "unapproved_legacy" | null;
  configurationVersion: number | null;
  configurationHash?: string;
  activatedBy?: string;
  activatedAt?: string;
  libraryVersion?: string;
  evaluatedAt?: string;
  fundedDataAvailable: boolean;
  proposalCounts?: Record<string, number>;
  openProposals?: number;
  unsupportedProposals?: number;
  tests: ConcentrationTest[];
  summary: ConcentrationSummary;
  lineage?: Record<string, unknown> & { source?: string; note?: string };
  error?: string;
}

export interface ConcentrationDrillthrough {
  available: boolean;
  reason?: string;
  testId?: string;
  displayName?: string;
  columns: string[];
  rows: Record<string, unknown>[];
  rowCount?: number;
  truncated?: boolean;
  loansInNumerator?: number;
  numeratorValue?: number | null;
  denominatorValue?: number | null;
  denominatorBasis?: string;
  reconciles?: boolean;
  reportingDate?: string | null;
  configurationVersion?: number;
}

export interface ConcentrationHistoryPoint {
  runId: string | null;
  reportingDate: string | null;
  value: number | null;
  status: ConcentrationTestStatus;
}

export interface ConcentrationHistorySeries {
  testId: string;
  displayName: string;
  unit: string | null;
  threshold: number | null;
  operator: string;
  warningFraction: number;
  points: ConcentrationHistoryPoint[];
}

export interface ConcentrationHistory {
  available: boolean;
  reason?: string;
  series: ConcentrationHistorySeries[];
  configurationVersion?: number;
  periods?: { runId: string | null; reportingDate: string | null }[];
}
