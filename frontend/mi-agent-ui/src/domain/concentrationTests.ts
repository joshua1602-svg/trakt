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

/** One forward-looking state block (Expected Forecast / Full Pipeline). */
export interface ConcentrationStateBlock {
  value: number | null;
  status: ConcentrationTestStatus | "indicative_only" | null;
  headroom: number | null;
  utilization: number | null;
  breachAmount: number | null;
}

export type ForecastTreatment =
  | "supported_with_exposure_weighting"
  | "supported_with_scenario_inclusion"
  | "indicative_only"
  | "state_independent"
  | "unsupported";

export interface EmergingRisk {
  category:
    | "current_breach"
    | "expected_breach"
    | "expected_warning_low_headroom"
    | "material_deterioration"
    | "full_pipeline_only_breach"
    | "data_or_methodology_limitation";
  rank: number;
  testId: string | null;
  displayName: string | null;
  statement: string;
  expectedHeadroom: number | null;
}

export interface ForecastMethodology {
  available: boolean;
  reason?: string;
  methodology?: string;
  basis?: string | null;
  observationWindowStart?: string | null;
  observationWindowEnd?: string | null;
  weeklyExtractsUsed?: number;
  trackedCaseCount?: number;
  observedCompletionCount?: number;
  minObservations?: number;
  stagesUsingHistoricalRates?: string[];
  stagesUsingConfigFallback?: string[];
  stageRates?: Record<
    string,
    { rate: number | null; observed: number; completed?: number; sufficient: boolean }
  >;
  stageTiming?: Record<string, { medianDays: number; observed: number }>;
  excludedStageCounts?: Record<string, number>;
  currentSnapshot?: string | null;
  pointInTimeNote?: string;
}

export interface PipelineDriver {
  caseId: string;
  balance: number;
  stage: string | null;
  dimensionValue: string | null;
  completionProbability: number;
  probabilitySource: string | null;
  expectedContribution: number;
  fullContribution: number;
  expectedCompletionMonth: string | null;
  impact: "tips_breach" | "tips_warning" | null;
}

export interface ConcentrationDrivers {
  available: boolean;
  reason?: string;
  testId?: string;
  displayName?: string;
  dimensionColumn?: string | null;
  drivers: PipelineDriver[];
  driverCount?: number;
  truncated?: boolean;
  expectedNumeratorMovement?: number;
  listedContribution?: number;
  topShareOfMovement?: number | null;
  reconciles?: boolean;
  reportingDate?: string | null;
  forecast?: Partial<ForecastMethodology>;
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
  /** The population the contract states this test is measured over. */
  population?: string | null;
  populationLabel?: string | null;
  populationBasis?: string | null;
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
  // ------ three-state extension (absent on legacy / states-unavailable) ----
  forecastTreatment?: ForecastTreatment;
  forecastTreatmentNote?: string;
  expected?: ConcentrationStateBlock | null;
  fullPipeline?: ConcentrationStateBlock | null;
  changeFundedToExpected?: number | null;
  changeFundedToFullPipeline?: number | null;
  expectedBreach?: boolean;
  fullPipelineBreach?: boolean;
  expectedNumerator?: number | null;
  expectedDenominator?: number | null;
  fullPipelineNumerator?: number | null;
  fullPipelineDenominator?: number | null;
  expectedBreachHorizon?: { available: boolean; period?: string | null; reason?: string } | null;
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
  expectedBreaches?: number;
  fullPipelineBreaches?: number;
  expectedWarnings?: number;
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
  forecast?: ForecastMethodology;
  states?: { available: boolean; reason?: string } & Record<string, unknown>;
  emergingRisks?: EmergingRisk[];
  lineage?: Record<string, unknown> & { source?: string; note?: string };
  /** The facility position, from the SAME frame and evaluation as `tests`. */
  borrowingBase?: BorrowingBaseSnapshot;
  facility?: FacilitySummary | null;
  eligiblePopulation?: EligiblePopulationDisclosure;
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

// --------------------------------------------------------------------------
// Borrowing base — mirrors `mi_agent_api.borrowing_base_api` /
// `mi_agent.borrowing_base`. Carried inside the concentration-test envelope so
// the Eligibility & Concentrations tab renders one consistent position from a
// single request, and served standalone at `/mi/borrowing-base`.
// --------------------------------------------------------------------------

/**
 * Every borrowing-base figure is a number OR the literal "NOT_CALCULABLE".
 * There is no third possibility, and a measure is never zero because an input
 * was missing — the UI must render the sentinel as a governed status, not as
 * a dash that reads like nothing to report.
 */
export const NOT_CALCULABLE = "NOT_CALCULABLE";
export type Measure = number | typeof NOT_CALCULABLE | null;

export function isCalculable(value: Measure): value is number {
  return typeof value === "number";
}

export interface FacilitySummary {
  clientId: string;
  facilityId: string;
  facilityLabel: string;
  facilityType: string;
  currency: string;
  commitment: number | null;
  advanceRate: number | null;
  advanceRatePct: number | null;
  concentrationDenominatorFloor: number | null;
  currentDrawnAmount: number | null;
  currentDrawnAmountAsOf: string | null;
  effectiveDate: string | null;
  maturityDate: string | null;
  environment: "prototype" | "production" | string;
  eligibilityRuleVersion: string | null;
  eligibilityRuleCount: number;
  eligibilityGoverned: boolean;
  prototypeAssumptionActive: boolean;
  concentrationPopulation: string;
  borrowingBaseTreatment: string;
  configSource: string;
  configVersion: string;
  configHash: string;
  governance?: Record<string, unknown>;
}

export interface BorrowingBaseInvariant {
  invariant: string;
  statement: string;
  expected: number;
  actual: number;
  holds: boolean;
}

export interface BreachedConcentration {
  testId: string | null;
  displayName: string | null;
  currentValue: number | null;
  threshold: number | null;
  unit: string | null;
  utilization: number | null;
  breachAmount: number | null;
  excessAmount: number | null;
  denominatorValue: number | null;
}

export interface BorrowingBaseSnapshot {
  available: boolean;
  reason?: string | null;
  portfolioId?: string;
  reportingDate?: string | null;
  toRunId?: string | null;
  facility: FacilitySummary | null;
  configurationProblems?: string[];
  eligibilityDerived?: boolean;

  facilityId?: string;
  currency?: string;

  // -- eligibility reconciliation ----------------------------------------
  financingPortfolioLoanCount?: number;
  financingPortfolioBalance?: number;
  eligibleLoanCount?: number;
  eligibleCurrentBalance?: number;
  eligibleShareOfFinancingPortfolioPct?: number | null;
  ineligibleLoanCount?: number;
  ineligibleCurrentBalance?: number;
  ineligibleShareOfFinancingPortfolioPct?: number | null;
  undeterminedLoanCount?: number;
  undeterminedCurrentBalance?: number;
  undeterminedShareOfFinancingPortfolioPct?: number | null;
  balanceColumn?: string;
  invariants?: BorrowingBaseInvariant[];
  reconciles?: boolean;

  // -- the facility calculation ------------------------------------------
  concentrationLimitDenominator?: Measure;
  concentrationLimitDenominatorFloor?: number | null;
  concentrationDenominatorFloorBinding?: boolean | null;
  advanceRate?: number | null;
  advanceRatePct?: number | null;
  grossBorrowingBase?: Measure;
  facilityCommitment?: number | null;
  availableBorrowingBase?: Measure;
  facilityCapBinding?: boolean | null;
  currentDrawnAmount?: Measure;
  borrowingBaseHeadroom?: Measure;
  borrowingBaseDeficiency?: Measure;
  borrowingBaseUtilisationPct?: Measure;
  facilityUtilisationPct?: Measure;

  concentrationAdjustment?: {
    treatment: string;
    amount: number;
    breachedTestCount: number;
    excessMeasuredPct: number | null;
    note: string;
  };

  // -- binding concentration (deterministic, never modelled) --------------
  nearestConcentrationLimit?: string | typeof NOT_CALCULABLE;
  nearestConcentrationLimitTestId?: string | null;
  nearestConcentrationHeadroomPct?: Measure;
  nearestConcentrationHeadroomAmount?: Measure;
  nearestConcentrationUtilisationPct?: Measure;
  nearestConcentrationStatus?: string;
  breachedConcentrationCount?: number;
  breachedConcentrations?: BreachedConcentration[];

  missingInputs?: string[];
  prototypeAssumptionsUsed?: string[];
  notes?: string[];
  measures?: Record<string, number | string>;
  receipt?: Record<string, unknown>;
}

/** How the Eligible Mortgage Loan population reaching the evaluator was formed. */
export interface EligiblePopulationDisclosure {
  basis:
    | "governed_eligibility"
    | "whole_book_no_facility_configured"
    | "whole_book_eligibility_not_derived";
  facilityId: string | null;
  eligibilityGoverned: boolean;
  prototypeAssumptionActive: boolean;
  eligibleLoanCount: number | null;
  fundedLoanCount: number;
  priorBasis: string;
  note: string;
}

/**
 * The loans carrying one governed eligibility status, with the reason each was
 * classified that way. Same governed field roles the concentration
 * drill-through discloses — an eligibility drill-down is a reason to show WHY
 * a loan was classified, not a wider view of the tape.
 */
export interface EligibilityLoans {
  available: boolean;
  reason?: string;
  status?: "ELIGIBLE" | "INELIGIBLE" | "UNDETERMINED";
  facilityId?: string | null;
  columns: string[];
  rows: Record<string, unknown>[];
  rowCount?: number;
  truncated?: boolean;
  reportingDate?: string | null;
  toRunId?: string | null;
  error?: string;
}
