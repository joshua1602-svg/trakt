/**
 * Risk Limits / Concentration shapes — mirror `mi_agent_api.risk_limits`.
 * Schedule 8 extracted limits vs funded actual exposure: actual, limit,
 * headroom, status, movement, source, confidence and missing fields, with
 * controlled needs-review / unavailable states.
 */

export type RiskStatus = "green" | "amber" | "red" | "needs_review" | "unavailable";

export interface RiskLimitTest {
  limitId: string;
  category: string;
  label: string;
  region: string | null;
  dimensionKey: string | null;
  actualValue: number | null;
  actualBasis: string;
  limitValue: number | null;
  unit: string | null;
  direction: string;
  headroom: number | null;
  status: RiskStatus;
  movementVsPrior: number | null;
  source: string;
  confidence: string | null;
  notes: string;
  missingFields: string[];
  sourceSnippet?: string | null;
  sourceSection?: string | null;
}

export interface RiskLimitsSummary {
  testsPassed: number;
  warnings: number;
  breaches: number;
  needsReview: number;
  unavailable: number;
  total: number;
  closestHeadroom: { label: string; headroom: number; limitId: string } | null;
  largestConcentration: { label: string; actualValue: number; limitId: string } | null;
}

export interface RiskLimitsSnapshot {
  portfolioId: string;
  toRunId: string | null;
  reportingDate: string | null;
  available: boolean;
  limitsStatus: string;
  limitsSource: string;
  limitsReason?: string | null;
  fundedDataAvailable: boolean;
  summary: RiskLimitsSummary;
  testsByCategory: Record<string, RiskLimitTest[]>;
  tests: RiskLimitTest[];
  observations: string[];
  lineage?: Record<string, unknown> & {
    limitSource?: string;
    sourceDocument?: string | null;
    dataSource?: string;
    exposureBasis?: string;
    reportingDate?: string | null;
    extractionMethod?: string;
    needsReviewCount?: number;
  };
  error?: string;
}
