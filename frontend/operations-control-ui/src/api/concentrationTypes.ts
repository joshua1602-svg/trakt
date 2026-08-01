/**
 * Types for the Concentration Tests review area.
 *
 * These mirror the API's response shapes exactly (`/ops/concentration/*`).
 * Nothing here computes anything: match outcomes, concerns, blockers and the
 * activation result all arrive already decided by the backend, because the
 * governed controls that decide them must not be re-implemented in a browser.
 *
 * Wire casing is mixed by design and mirrored faithfully: the review package
 * envelope is camelCase; proposal and version documents are snake_case.
 */

export type ProposalStatus =
  | "proposed"
  | "pending_confirmation"
  | "pending_approval"
  | "approved"
  | "rejected"
  | "superseded"
  | "unsupported"
  | "not_applicable"
  | "clarification_requested";

export type MatchOutcome = "matched" | "composed" | "ambiguous" | "unsupported";

export interface SourceEvidence {
  source_reference: string;
  source_text: string;
  locator: string;
  supplied_at: string;
}

export interface ApprovalRecord {
  decision: string;
  operator: string;
  decided_at: string;
  approved_metric_id: string;
  approved_parameters: Record<string, unknown>;
  source_version: number;
  effective_date: string;
  comments: string;
  prior_active_test_id: string;
}

export interface OperatorNote {
  at: string;
  actor: string;
  note: string;
}

export interface ConfirmationAnswer {
  question: string;
  answer: string;
  answered_by: string;
  answered_at: string;
}

export interface TestProposal {
  proposal_id: string;
  onboarding_run_id: string;
  client_id: string;
  portfolio_id: string;
  evidence: SourceEvidence;
  extracted_test_name: string;
  extracted_category: string;
  proposed_metric_id: string;
  match_outcome: MatchOutcome;
  extracted_operator: string;
  extracted_threshold: number | null;
  extracted_unit: string;
  numerator_basis: string;
  denominator_basis: string;
  weighting_basis: string;
  parameters: Record<string, unknown>;
  segment_filters: Record<string, unknown>;
  bucket_boundaries: number[];
  effective_date: string;
  expiry_date: string;
  cure_or_waiver_notes: string;
  extraction_confidence: number;
  mapping_confidence: number;
  concern_codes: string[];
  concern_messages: string[];
  confirmation_questions: string[];
  confirmation_answers: ConfirmationAnswer[];
  status: ProposalStatus;
  approval: Partial<ApprovalRecord>;
  severity: string;
  operator_notes: OperatorNote[];
  implementation_request: Record<string, unknown>;
  created_at: string;
  updated_at: string;
  version: number;
}

export interface VersionHeader {
  version: number;
  status: string;
  activated_by: string;
  activated_at: string;
  test_count: number;
  content_hash: string;
  reason: string;
}

export interface ActiveTestDoc {
  test_id: string;
  proposal_id: string;
  metric_id: string;
  display_name: string;
  category: string;
  operator: string;
  threshold: number | null;
  unit: string;
  parameters: Record<string, unknown>;
  effective_date: string;
  evidence: SourceEvidence;
  approval: ApprovalRecord;
  severity: string;
}

export interface ActiveConfigurationDoc {
  client_id: string;
  version: number;
  status: string;
  tests: ActiveTestDoc[];
  activated_by: string;
  activated_at: string;
  based_on_version: number | null;
  reason: string;
  content_hash: string;
}

export interface ConcentrationReviewPackage {
  clientId: string;
  proposals: TestProposal[];
  countsByStatus: Record<string, number>;
  currentVersion: number | null;
  currentTestCount: number;
  versions: VersionHeader[];
  libraryVersion: string;
}

export interface LibraryMetricDoc {
  metric_id: string;
  display_name: string;
  category: string;
  description: string;
  unit: string;
  supported_operators: string[];
  parameters: Record<string, Record<string, unknown>>;
  implementation_status: string;
}

export interface ProposalUpdatePayload {
  metric_id?: string;
  parameters?: Record<string, unknown>;
  threshold?: number;
  operator?: string;
  effective_date?: string;
  expiry_date?: string;
  resolved_concerns?: string[];
  note?: string;
}
