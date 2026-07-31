/**
 * Client Onboarding — types mirroring `operations_control/onboarding/`.
 *
 * The browser renders the information model the backend serves; it does not
 * hold one. Adding a field to the governed catalogue reaches the wizard without
 * a change in this file.
 */

export type CaseKind = "new_client" | "migration" | "amendment";

export type CaseStatus =
  | "draft"
  | "information_requested"
  | "awaiting_client"
  | "in_review"
  | "changes_required"
  | "ready_for_approval"
  | "approved"
  | "activated"
  | "withdrawn";

export type FieldSource =
  | "client_supplied"
  | "operator_supplied"
  | "inferred"
  | "derived"
  | "trakt_default"
  | "system_generated"
  | "delivery_specific";

export interface FieldOption {
  value: string;
  label: string;
}

/** One question, exactly as the governed catalogue declares it. */
export interface CatalogueField {
  key: string;
  label: string;
  scope: string;
  source: FieldSource;
  type: string;
  help: string;
  required: boolean;
  required_when: string;
  validation: string;
  options: FieldOption[];
  options_from: string;
  default: unknown;
  derived_default: string;
  max_length: number | null;
  unique_across: string;
  consumers: string[];
  writes_to: string;
  evidence_required: boolean;
  sensitive: boolean;
  amendable: boolean;
  regime_code: string;
  /** Set for regime fields: which reporting product they belong to. */
  product: string;
}

export interface CatalogueSection {
  key: string;
  label: string;
  help: string;
  repeatable: boolean;
  repeatable_key: string;
  item_label_field: string;
  min_items: number;
  derived_from: string;
  from_regime: boolean;
  fields: CatalogueField[];
}

export interface Vocabulary {
  label: string;
  help: string;
  owner: string;
  values: FieldOption[];
}

export interface RegimeProduct {
  label: string;
  regime_id?: string;
  always_applies?: boolean;
  derived_from?: string;
  requires_dataset?: string;
  supported_by_assets_declaring?: string;
  fields: Record<string, unknown>[];
  unrepresented?: { key: string; label: string; reason: string }[];
}

export interface Catalogue {
  version: number;
  sections: CatalogueSection[];
  vocabularies: Record<string, Vocabulary>;
  not_collected: { key: string; label: string; source: string; reason: string }[];
  regime_products: Record<string, RegimeProduct>;
}

export interface OnboardingReference {
  catalogue: Catalogue;
  steps: { key: string; label: string }[];
  statuses: { key: CaseStatus; label: string }[];
  kinds: { key: CaseKind; label: string }[];
}

export interface CaseProblem {
  section: string;
  field: string;
  message: string;
  severity: "blocking" | "advisory";
  index: number | null;
  owner: "client" | "operator";
}

export interface ChecklistRow {
  section: string;
  section_label: string;
  field: string;
  label: string;
  help: string;
  index: number | null;
  scope: string;
  evidence_required: boolean;
  sensitive: boolean;
}

export interface InformationRequest {
  request_id: string;
  items: ChecklistRow[];
  responsible_party: string;
  status: string;
  requested_by: string;
  requested_at: string;
  sent_at: string;
  due_date: string;
  response_note: string;
  responded_at: string;
  responded_by: string;
  evidence: { name: string; reference: string; received_at?: string }[];
  reviewed_by: string;
  reviewed_at: string;
  review_note: string;
}

export interface OpenQuestion {
  question_id: string;
  question: string;
  origin: string;
  raised_at: string;
  raised_by: string;
  resolved_at?: string;
  resolved_by?: string;
  resolution?: string;
}

export interface CaseEvent {
  event: string;
  actor: string;
  at: string;
  reason: string;
  before: Record<string, unknown>;
  after: Record<string, unknown>;
  detail: Record<string, unknown>;
}

export interface ProductEligibility {
  label: string;
  applies: boolean;
  eligible: boolean;
  derived: boolean;
  reason: string;
}

/** Answers are shaped by the catalogue: a block per section, a list per
 *  repeatable section, and regime answers nested per product. */
export type CaseAnswers = Record<string, unknown>;

export interface OnboardingCase {
  case_id: string;
  kind: CaseKind;
  kind_label: string;
  status: CaseStatus;
  status_label: string;
  client_id: string;
  client_name: string;
  portfolios: number;
  outstanding_requests: InformationRequest[] | number;
  unresolved_questions: OpenQuestion[] | number;
  updated_at: string;
  updated_by: string;
  created_by: string;
  answers: CaseAnswers;
  provenance: Record<string, string>;
  based_on_version: number | null;
  information_requests: InformationRequest[];
  open_questions: OpenQuestion[];
  events: CaseEvent[];
  product_eligibility: Record<string, ProductEligibility>;
  steps: { key: string; label: string; problems: number }[];
  problems: CaseProblem[];
  blocking: CaseProblem[];
  by_step: Record<string, CaseProblem[]>;
  client_checklist: ChecklistRow[];
  ready: boolean;
  approved_by: string;
  approved_at: string;
  approval_reason: string;
  activated_version: number | null;
  withdrawal_reason: string;
}

export interface PlannedSourceRecord {
  client_id: string;
  source_portfolio_id: string;
  dataset: string;
  frequency: string;
  source_system: string | null;
  regime_required: boolean;
  status: string;
  action: string;
  portfolio_label: string;
  expected_location: string;
}

export interface PlannedArtefact {
  kind: string;
  rel: string;
  label: string;
  action: string;
  text: string;
  previous_text: string;
  records: PlannedSourceRecord[];
  summary: string[];
}

export interface AnswerChange {
  path: string;
  label: string;
  before: string;
  after: string;
  action: string;
}

export interface CasePreview extends Omit<OnboardingCase, "answers"> {
  artefacts: PlannedArtefact[];
  changes: AnswerChange[];
  current_version: number;
  next_version: number;
  unrepresented: { key?: string; label: string; reason: string; product?: string }[];
  generated_identifiers: { label: string; value: string }[];
  defaults_used: { label: string; value: string; section: string }[];
  answers: CaseAnswers;
}

export interface CaseRow {
  case_id: string;
  kind: CaseKind;
  kind_label: string;
  status: CaseStatus;
  status_label: string;
  client_id: string;
  client_name: string;
  portfolios: number;
  outstanding_requests: number;
  unresolved_questions: number;
  updated_at: string;
  updated_by: string;
  created_by: string;
}

export interface ActiveClientRow {
  client_id: string;
  display_name: string;
  version: number;
  portfolios: number;
  products: string[];
  activated_at: string;
  activated_by: string;
}

export interface OnboardingHome {
  drafts: CaseRow[];
  awaiting_client: CaseRow[];
  in_review: CaseRow[];
  approved: CaseRow[];
  active_clients: ActiveClientRow[];
  legacy_clients: { client_id: string; display_name: string }[];
  recently_completed: CaseRow[];
}

export interface ConfigurationVersionRow {
  version: number;
  status: string;
  approved_by: string;
  approved_at: string;
  activated_by: string;
  activated_at: string;
  reason: string;
  case_id: string;
  case_kind: string;
  based_on_version: number | null;
  change_count: number;
  changes: AnswerChange[];
  artefacts: { kind: string; target: string; action: string }[];
  content_hash: string;
}

export interface OnboardingClientDetail {
  client_id: string;
  status: string;
  version: number;
  activated_by?: string;
  activated_at?: string;
  reason?: string;
  answers: CaseAnswers | null;
  live_source_registrations: Record<string, unknown>[];
  artefacts?: { kind: string; target: string; action: string }[];
  cases: CaseRow[];
  history: ConfigurationVersionRow[];
  summary?: string;
}

export interface ActivationResult {
  case_id: string;
  client_id: string;
  version: number;
  artefacts: { kind: string; target: string; action: string }[];
  changes: AnswerChange[];
}
