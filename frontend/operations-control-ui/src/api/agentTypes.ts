/**
 * Types for the OCC Agent tab.
 *
 * These mirror the API's response shapes exactly. Nothing here computes
 * anything: the lifecycle, the readiness verdict, the decision cards and the
 * status text all arrive already decided by the backend, because the controls
 * that decide them are governed and must not be re-implemented in a browser.
 *
 * A practice case is two records. `onboarding` is Client Onboarding's own case —
 * the same shape the `/onboarding` screens render — and `run` is the practice
 * execution beside it. They are never merged, here or anywhere else.
 */

import type {
  CasePreview,
  ChecklistRow,
  InformationRequest,
  OnboardingCase,
  OnboardingReference,
} from "./onboardingTypes";

/** The synthetic runtime policy, as recorded on every run. */
export interface SyntheticPolicy {
  runtime_mode: string;
  allow_external_email: boolean;
  allow_live_blob_write: boolean;
  allow_live_pipeline_trigger: boolean;
  allow_production_config_write: boolean;
  allow_publish: boolean;
  allow_live_case_access: boolean;
  /** Activation is Client Onboarding's own write boundary. Always false here. */
  allow_activate_configuration: boolean;
}

/** One state in the practice EXECUTION lifecycle, with its full contract. */
export interface LifecycleState {
  state: string;
  label: string;
  permitted_prior: string[];
  required_inputs: string[];
  automatic_actions: string[];
  deterministic_controls: string[];
  required_approvals: string[];
  allowed_human_actions: string[];
  next_states: string[];
  blocking_conditions: string[];
  occ_stage: string;
  terminal: boolean;
  /** Present on the per-run lifecycle, absent on the catalogue. */
  reached?: boolean;
  current?: boolean;
}

/** The list-view projection of a practice case. */
export interface CaseSummary {
  case_ref: string;
  tenant: string;
  state: string;
  state_label: string;
  readiness_status: string;
  runtime_mode: string;
  synthetic: boolean;
  open_decisions: number;
  fixture_id: string;
  created_at: string;
  updated_at: string;
  /** Joined from the onboarding case, so the list reads as a client list. */
  onboarding_status?: string;
  onboarding_status_label?: string;
  client_id?: string;
  client_name?: string;
  onboarding_missing?: boolean;
}

export interface AgentMessage {
  role: "operator" | "agent";
  text: string;
  at: string;
  refs: string[];
}

export interface SyntheticArtefact {
  artefact_id: string;
  source_file: string;
  artefact_type: string;
  synthetic_location: string;
  intended_live_uri: string;
  execution_status: string;
  sha256: string;
  size: number;
  columns: string[];
  row_count: number;
  recognition_confidence: number | null;
  recognition_basis: string;
  provided_by: string;
  provided_at: string;
  fixture_id: string;
}

/** A structured decision card. Everything the human needs is on it. */
export interface DecisionCard {
  decision_id: string;
  kind: string;
  title: string;
  question: string;
  blocking: boolean;
  status: string;
  issue: string;
  evidence: { label?: string; kind?: string; data?: unknown }[];
  recommendation: string;
  recommendation_source: string;
  confidence: number | null;
  materiality: string;
  downstream_consequence: string;
  options: { value: string; label: string }[];
  subject: Record<string, unknown>;
  resolved_value?: string;
  resolved_by?: string;
}

export interface ReadinessCriterion {
  key: string;
  label: string;
  passed: boolean;
  detail: string;
  remedy: string;
  /** Which half of the process the criterion belongs to. */
  stage: "onboarding" | "execution" | "boundary";
}

export interface Readiness {
  ready: boolean;
  status: string;
  criteria: ReadinessCriterion[];
  outstanding: ReadinessCriterion[];
}

/** The execution facts read off the onboarding case. Never entered by hand. */
export interface ExecutionFacts {
  client_id: string;
  client_name: string;
  portfolio_id: string;
  portfolio_name: string;
  asset_class: string;
  dataset: string;
  cadence: string;
  jurisdiction: string;
  products: string[];
  /** The products' display labels, from the product declaration. */
  product_labels: string[];
  outcome: string;
  regime: string;
  basis: Record<string, string>;
}

/** The persisted run document. Chat history is presentation, not state. */
export interface SyntheticRunDoc {
  case_ref: string;
  tenant: string;
  initiating_user: string;
  state: string;
  runtime_mode: string;
  version: number;
  synthetic: boolean;
  portfolio_id: string;
  dataset: string;
  reporting_period: string;
  facts: ExecutionFacts | Record<string, never>;
  received_artefacts: SyntheticArtefact[];
  stage_outcomes: Record<string, string>;
  mapping_report: Record<string, unknown>[];
  open_decisions: DecisionCard[];
  control_results: Record<string, unknown>[];
  planned_pipeline_actions: Record<string, unknown>[];
  orchestration_plan: Record<string, unknown>;
  assembler_plan: Record<string, unknown>;
  readiness: Readiness | Record<string, never>;
  readiness_status: string;
  readiness_package_ref: string;
  review_package_ref: string;
  /** Which adapter this case is worked under. "synthetic" unless live is on. */
  mode: string;
  pack: Record<string, unknown>;
  pack_status: string;
  pack_history: PackHistoryEntry[];
  pack_receipt: Record<string, unknown>;
  activation_intent: Record<string, unknown>;
  activation_result: Record<string, unknown>;
  approvals: Record<string, unknown>[];
  blockers: string[];
  observations: string[];
  messages: AgentMessage[];
  fixture_id: string;
  created_at: string;
  updated_at: string;
}

/** One step of the pack's own DRAFTED -> ... -> SENT workflow. */
export interface PackHistoryEntry {
  status: string;
  actor: string;
  at: string;
  note: string;
}

/**
 * Where one catalogue field sits, and therefore what happens to it.
 *
 * Only `client` becomes a question. The rest are pre-populated, derived,
 * deferred to the first delivery, or internal — and each is reported with a
 * reason, so "why is that not being asked?" always has an answer.
 */
export type FieldCategory =
  | "known"
  | "client"
  | "derived"
  | "first_delivery"
  | "internal"
  | "not_applicable";

export interface FieldClassification {
  section: string;
  field: string;
  label: string;
  category: FieldCategory;
  number: number;
  category_label: string;
  reason: string;
  source: string;
  required: boolean;
  confirm: boolean;
  value: unknown;
  provenance: string;
  index: number | null;
  item: string;
}

export interface ClassificationSummary {
  total: number;
  counts: Record<FieldCategory, number>;
  labels: Record<FieldCategory, string>;
  numbers: Record<FieldCategory, number>;
  client_facing: number;
  confirmations: number;
}

export interface AgentClassification {
  summary: ClassificationSummary;
  fields: FieldClassification[];
}

/** One question on the structured client form. */
export interface ClientFormField {
  /** `section.field` or `section[index].field`. The authoritative key. */
  key: string;
  section: string;
  field: string;
  label: string;
  help: string;
  type: string;
  options: { value: string; label: string }[];
  required: boolean;
  sensitive: boolean;
  evidence_required: boolean;
  max_length: number | null;
  validation: string;
  /** Pre-populated from what Trakt already knows. */
  value: unknown;
  index: number | null;
  item: string;
}

export interface ClientFormGroup {
  key: string;
  label: string;
  help: string;
  repeatable: boolean;
  index: number | null;
  item: string;
  fields: ClientFormField[];
  required: number;
}

export interface ClientFormStep {
  key: string;
  label: string;
  help: string;
  unlocked_by: string;
  groups: ClientFormGroup[];
  questions: number;
  required: number;
}

/** The progressive, conditional form — only what this client is asked. */
export interface ClientFormView {
  case_ref: string;
  client_name: string;
  steps: ClientFormStep[];
  /** Steps that exist but are not open yet, and what would open them. */
  locked: { step: string; label: string; unlocked_by: string }[];
  questions: number;
  required: number;
  content_hash: string;
}

export interface AgentClientForm {
  form: ClientFormView;
}

/** One question in the client pack. Every one is a catalogue field. */
export interface PackQuestion {
  section: string;
  field: string;
  label: string;
  help: string;
  /** "answered" | "outstanding" | "derived" */
  status: string;
  value: unknown;
  provenance: string;
  index: number | null;
  item: string;
  required: boolean;
  evidence_required: boolean;
  sensitive: boolean;
  writes_to: string;
  step: string;
  step_label: string;
}

export interface PackSection {
  key: string;
  label: string;
  help: string;
  repeatable: boolean;
  questions: PackQuestion[];
  outstanding: number;
  step: string;
  step_label: string;
  index: number | null;
  item: string;
}

/** What happened when the pack was issued. `sent` is the honest answer. */
export interface PackReceipt {
  adapter: string;
  sent: boolean;
  at: string;
  receipt_id: string;
  to: string[];
  subject: string;
  artefacts: { name?: string; ref?: string }[];
  content_hash: string;
  statement: string;
}

/** The pack as the case workspace shows it. */
export interface PackView {
  status: string;
  history: PackHistoryEntry[];
  outstanding: number;
  questions: number;
  sections: PackSection[];
  email: { to: string[]; cc: string[]; subject: string; body: string };
  artefacts: { name: string; ref: string }[];
  /** Category 1: already known, and material enough to put to a human. */
  confirmations: PackQuestion[];
  /** What the client is NOT asked, and why. Reported rather than hidden. */
  not_asked: { key: string; label: string; category: FieldCategory;
    number: number; reason: string }[];
  summary: ClassificationSummary | Record<string, never>;
  mapping_statement: string;
  receipt: PackReceipt | Record<string, never>;
  /** False in every environment without a mail integration. */
  sent: boolean;
}

/** What confirming activation would cause. Built from the case, never prose. */
export interface ActivationIntent {
  client_id: string;
  client_name: string;
  portfolio_id: string;
  dataset: string;
  reporting_period: string;
  files: { name: string; target: string; sha256?: string }[];
  target_locations: string[];
  actions: string[];
  configuration_artefacts: string[];
  statement: string;
}

export interface ActivationResult {
  ok: boolean;
  mode: string;
  client_id: string;
  version: number | null;
  artefacts_written: Record<string, unknown>[];
  batch_id: string;
  workflow_id: string;
  files_placed: string[];
  message: string;
  error: string;
}

/** Where the case stands on the road to production. */
export interface ActivationView {
  mode: string;
  adapter: string;
  /** The OCC_AGENT_LIVE_ENABLED flag. False in every checked-in environment. */
  live_enabled: boolean;
  intent: ActivationIntent | Record<string, never>;
  result: ActivationResult | Record<string, never>;
  approval: Record<string, unknown>;
}

/** Everything that must be true before anything reaches production. */
export interface ActivationPreconditions {
  mode: string;
  flag_enabled: boolean;
  case_ref: string;
  onboarding_status: string;
  configuration_approved: boolean;
  readiness_passed: boolean;
  tenant: string;
  client_id: string;
  portfolio_id: string;
  configuration_valid: boolean;
  configuration_problems: string[];
  artefacts_present: number;
  required_artefacts_satisfied: boolean;
  approval_audited: boolean;
  already_activated: boolean;
  confirmed: boolean;
}

/** The confirmation screen: what would happen, and every reason it may not. */
export interface AgentActivation {
  intent: ActivationIntent;
  preconditions: ActivationPreconditions;
  refusals: string[];
  mode: string;
  live_enabled: boolean;
}

/** A deep link into an existing OCC view, rather than a copy of it. */
export interface OccLink {
  label: string;
  to: string;
  why: string;
}

/** Everything the case workspace renders. One request, one shape. */
export interface AgentStatus {
  case_ref: string;
  run: SyntheticRunDoc;
  summary: CaseSummary;
  /** Client Onboarding's own presentation of the case, unchanged. */
  onboarding: OnboardingCase;
  facts: ExecutionFacts;
  state: LifecycleState;
  lifecycle: LifecycleState[];
  stage_outcomes: Record<string, string>;
  readiness: Readiness;
  policy: SyntheticPolicy;
  open_decisions: DecisionCard[];
  observations: string[];
  blockers: string[];
  occ_links: OccLink[];
  /** The client-facing half: drafted, approved, issued — and whether sent. */
  pack: PackView;
  review_package_ref: string;
  activation: ActivationView;
  /** True while a practice run is executing on its own thread. */
  running: boolean;
  /** True when a stage was simulated rather than executed for real. */
  anything_simulated: boolean;
  /** True when a stage was hard-blocked by a deterministic control. */
  anything_blocked: boolean;
  /** Always false. Surfaced so the tab can state it rather than imply it. */
  configuration_written: boolean;
}

/** What one natural-language turn returned. */
export interface AgentTurn extends AgentStatus {
  reply: string;
  applied: boolean;
  proposal: AgentProposal | null;
}

/** A material change the agent proposes but has not applied. */
export interface AgentProposal {
  proposal_id: string;
  action: string;
  payload: Record<string, unknown>;
  summary: string;
  basis: string;
  material: boolean;
  confidence: number;
  /** The four populations a turn must report. Present for answer proposals. */
  disclosure?: AgentDisclosure;
  plan?: Record<string, unknown>;
  complete?: boolean;
}

/**
 * What the agent understood, and — just as importantly — what it did not.
 *
 * A proposal carrying anything in `questions` or `unrecognised` has applied
 * NOTHING. Confirming applies only what `understood` lists.
 */
export interface AgentDisclosure {
  understood: string[];
  proposed: string;
  questions: string[];
  unrecognised: string[];
}

export interface ScenarioSummary {
  fixture_id: string;
  label: string;
  description: string;
  instruction: string;
  files: { filename: string; artefact_type: string; bytes: number }[];
  client_response: string[];
  reporting_period: string;
  expected_state: string;
  expected_onboarding_status: string;
  expected_human_steps: string[];
  demonstrates: string;
}

export interface AgentMeta {
  enabled: boolean;
  flag: string;
  runtime_mode: string;
  policy: SyntheticPolicy;
  lifecycle: LifecycleState[];
  /** The wizard's own reference data, so the tab never restates the catalogue. */
  onboarding_reference: OnboardingReference;
  scenarios: ScenarioSummary[];
  tenant: string;
}

export interface AgentAudit {
  events: Record<string, unknown>[];
  chain_intact: boolean;
  onboarding_events: Record<string, unknown>[];
}

/** What the client still has to tell us, and what has been asked so far. */
export interface AgentChecklist {
  checklist: ChecklistRow[];
  requests: InformationRequest[];
}

/** Exactly what activation would create. It creates nothing. */
export interface AgentPreview {
  preview: CasePreview;
  written: boolean;
  execution_status: string;
}

export interface AgentReadinessPackage {
  readiness: Readiness;
  package: Record<string, unknown> | null;
}

/** The pack, plus the document a human would actually read and send. */
export interface AgentPack {
  pack: Record<string, unknown> & { sections: PackSection[] };
  document: string;
  status: string;
  history: PackHistoryEntry[];
  receipt: PackReceipt | Record<string, never>;
}

/** The complete package an approver decides on. */
export interface AgentReviewPackage {
  package: Record<string, unknown>;
  document: string;
}
