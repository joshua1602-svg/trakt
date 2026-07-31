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
  approvals: Record<string, unknown>[];
  blockers: string[];
  observations: string[];
  messages: AgentMessage[];
  fixture_id: string;
  created_at: string;
  updated_at: string;
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
