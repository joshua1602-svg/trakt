/**
 * Types for the OCC Agent tab.
 *
 * These mirror the API's response shapes exactly. Nothing here computes
 * anything: the lifecycle, the readiness verdict, the decision cards and the
 * status text all arrive already decided by the backend, because the controls
 * that decide them are governed and must not be re-implemented in a browser.
 */

/** The synthetic runtime policy, as recorded on every case. */
export interface SyntheticPolicy {
  runtime_mode: string;
  allow_external_email: boolean;
  allow_live_blob_write: boolean;
  allow_live_pipeline_trigger: boolean;
  allow_production_config_write: boolean;
  allow_publish: boolean;
  allow_live_case_access: boolean;
}

/** One state in the case lifecycle, with its full contract. */
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
  /** Present on the per-case lifecycle, absent on the catalogue. */
  reached?: boolean;
  current?: boolean;
}

/** The list-view projection of a case. */
export interface CaseSummary {
  case_id: string;
  tenant: string;
  client_id: string;
  client_name: string;
  portfolio_id: string;
  portfolio_name: string;
  asset_type: string;
  state: string;
  state_label: string;
  readiness_status: string;
  runtime_mode: string;
  synthetic: boolean;
  open_decisions: number;
  fixture_id: string;
  created_at: string;
  updated_at: string;
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

export interface ProposedValue {
  key: string;
  value: unknown;
  source: string;
  evidence: string;
  confidence: number | null;
  downstream_impact: string;
  material: boolean;
  requires_human_confirmation: boolean;
  confirmed: boolean;
  confirmed_by: string;
}

export interface ReadinessCriterion {
  key: string;
  label: string;
  passed: boolean;
  detail: string;
  remedy: string;
}

export interface Readiness {
  ready: boolean;
  status: string;
  criteria: ReadinessCriterion[];
  outstanding: ReadinessCriterion[];
}

/** The persisted case document. Chat history is presentation, not state. */
export interface SyntheticCaseDoc {
  case_id: string;
  tenant: string;
  initiating_user: string;
  state: string;
  runtime_mode: string;
  case_version: number;
  synthetic: boolean;
  client_id: string;
  client_name: string;
  portfolio_id: string;
  portfolio_name: string;
  asset_type: string;
  extracted_requirements: Record<string, unknown>;
  confirmed_requirements: Record<string, unknown>;
  unresolved_questions: string[];
  onboarding_pack: Record<string, unknown>;
  pack_issued_synthetically_at: string;
  required_artefacts: { role: string; label: string; required: boolean }[];
  received_artefacts: SyntheticArtefact[];
  proposed_configuration: Record<string, unknown>;
  confirmed_configuration: Record<string, unknown>;
  configuration_provenance: ProposedValue[];
  mapping_decisions: Record<string, unknown>[];
  open_decisions: DecisionCard[];
  control_results: Record<string, unknown>[];
  stage_outcomes: Record<string, string>;
  blockers: string[];
  observations: string[];
  planned_pipeline_actions: Record<string, unknown>[];
  orchestration_plan: Record<string, unknown>;
  assembler_plan: Record<string, unknown>;
  readiness_status: string;
  readiness_package_ref: string;
  approval_history: Record<string, unknown>[];
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
  case: SyntheticCaseDoc;
  summary: CaseSummary;
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
  change?: {
    kind: string;
    updates: Record<string, unknown>;
    before: Record<string, unknown>;
    affected_pack_sections: string[];
    material: boolean;
    summary: string;
  };
}

export interface ScenarioSummary {
  fixture_id: string;
  label: string;
  description: string;
  instruction: string;
  files: { filename: string; artefact_type: string; bytes: number }[];
  expected_state: string;
  expected_human_steps: string[];
  demonstrates: string;
}

export interface AgentMeta {
  enabled: boolean;
  flag: string;
  runtime_mode: string;
  policy: SyntheticPolicy;
  lifecycle: LifecycleState[];
  returnable_states: string[];
  scenarios: ScenarioSummary[];
  tenant: string;
}

export interface AgentAudit {
  events: Record<string, unknown>[];
  chain_intact: boolean;
}

export interface AgentPack {
  pack: {
    sections?: {
      key: string;
      title: string;
      body: string;
      items: Record<string, unknown>[];
      depends_on: string[];
    }[];
    outcome?: string;
    required_roles?: string[];
    optional_roles?: string[];
    content_hash?: string;
  };
  required_artefacts: { role: string; label: string; required: boolean }[];
  issued_at: string;
}

export interface AgentReadinessPackage {
  readiness: Readiness;
  package: Record<string, unknown> | null;
}
