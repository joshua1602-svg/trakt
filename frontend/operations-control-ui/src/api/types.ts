/** Payload types for the Trakt Operations Control API. */

export type WorkflowOutcome = "mi" | "mi_annex2";

export type WorkflowType = "new_client" | "new_portfolio" | "recurring" | "backfill";

export type WorkflowStatus =
  | "received"
  | "running"
  | "needs_review"
  | "blocked"
  | "awaiting_publication"
  | "published"
  | "held"
  | "cancelled"
  | "failed";

export type StageStatus =
  | "waiting"
  | "running"
  | "needs_review"
  | "blocked"
  | "ready"
  | "approved"
  | "rejected"
  | "completed";

export interface Delivery {
  delivery_id: string;
  client_id: string;
  portfolio_id: string;
  reporting_period: string;
  dataset: string;
  frequency: string;
  files: string[];
  classification: WorkflowType;
  classification_label: string;
  classification_sentence: string;
  registered_at: string;
}

export interface RegisterDeliveryInput {
  client_id: string;
  portfolio_id: string;
  input_path: string;
  dataset?: string;
  frequency?: string;
  reporting_period: string;
}

export interface WorkflowRow {
  workflow_id: string;
  client_id: string;
  portfolio_id: string;
  reporting_period: string;
  outcome: WorkflowOutcome;
  workflow_type: WorkflowType;
  workflow_type_label: string;
  status: WorkflowStatus;
  open_decisions: number;
  created_at: string;
  updated_at: string;
}

export interface EvidenceItem {
  label: string;
  kind: string;
  data: unknown;
}

export interface EvidenceTable {
  columns: string[];
  rows: (string | number)[][];
}

export interface Stage {
  stage: string;
  label: string;
  status: StageStatus;
  status_label: string;
  summary?: string;
  why_it_matters?: string;
  warnings?: string[];
  blockers?: string[];
  evidence?: EvidenceItem[];
  decision_count?: number;
}

export interface Workflow extends WorkflowRow {
  outcome_label: string;
  status_sentence: string;
  interrupted: boolean;
  blockers: string[];
  stages: Stage[];
  rerun_count: number;
}

export interface StartWorkflowInput {
  client_id: string;
  delivery_id: string;
  outcome: WorkflowOutcome;
  workflow_type?: WorkflowType;
  override_reason?: string;
  start: true;
}

export type BatchStatus =
  | "receiving"
  | "incomplete"
  | "classifying"
  | "review_required"
  | "configuration_required"
  | "ready"
  | "running"
  | "completed"
  | "failed";

export interface BatchFile {
  source_file_id: string;
  filename: string;
  role: string;
  role_label: string;
  status: string;
  status_sentence: string;
  confidence: number;
}

export interface BatchInputRole {
  role: string;
  label: string;
  required: boolean;
  satisfied: boolean;
}

export interface Batch {
  batch_id: string;
  client_id: string;
  portfolio_id: string;
  reporting_date: string;
  workflow_type: WorkflowOutcome;
  status: BatchStatus;
  status_label: string;
  status_sentence: string;
  auto_start_when_ready: boolean;
  files: BatchFile[];
  input_roles: BatchInputRole[];
  missing_roles: string[];
  configuration_ready: boolean;
  blocking_decisions: string[];
  workflow_id: string | null;
  created_at: string;
  updated_at: string;
}

export interface CreateBatchInput {
  client_id: string;
  portfolio_id: string;
  reporting_date: string;
  workflow_type: WorkflowOutcome;
  auto_start_when_ready: boolean;
}

export type ReviewScope = "file" | "portfolio" | "client" | "global";

export interface Recommendation {
  value: string;
  label: string;
  confidence: number;
}

export interface ReviewOption {
  value: string;
  label: string;
}

export interface Review {
  decision_id: string;
  workflow_id: string;
  client_id: string;
  stage: string;
  kind: string;
  title: string;
  question: string;
  blocking: boolean;
  status: string;
  recommendation: Recommendation | null;
  options: ReviewOption[];
  allowed_scopes: string[];
  default_scope: string;
  scope_explanations: Record<string, string>;
  created_at: string;
  resolved_by?: string;
  resolved_at?: string;
}

export interface DecisionInput {
  action: "approve" | "reject" | "amend";
  value: string;
  scope: string;
  reason: string;
}

export interface Rule {
  rule_id: string;
  version: number;
  kind: string;
  scope: string;
  client_id: string;
  portfolio_id: string;
  status: string;
  source_term: string;
  approved_meaning: string;
  description: string;
  approved_by: string;
  approved_at: string;
}

export interface DecisionResult {
  review: Review;
  rule: Rule | null;
  rerun_scheduled: boolean;
}

export interface Publication {
  publication_id: string;
  client_id: string;
  workflow_id: string;
  workflow_type_label: string;
  reporting_period: string;
  status: string;
  version: number;
  rule_version_count: number;
  previous_publication_id: string | null;
  approved_by: string;
  published_at: string | null;
  prepared_at: string;
}

export interface DashboardTiles {
  new_deliveries: number;
  needs_attention: number;
  blocked: number;
  ready_to_publish: number;
  recently_published: number;
}

export interface Dashboard {
  tiles: DashboardTiles;
  needs_attention: WorkflowRow[];
  recently_published: Publication[];
}

export interface RulesQuery {
  client?: string;
  q?: string;
  kind?: string;
  scope?: string;
}
