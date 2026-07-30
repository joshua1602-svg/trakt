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
  | "failed"
  | "withdrawn";

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

/** How far a delivery has got through its governed workflow. */
export type StepStatus = "complete" | "current" | "pending" | "blocked" | "not_applicable";

export interface StepFact {
  label: string;
  value: string;
}

export interface StepFile {
  filename: string;
  size: string;
  received_at: string;
  received_from: string;
  checksum: string;
  role_label: string;
}

export interface StepDecision {
  decision_id: string;
  title: string;
  question: string;
  blocking: boolean;
  status: string;
  answer: string;
  scope: string;
  actor: string;
  at: string;
  overridden: boolean;
}

export interface ApprovalScopeOption {
  value: string;
  label: string;
  explanation: string;
}

export interface ApprovalPanel {
  available: boolean;
  headline: string;
  evidence_lines: string[];
  question: string;
  scope_question: string;
  /** What "remember" actually means today. Recorded, not acted on. */
  scope_note?: string;
  scopes: ApprovalScopeOption[];
  default_scope: string;
  consequence: string;
  scope_consequences: Record<string, string>;
  version: number | null;
}

/** One step of the durable delivery case file. */
export interface WorkflowStep {
  key: string;
  label: string;
  status: StepStatus;
  status_label: string;
  summary: string;
  facts: StepFact[];
  warnings: string[];
  blockers: string[];
  detail?: string[];
  files?: StepFile[];
  resolved?: StepDecision[];
  unresolved?: StepDecision[];
  evidence?: EvidenceItem[];
  approval?: ApprovalPanel;
  outputs?: string[];
}

export interface Workflow extends WorkflowRow {
  outcome_label: string;
  status_sentence: string;
  interrupted: boolean;
  blockers: string[];
  stages: Stage[];
  /** Set once an operator has taken this delivery out of active work. */
  withdrawn?: WithdrawalRecord | null;
  /** The governed case file. Present from every backend that serves steps. */
  steps?: WorkflowStep[];
  current_step?: string;
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

/** Which book an input pack describes. Regime reporting is prepared from the
 * funded book only; pipeline is a forward-looking MI view. */
export type BatchDataset = "funded" | "pipeline";

export interface Batch {
  batch_id: string;
  client_id: string;
  portfolio_id: string;
  reporting_date: string;
  workflow_type: WorkflowOutcome;
  dataset: BatchDataset;
  dataset_label: string;
  status: BatchStatus;
  status_label: string;
  status_sentence: string;
  auto_start_when_ready: boolean;
  files: BatchFile[];
  input_roles: BatchInputRole[];
  missing_roles: string[];
  configuration_ready: boolean;
  blocking_decisions: string[];
  /** Where the server filed this pack. Read-only — the browser never sets it. */
  source_prefix: string;
  workflow_id: string | null;
  created_at: string;
  updated_at: string;
}

export interface CreateBatchInput {
  client_id: string;
  portfolio_id: string;
  reporting_date: string;
  workflow_type: WorkflowOutcome;
  dataset: BatchDataset;
  auto_start_when_ready: boolean;
  /** Explicit assertion that this book is new. The API refuses an unknown
   *  portfolio without it, so a typo cannot open a delivery under a portfolio
   *  the client does not have. */
  new_portfolio?: boolean;
}

/**
 * A portfolio (source book) registered to a client, as the source registry
 * holds it. `portfolio_id` is the governed identifier — `direct_001` — and is
 * never free text on the way in.
 */
export interface PortfolioOption {
  portfolio_id: string;
  display_name: string;
  book_type: string;
  book_type_label: string;
  datasets: string[];
  dataset_labels: string[];
  frequencies: Record<string, string>;
  regime_required: boolean;
  outcome: WorkflowOutcome;
  outcome_label: string;
  reporting_requirement: string;
  asset_id: string;
  asset_label: string;
  statuses: string[];
  registered: boolean;
}

export interface WithdrawalReason {
  value: string;
  label: string;
}

export interface WithdrawalRecord {
  by: string;
  at: string;
  reason_code: string;
  reason_label: string;
  note: string;
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
