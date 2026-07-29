import type {
  Comparison,
  ConfigCatalogue,
  ConfigLayer,
  ConfigOverview,
  ConfigVersion,
  CreateDraftInput,
  ImpactAnalysis,
  AuditTrail,
  Principal,
  ValidationSummary,
  ValidationResult,
} from "./adminTypes";
import type {
  Dashboard,
  DecisionInput,
  DecisionResult,
  Delivery,
  Publication,
  RegisterDeliveryInput,
  Review,
  Rule,
  RulesQuery,
  StartWorkflowInput,
  Workflow,
  WorkflowRow,
} from "./types";

/** An error carrying a plain-English message safe to show to the operator. */
export class OpsError extends Error {
  readonly errorCode?: string;
  /** Extra operator-safe context, e.g. the blockers that refused an activation. */
  readonly blockers: { kind: string; message: string }[];

  constructor(
    message: string,
    errorCode?: string,
    blockers?: { kind: string; message: string }[],
  ) {
    super(message);
    this.name = "OpsError";
    this.errorCode = errorCode;
    this.blockers = blockers ?? [];
  }

  /** True when the backend refused because the caller is not an administrator. */
  get isForbidden(): boolean {
    return this.errorCode === "OPS_ADMIN_REQUIRED" || this.errorCode === "OPS_FORBIDDEN";
  }
}

export interface OpsClient {
  health(): Promise<void>;
  getDashboard(): Promise<Dashboard>;
  getClients(): Promise<string[]>;
  getDeliveries(client?: string): Promise<Delivery[]>;
  registerDelivery(input: RegisterDeliveryInput): Promise<Delivery>;
  startWorkflow(input: StartWorkflowInput): Promise<Workflow>;
  getWorkflows(params?: { client?: string; status?: string }): Promise<WorkflowRow[]>;
  getWorkflow(workflowId: string): Promise<Workflow>;
  rerunWorkflow(workflowId: string): Promise<Workflow>;
  cancelWorkflow(workflowId: string, reason: string): Promise<Workflow>;
  publishWorkflow(workflowId: string): Promise<Publication>;
  holdWorkflow(workflowId: string, reason: string): Promise<Publication>;
  getReviews(params?: { client?: string; workflow_id?: string }): Promise<Review[]>;
  getReview(decisionId: string): Promise<Review>;
  submitDecision(decisionId: string, input: DecisionInput): Promise<DecisionResult>;
  getRules(params?: RulesQuery): Promise<Rule[]>;
  getRuleHistory(ruleId: string): Promise<Rule[]>;
  getHistory(client?: string): Promise<Publication[]>;

  /** Who is signed in. Used only to decide what to offer — never to authorise. */
  getPrincipal(): Promise<Principal>;

  // -- administrator configuration (every call is re-authorised server-side) --
  getConfigOverview(): Promise<ConfigOverview>;
  getConfigCatalogue(): Promise<ConfigCatalogue>;
  getConfigVersion(layer: ConfigLayer, version: number, file?: string): Promise<ConfigVersion>;
  compareConfigVersions(layer: ConfigLayer, from: number, to: number): Promise<Comparison>;
  getConfigImpact(layer: ConfigLayer, version?: number): Promise<ImpactAnalysis>;
  getConfigAudit(): Promise<AuditTrail>;
  createConfigDraft(
    layer: ConfigLayer,
    input?: CreateDraftInput,
  ): Promise<{ version: number; status: string }>;
  validateConfigVersion(
    layer: ConfigLayer,
    version: number,
  ): Promise<{ validation: ValidationResult; validation_summary: ValidationSummary }>;
  activateConfigVersion(layer: ConfigLayer, version: number): Promise<{ active_version: number }>;
  rollbackConfig(layer: ConfigLayer, toVersion: number): Promise<{ active_version: number }>;
}
