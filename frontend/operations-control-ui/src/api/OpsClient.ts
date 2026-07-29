import type {
  Batch,
  CreateBatchInput,
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

  constructor(message: string, errorCode?: string) {
    super(message);
    this.name = "OpsError";
    this.errorCode = errorCode;
  }
}

export interface OpsClient {
  health(): Promise<void>;
  getDashboard(): Promise<Dashboard>;
  getClients(): Promise<string[]>;
  getDeliveries(client?: string): Promise<Delivery[]>;
  registerDelivery(input: RegisterDeliveryInput): Promise<Delivery>;
  createBatch(input: CreateBatchInput): Promise<Batch>;
  listBatches(client?: string): Promise<Batch[]>;
  getBatch(batchId: string, client?: string): Promise<Batch>;
  registerBatchFile(batchId: string, path: string, client?: string): Promise<Batch>;
  startBatch(batchId: string, client?: string): Promise<Batch>;
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
}
