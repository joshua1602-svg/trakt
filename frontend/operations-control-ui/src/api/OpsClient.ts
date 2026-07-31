import type {
  AgentAudit,
  AgentMeta,
  AgentPack,
  AgentReadinessPackage,
  AgentStatus,
  AgentTurn,
  CaseSummary,
} from "./agentTypes";
import type {
  AuditTrail,
  Comparison,
  ConfigCatalogue,
  ConfigLayer,
  ConfigOverview,
  ConfigVersion,
  CreateDraftInput,
  ImpactAnalysis,
  Principal,
  ValidationResult,
  ValidationSummary,
} from "./adminTypes";
import type {
  ActivationResult,
  CasePreview,
  ChecklistRow,
  ConfigurationVersionRow,
  OnboardingCase,
  OnboardingClientDetail,
  OnboardingHome,
  OnboardingReference,
} from "./onboardingTypes";
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
  createBatch(input: CreateBatchInput): Promise<Batch>;
  listBatches(client?: string): Promise<Batch[]>;
  getBatch(batchId: string, client?: string): Promise<Batch>;
  /**
   * Send the files themselves. There is deliberately no way to name a storage
   * location from the browser — the server derives the governed destination
   * from the input pack's own client, portfolio, book and reporting period.
   */
  uploadBatchFiles(batchId: string, files: File[], client?: string): Promise<Batch>;
  startBatch(batchId: string, client?: string): Promise<Batch>;
  startWorkflow(input: StartWorkflowInput): Promise<Workflow>;
  getWorkflows(params?: { client?: string; status?: string }): Promise<WorkflowRow[]>;
  getWorkflow(workflowId: string): Promise<Workflow>;
  rerunWorkflow(workflowId: string): Promise<Workflow>;
  cancelWorkflow(workflowId: string, reason: string): Promise<Workflow>;
  publishWorkflow(workflowId: string, rememberScope?: string): Promise<Publication>;
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

  // -- Client Onboarding ---------------------------------------------------- //
  /** The governed information model the wizard renders. */
  getOnboardingReference(): Promise<OnboardingReference>;
  getOnboardingHome(): Promise<OnboardingHome>;
  /** Start a new client. Blank: no client is selected and nothing is read. */
  startNewClientCase(): Promise<OnboardingCase>;
  /** Secondary: bring an existing client's configuration into the same model. */
  startMigrationCase(clientId: string): Promise<OnboardingCase>;
  /** Change an active client, starting from the version in force. */
  startAmendmentCase(clientId: string): Promise<OnboardingCase>;
  getCase(caseId: string): Promise<OnboardingCase>;
  saveCaseStep(caseId: string, step: string, payload: Record<string, unknown>): Promise<OnboardingCase>;
  addPipelineBook(caseId: string, portfolioId: string): Promise<OnboardingCase>;
  /** Record a sample pack; turns format, file names and asset class into inferences. */
  registerSample(caseId: string, files: { name: string; headers?: string[] }[]): Promise<OnboardingCase>;
  removeSource(caseId: string, portfolioId: string, dataset: string): Promise<OnboardingCase>;
  getCaseChecklist(caseId: string): Promise<ChecklistRow[]>;
  createInformationRequest(
    caseId: string,
    items: ChecklistRow[],
    options?: { responsible_party?: string; due_date?: string; note?: string },
  ): Promise<OnboardingCase>;
  markRequestSent(caseId: string, requestId: string): Promise<OnboardingCase>;
  recordRequestResponse(
    caseId: string,
    requestId: string,
    body: { note?: string; answers?: Record<string, unknown>; evidence?: { name: string; reference: string }[] },
  ): Promise<OnboardingCase>;
  addCaseQuestion(caseId: string, question: string): Promise<OnboardingCase>;
  resolveCaseQuestion(caseId: string, questionId: string, resolution: string): Promise<OnboardingCase>;
  getCasePreview(caseId: string): Promise<CasePreview>;
  submitCase(caseId: string): Promise<OnboardingCase>;
  approveCase(caseId: string, reason: string): Promise<OnboardingCase>;
  activateCase(caseId: string): Promise<ActivationResult>;
  withdrawCase(caseId: string, reason: string): Promise<OnboardingCase>;
  getOnboardingClient(clientId: string): Promise<OnboardingClientDetail>;
  getOnboardingClientVersion(clientId: string, version: number): Promise<ConfigurationVersionRow>;

  // -- OCC Agent (synthetic onboarding cases) ------------------------------ //
  // Every one of these is served by the synthetic case store. None of them can
  // reach a live workflow, and the backend refuses them all when the feature
  // flag is off — hiding the tab is convenience, not the control.
  getAgentMeta(): Promise<AgentMeta>;
  listAgentCases(state?: string): Promise<CaseSummary[]>;
  createAgentCase(instruction: string, fixtureId?: string): Promise<AgentStatus>;
  getAgentCase(caseId: string): Promise<AgentStatus>;
  instructAgent(caseId: string, text: string, confirm?: boolean): Promise<AgentTurn>;
  answerAgentDecision(
    caseId: string,
    input: { decision_id: string; action: string; value?: string; reason?: string },
  ): Promise<AgentStatus>;
  /** Named lifecycle steps, for the operator controls beside the conversation. */
  runAgentStep(
    caseId: string,
    step:
      | "requirements/confirm"
      | "pack/generate"
      | "pack/approve"
      | "artefacts/classify"
      | "configuration/draft"
      | "configuration/approve"
      | "run"
      | "plan"
      | "readiness/approve"
      | "cancel",
  ): Promise<AgentStatus>;
  uploadAgentArtefacts(caseId: string, files: File[]): Promise<AgentStatus>;
  /** Generate a client response from the case's own confirmed requirements. */
  generateAgentResponse(caseId: string): Promise<AgentStatus>;
  loadAgentFixtureArtefacts(caseId: string, fixtureId: string): Promise<AgentStatus>;
  returnAgentCaseToStage(caseId: string, targetState: string, reason?: string): Promise<AgentStatus>;
  runAgentScenario(fixtureId: string): Promise<AgentStatus>;
  getAgentPack(caseId: string): Promise<AgentPack>;
  getAgentReadiness(caseId: string): Promise<AgentReadinessPackage>;
  getAgentAudit(caseId: string): Promise<AgentAudit>;
}
