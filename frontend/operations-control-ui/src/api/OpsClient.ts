import type {
  AgentActivation,
  AgentAudit,
  AgentClassification,
  AgentClientForm,
  AgentChecklist,
  AgentMeta,
  AgentPack,
  AgentPreview,
  AgentReadinessPackage,
  AgentReviewPackage,
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

  // -- OCC Agent (practice onboarding cases) -------------------------------- //
  // A practice case is a REAL onboarding case, opened by the calls above and
  // stored in an isolated container, plus a practice execution beside it. None
  // of these can reach a live workflow or activate a configuration, and the
  // backend refuses them all when the feature flag is off — hiding the tab is
  // convenience, not the control.
  getAgentMeta(): Promise<AgentMeta>;
  listAgentCases(state?: string): Promise<CaseSummary[]>;
  createAgentCase(instruction: string, fixtureId?: string): Promise<AgentStatus>;
  getAgentCase(caseRef: string): Promise<AgentStatus>;
  instructAgent(caseRef: string, text: string, confirm?: boolean): Promise<AgentTurn>;
  answerAgentDecision(
    caseRef: string,
    input: { decision_id: string; action: string; value?: string; reason?: string },
  ): Promise<AgentStatus>;
  /** Named lifecycle steps, for the operator controls beside the conversation. */
  runAgentStep(
    caseRef: string,
    step:
      | "information-requests"
      | "submit"
      | "approve"
      | "request-changes"
      | "run"
      | "plan"
      | "readiness/approve"
      | "review"
      | "cancel",
    body?: Record<string, unknown>,
  ): Promise<AgentStatus>;
  /** Answer one wizard step directly, through Client Onboarding itself. */
  saveAgentStep(
    caseRef: string,
    step: string,
    payload: Record<string, unknown>,
  ): Promise<AgentStatus>;
  recordAgentClientResponse(
    caseRef: string,
    input: { request_id: string; answers: Record<string, unknown>; note?: string; accept?: boolean },
  ): Promise<AgentStatus>;
  /** Name which delivery the practice run is for. */
  setAgentRunTarget(
    caseRef: string,
    input: { portfolio_id?: string; dataset?: string; reporting_period?: string },
  ): Promise<AgentStatus>;
  uploadAgentArtefacts(caseRef: string, files: File[]): Promise<AgentStatus>;
  /** Generate a client response from the delivery outcome the case implies. */
  generateAgentResponse(caseRef: string): Promise<AgentStatus>;
  loadAgentFixtureArtefacts(caseRef: string, fixtureId: string): Promise<AgentStatus>;
  runAgentScenario(fixtureId: string): Promise<AgentStatus>;
  getAgentChecklist(caseRef: string): Promise<AgentChecklist>;
  /** What activation WOULD create. It creates nothing. */
  getAgentPreview(caseRef: string): Promise<AgentPreview>;
  getAgentReadiness(caseRef: string): Promise<AgentReadinessPackage>;
  getAgentAudit(caseRef: string): Promise<AgentAudit>;

  // -- the client pack ----------------------------------------------------- //
  /** The pack, plus the document a human would read and send by hand. */
  getAgentPack(caseRef: string): Promise<AgentPack>;
  /** The structured form this client should see now: only what they can answer. */
  getAgentClientForm(caseRef: string): Promise<AgentClientForm>;
  /** Persist a structured response, verbatim. Keys are catalogue keys. */
  submitAgentClientForm(
    caseRef: string,
    answers: Record<string, unknown>,
    options?: { request_id?: string; strict?: boolean },
  ): Promise<AgentStatus>;
  /** Every catalogue field, in one of the five categories, and why. */
  getAgentClassification(caseRef: string): Promise<AgentClassification>;
  draftAgentPack(caseRef: string): Promise<AgentStatus>;
  /** A human approves the pack for issue. The agent cannot do this. */
  approveAgentPack(caseRef: string, reason?: string): Promise<AgentStatus>;
  /** Record the pack as issued. The receipt says whether anything was sent. */
  sendAgentPack(caseRef: string, to?: string[]): Promise<AgentStatus>;

  // -- review, approval and the confirmation gate --------------------------- //
  /** Assemble the review package and submit the case for review. */
  requestAgentReview(caseRef: string): Promise<AgentStatus>;
  getAgentReview(caseRef: string): Promise<AgentReviewPackage>;
  /** Approve the configuration. This starts nothing. */
  approveAgentActivation(caseRef: string, reason?: string): Promise<AgentStatus>;
  /** What confirming would do, and every reason it currently may not. */
  getAgentActivation(caseRef: string): Promise<AgentActivation>;
  /** The one call that can reach production, through the server's one gate. */
  confirmAgentActivation(caseRef: string, confirmation: string): Promise<AgentStatus>;
}
