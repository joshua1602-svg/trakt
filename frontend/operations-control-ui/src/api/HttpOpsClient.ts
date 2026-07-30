import { copy } from "@/lib/copy";
import { announceUnauthorized, clearToken, getToken } from "@/lib/token";
import { OpsError, type OpsClient } from "./OpsClient";
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

interface ErrorFields {
  errorCode?: string;
  message?: string;
  blockers?: { kind: string; message: string }[];
}

type ApiEnvelope = { ok?: boolean; detail?: ErrorFields | string } & ErrorFields &
  Record<string, unknown>;

function query(params: Record<string, string | undefined>): string {
  const search = new URLSearchParams();
  for (const [key, value] of Object.entries(params)) {
    if (value) search.set(key, value);
  }
  const s = search.toString();
  return s ? `?${s}` : "";
}

export class HttpOpsClient implements OpsClient {
  private readonly baseUrl: string;

  constructor(baseUrl?: string) {
    this.baseUrl = baseUrl ?? import.meta.env.VITE_OPS_API_URL ?? "";
  }

  private async request<T>(path: string, init?: RequestInit): Promise<T> {
    let response: Response;
    // A multipart body carries its own boundary — setting Content-Type here
    // would corrupt it, so the browser is left to set it.
    const isForm = typeof FormData !== "undefined" && init?.body instanceof FormData;
    try {
      response = await fetch(`${this.baseUrl}${path}`, {
        ...init,
        headers: {
          ...(isForm ? {} : { "Content-Type": "application/json" }),
          "X-Operator-Token": getToken() ?? "",
          ...(init?.headers ?? {}),
        },
      });
    } catch {
      throw new OpsError(copy.errors.network);
    }

    if (response.status === 401) {
      clearToken();
      announceUnauthorized();
      throw new OpsError(copy.errors.signedOut);
    }

    let body: ApiEnvelope;
    try {
      body = await response.json();
    } catch {
      throw new OpsError(copy.errors.generic);
    }

    if (!response.ok || body.ok === false) {
      // FastAPI wraps `HTTPException(detail={...})` in a `detail` envelope;
      // the OpsError handler returns the same fields at the top level.
      const detail = typeof body.detail === "object" && body.detail !== null ? body.detail : body;
      throw new OpsError(
        detail.message || copy.errors.generic,
        detail.errorCode,
        Array.isArray(detail.blockers) ? detail.blockers : [],
      );
    }
    return body as T;
  }

  private post<T>(path: string, payload: unknown): Promise<T> {
    return this.request<T>(path, { method: "POST", body: JSON.stringify(payload) });
  }

  async health(): Promise<void> {
    await this.request<{ ok: boolean }>("/health");
  }

  async getDashboard(): Promise<Dashboard> {
    const body = await this.request<{ ok: true } & Dashboard>("/ops/dashboard");
    return body;
  }

  async getClients(): Promise<string[]> {
    const body = await this.request<{ clients: string[] }>("/ops/clients");
    return body.clients;
  }

  async getDeliveries(client?: string): Promise<Delivery[]> {
    const body = await this.request<{ deliveries: Delivery[] }>(
      `/ops/deliveries${query({ client })}`,
    );
    return body.deliveries;
  }

  async registerDelivery(input: RegisterDeliveryInput): Promise<Delivery> {
    const body = await this.post<{ delivery: Delivery }>("/ops/deliveries", input);
    return body.delivery;
  }

  async createBatch(input: CreateBatchInput): Promise<Batch> {
    const body = await this.post<{ batch: Batch }>("/ops/batches", input);
    return body.batch;
  }

  async listBatches(client?: string): Promise<Batch[]> {
    const body = await this.request<{ batches: Batch[] }>(`/ops/batches${query({ client })}`);
    return body.batches;
  }

  async getBatch(batchId: string, client?: string): Promise<Batch> {
    const body = await this.request<{ batch: Batch }>(
      `/ops/batches/${encodeURIComponent(batchId)}${query({ client })}`,
    );
    return body.batch;
  }

  /** Multipart upload. The destination is the server's decision, not ours. */
  async uploadBatchFiles(batchId: string, files: File[], client?: string): Promise<Batch> {
    const form = new FormData();
    for (const file of files) form.append("files", file, file.name);
    // No Content-Type header: the browser sets the multipart boundary itself.
    const body = await this.request<{ batch: Batch }>(
      `/ops/batches/${encodeURIComponent(batchId)}/upload${query({ client })}`,
      { method: "POST", body: form },
    );
    return body.batch;
  }

  async startBatch(batchId: string, client?: string): Promise<Batch> {
    const body = await this.post<{ batch: Batch }>(
      `/ops/batches/${encodeURIComponent(batchId)}/start${query({ client })}`,
      {},
    );
    return body.batch;
  }

  async startWorkflow(input: StartWorkflowInput): Promise<Workflow> {
    const body = await this.post<{ workflow: Workflow }>("/ops/workflows", input);
    return body.workflow;
  }

  async getWorkflows(params?: { client?: string; status?: string }): Promise<WorkflowRow[]> {
    const body = await this.request<{ workflows: WorkflowRow[] }>(
      `/ops/workflows${query({ client: params?.client, status: params?.status })}`,
    );
    return body.workflows;
  }

  async getWorkflow(workflowId: string): Promise<Workflow> {
    const body = await this.request<{ workflow: Workflow }>(
      `/ops/workflows/${encodeURIComponent(workflowId)}`,
    );
    return body.workflow;
  }

  async rerunWorkflow(workflowId: string): Promise<Workflow> {
    const body = await this.post<{ workflow: Workflow }>(
      `/ops/workflows/${encodeURIComponent(workflowId)}/rerun`,
      {},
    );
    return body.workflow;
  }

  async cancelWorkflow(workflowId: string, reason: string): Promise<Workflow> {
    const body = await this.post<{ workflow: Workflow }>(
      `/ops/workflows/${encodeURIComponent(workflowId)}/cancel`,
      { reason },
    );
    return body.workflow;
  }

  async publishWorkflow(workflowId: string, rememberScope?: string): Promise<Publication> {
    const body = await this.post<{ publication: Publication }>(
      `/ops/workflows/${encodeURIComponent(workflowId)}/publish`,
      { remember_scope: rememberScope ?? "delivery" },
    );
    return body.publication;
  }

  async holdWorkflow(workflowId: string, reason: string): Promise<Publication> {
    const body = await this.post<{ publication: Publication }>(
      `/ops/workflows/${encodeURIComponent(workflowId)}/hold`,
      { reason },
    );
    return body.publication;
  }

  async getReviews(params?: { client?: string; workflow_id?: string }): Promise<Review[]> {
    const body = await this.request<{ reviews: Review[] }>(
      `/ops/reviews${query({ client: params?.client, workflow_id: params?.workflow_id })}`,
    );
    return body.reviews;
  }

  async getReview(decisionId: string): Promise<Review> {
    const body = await this.request<{ review: Review }>(
      `/ops/reviews/${encodeURIComponent(decisionId)}`,
    );
    return body.review;
  }

  async submitDecision(decisionId: string, input: DecisionInput): Promise<DecisionResult> {
    const body = await this.post<{ ok: true } & DecisionResult>(
      `/ops/reviews/${encodeURIComponent(decisionId)}/decision`,
      input,
    );
    return body;
  }

  async getRules(params?: RulesQuery): Promise<Rule[]> {
    const body = await this.request<{ rules: Rule[] }>(
      `/ops/rules${query({
        client: params?.client,
        q: params?.q,
        kind: params?.kind,
        scope: params?.scope,
      })}`,
    );
    return body.rules;
  }

  async getRuleHistory(ruleId: string): Promise<Rule[]> {
    const body = await this.request<{ history: Rule[] }>(
      `/ops/rules/${encodeURIComponent(ruleId)}/history`,
    );
    return body.history;
  }

  async getHistory(client?: string): Promise<Publication[]> {
    const body = await this.request<{ history: Publication[] }>(`/ops/history${query({ client })}`);
    return body.history;
  }

  async getPrincipal(): Promise<Principal> {
    const body = await this.request<{ principal: Principal }>("/ops/me");
    return body.principal;
  }

  async getConfigOverview(): Promise<ConfigOverview> {
    return this.request<{ ok: true } & ConfigOverview>("/ops/admin/config");
  }

  async getConfigCatalogue(): Promise<ConfigCatalogue> {
    return this.request<{ ok: true } & ConfigCatalogue>("/ops/admin/config/catalogue");
  }

  async getConfigVersion(
    layer: ConfigLayer,
    version: number,
    file?: string,
  ): Promise<ConfigVersion> {
    const body = await this.request<{ version: ConfigVersion }>(
      `/ops/admin/config/${layer}/${version}${query({ file })}`,
    );
    return body.version;
  }

  async compareConfigVersions(
    layer: ConfigLayer,
    from: number,
    to: number,
  ): Promise<Comparison> {
    const body = await this.request<{ comparison: Comparison }>(
      `/ops/admin/config/${layer}/compare?from_version=${from}&to_version=${to}`,
    );
    return body.comparison;
  }

  async getConfigImpact(layer: ConfigLayer, version?: number): Promise<ImpactAnalysis> {
    const body = await this.request<{ impact: ImpactAnalysis }>(
      `/ops/admin/config/${layer}/impact${query({ version: version?.toString() })}`,
    );
    return body.impact;
  }

  async getConfigAudit(): Promise<AuditTrail> {
    return this.request<{ ok: true } & AuditTrail>("/ops/admin/config/audit");
  }

  async createConfigDraft(
    layer: ConfigLayer,
    input?: CreateDraftInput,
  ): Promise<{ version: number; status: string }> {
    return this.post<{ version: number; status: string }>(
      `/ops/admin/config/${layer}/draft`,
      {
        from_version: input?.from_version ?? null,
        edits: input?.edits ?? {},
        notes: input?.notes ?? "",
      },
    );
  }

  async validateConfigVersion(
    layer: ConfigLayer,
    version: number,
  ): Promise<{ validation: ValidationResult; validation_summary: ValidationSummary }> {
    return this.post(`/ops/admin/config/${layer}/${version}/validate`, {});
  }

  async activateConfigVersion(
    layer: ConfigLayer,
    version: number,
  ): Promise<{ active_version: number }> {
    return this.post(`/ops/admin/config/${layer}/${version}/activate`, {});
  }

  async rollbackConfig(
    layer: ConfigLayer,
    toVersion: number,
  ): Promise<{ active_version: number }> {
    return this.post(`/ops/admin/config/${layer}/rollback`, { to_version: toVersion });
  }

  // -- Client Onboarding ---------------------------------------------------- //

  async getOnboardingReference(): Promise<OnboardingReference> {
    return this.request<OnboardingReference>("/ops/onboarding/reference");
  }

  async getOnboardingHome(): Promise<OnboardingHome> {
    const body = await this.request<{ home: OnboardingHome }>("/ops/onboarding/home");
    return body.home;
  }

  private async caseCall(path: string, payload?: unknown): Promise<OnboardingCase> {
    const body = await this.post<{ case: OnboardingCase }>(path, payload ?? {});
    return body.case;
  }

  async startNewClientCase(): Promise<OnboardingCase> {
    return this.caseCall("/ops/onboarding/cases");
  }

  async startMigrationCase(clientId: string): Promise<OnboardingCase> {
    return this.caseCall("/ops/onboarding/cases/migration", { client_id: clientId });
  }

  async startAmendmentCase(clientId: string): Promise<OnboardingCase> {
    return this.caseCall("/ops/onboarding/cases/amendment", { client_id: clientId });
  }

  async getCase(caseId: string): Promise<OnboardingCase> {
    const body = await this.request<{ case: OnboardingCase }>(
      `/ops/onboarding/cases/${encodeURIComponent(caseId)}`,
    );
    return body.case;
  }

  async saveCaseStep(
    caseId: string,
    step: string,
    payload: Record<string, unknown>,
  ): Promise<OnboardingCase> {
    const body = await this.request<{ case: OnboardingCase }>(
      `/ops/onboarding/cases/${encodeURIComponent(caseId)}`,
      { method: "PUT", body: JSON.stringify({ step, payload }) },
    );
    return body.case;
  }

  async addPipelineBook(caseId: string, portfolioId: string): Promise<OnboardingCase> {
    return this.caseCall(`/ops/onboarding/cases/${encodeURIComponent(caseId)}/pipeline-book`, {
      portfolio_id: portfolioId,
    });
  }

  async removeSource(
    caseId: string,
    portfolioId: string,
    dataset: string,
  ): Promise<OnboardingCase> {
    const body = await this.request<{ case: OnboardingCase }>(
      `/ops/onboarding/cases/${encodeURIComponent(caseId)}/sources/` +
        `${encodeURIComponent(portfolioId)}/${encodeURIComponent(dataset)}`,
      { method: "DELETE" },
    );
    return body.case;
  }

  async getCaseChecklist(caseId: string): Promise<ChecklistRow[]> {
    const body = await this.request<{ checklist: ChecklistRow[] }>(
      `/ops/onboarding/cases/${encodeURIComponent(caseId)}/checklist`,
    );
    return body.checklist;
  }

  async createInformationRequest(
    caseId: string,
    items: ChecklistRow[],
    options?: { responsible_party?: string; due_date?: string; note?: string },
  ): Promise<OnboardingCase> {
    return this.caseCall(`/ops/onboarding/cases/${encodeURIComponent(caseId)}/requests`, {
      items,
      responsible_party: options?.responsible_party ?? "client",
      due_date: options?.due_date ?? "",
      note: options?.note ?? "",
    });
  }

  async markRequestSent(caseId: string, requestId: string): Promise<OnboardingCase> {
    return this.caseCall(
      `/ops/onboarding/cases/${encodeURIComponent(caseId)}/requests/` +
        `${encodeURIComponent(requestId)}/sent`,
    );
  }

  async recordRequestResponse(
    caseId: string,
    requestId: string,
    body: {
      note?: string;
      answers?: Record<string, unknown>;
      evidence?: { name: string; reference: string }[];
    },
  ): Promise<OnboardingCase> {
    return this.caseCall(
      `/ops/onboarding/cases/${encodeURIComponent(caseId)}/requests/` +
        `${encodeURIComponent(requestId)}/response`,
      body,
    );
  }

  async addCaseQuestion(caseId: string, question: string): Promise<OnboardingCase> {
    return this.caseCall(`/ops/onboarding/cases/${encodeURIComponent(caseId)}/questions`, {
      question,
    });
  }

  async resolveCaseQuestion(
    caseId: string,
    questionId: string,
    resolution: string,
  ): Promise<OnboardingCase> {
    return this.caseCall(
      `/ops/onboarding/cases/${encodeURIComponent(caseId)}/questions/` +
        `${encodeURIComponent(questionId)}/resolve`,
      { resolution },
    );
  }

  async getCasePreview(caseId: string): Promise<CasePreview> {
    const body = await this.request<{ preview: CasePreview }>(
      `/ops/onboarding/cases/${encodeURIComponent(caseId)}/preview`,
    );
    return body.preview;
  }

  async submitCase(caseId: string): Promise<OnboardingCase> {
    return this.caseCall(`/ops/onboarding/cases/${encodeURIComponent(caseId)}/submit`);
  }

  async approveCase(caseId: string, reason: string): Promise<OnboardingCase> {
    return this.caseCall(`/ops/onboarding/cases/${encodeURIComponent(caseId)}/approve`, {
      reason,
    });
  }

  async activateCase(caseId: string): Promise<ActivationResult> {
    return this.post<ActivationResult>(
      `/ops/onboarding/cases/${encodeURIComponent(caseId)}/activate`,
      {},
    );
  }

  async withdrawCase(caseId: string, reason: string): Promise<OnboardingCase> {
    return this.caseCall(`/ops/onboarding/cases/${encodeURIComponent(caseId)}/withdraw`, {
      reason,
    });
  }

  async getOnboardingClient(clientId: string): Promise<OnboardingClientDetail> {
    const body = await this.request<{ client: OnboardingClientDetail }>(
      `/ops/onboarding/clients/${encodeURIComponent(clientId)}`,
    );
    return body.client;
  }

  async getOnboardingClientVersion(
    clientId: string,
    version: number,
  ): Promise<ConfigurationVersionRow> {
    const body = await this.request<{ version: ConfigurationVersionRow }>(
      `/ops/onboarding/clients/${encodeURIComponent(clientId)}/versions/${version}`,
    );
    return body.version;
  }
}
