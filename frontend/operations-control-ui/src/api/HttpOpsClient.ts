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
    try {
      response = await fetch(`${this.baseUrl}${path}`, {
        ...init,
        headers: {
          "Content-Type": "application/json",
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

  async publishWorkflow(workflowId: string): Promise<Publication> {
    const body = await this.post<{ publication: Publication }>(
      `/ops/workflows/${encodeURIComponent(workflowId)}/publish`,
      {},
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
}
