import { copy } from "@/lib/copy";
import { announceUnauthorized, clearToken, getToken } from "@/lib/token";
import { OpsError, type OpsClient } from "./OpsClient";
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

    let body: { ok?: boolean; errorCode?: string; message?: string } & Record<string, unknown>;
    try {
      body = await response.json();
    } catch {
      throw new OpsError(copy.errors.generic);
    }

    if (!response.ok || body.ok === false) {
      throw new OpsError(body.message || copy.errors.generic, body.errorCode);
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

  async registerBatchFile(batchId: string, path: string, client?: string): Promise<Batch> {
    const body = await this.post<{ batch: Batch }>(
      `/ops/batches/${encodeURIComponent(batchId)}/files${query({ client })}`,
      { path },
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
}
