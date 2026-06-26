/**
 * Agent request/response envelope — mirrors `mi_agent.mi_agent_workflow
 * .run_mi_agent_query` so a future HTTP client is a drop-in for the mock.
 */

import type { Artifact } from "./artifacts";
import type { MIQuerySpec } from "./mi";

/** MI intents the agent routes to (mirrors the deterministic interpreter). */
export const INTENTS = [
  "portfolio_overview",
  "concentration_risk",
  "pipeline",
  "static_pools",
  "validation",
  "risk_monitoring",
  "scenario",
  "unknown",
] as const;
export type Intent = (typeof INTENTS)[number];

/** Portfolio selection context (mirrors PortfolioReferenceConfig). */
export interface PortfolioContext {
  id: string;
  name: string;
  entity: string;
}

/** Reporting / as-of context. */
export interface ReportingContext {
  /** ISO date. */
  asOf: string;
  /** Prior period for movement comparisons (ISO). */
  comparedTo?: string;
}

export interface AgentRequestOptions {
  parserMode?: "deterministic" | "llm";
  topN?: number;
}

/** The active MI workspace view a question runs against. */
export type WorkspaceView = "funded" | "pipeline" | "forecast";

export interface AgentRequest {
  question: string;
  portfolio: PortfolioContext;
  reporting: ReportingContext;
  options?: AgentRequestOptions;
  /** Active workspace view (funded | pipeline | forecast); explicit wording in
   * the question can override it backend-side. */
  datasetContext?: WorkspaceView;
}

export interface AgentResponse {
  ok: boolean;
  question: string;
  intent: Intent;
  /** Human-readable interpretation ("Interpreted as: …"). */
  interpreted?: string;
  narrative: string;
  assumptions: string[];
  artifacts: Artifact[];
  /** Business-facing warnings only (technical diagnostics live in `diagnostics`). */
  warnings: string[];
  /** Engineer-facing technical diagnostics, hidden behind a disclosure. */
  diagnostics?: string[];
  spec?: Partial<MIQuerySpec>;
  /** Interpreter confidence 0–1 (retained for the Query Logic panel). */
  confidence?: number;
  error?: string;
}

/* ----------------------------- Chat model ---------------------------- */

export type ChatRole = "user" | "assistant";

export interface ArtifactRef {
  id: string;
  title: string;
  type: string;
}

export interface ChatMessage {
  id: string;
  role: ChatRole;
  content: string;
  createdAt: string;
  pending?: boolean;
  error?: boolean;
  interpreted?: string;
  assumptions?: string[];
  artifactRefs?: ArtifactRef[];
  warnings?: string[];
  diagnostics?: string[];
  intent?: Intent;
  /** Parsed query spec, retained for the collapsed Query Logic panel. */
  spec?: Partial<MIQuerySpec>;
  /** Interpreter confidence 0–1, retained for the Query Logic panel. */
  confidence?: number;
}
