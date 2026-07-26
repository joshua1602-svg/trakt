/**
 * Lightweight localStorage persistence for the workspace. Versioned so schema
 * changes invalidate cleanly rather than crashing on stale shapes.
 */

import type { Artifact, ChatMessage } from "@/domain";

const KEY = "trakt.mi-agent.workspace.v3";

export interface PersistedState {
  /** Selected funded portfolio (client) id. */
  clientId?: string;
  /** Selected reporting run id. */
  runId?: string;
  messages: ChatMessage[];
  artifacts: Artifact[];
}

/**
 * Context handed in via URL query parameters — used by Copilot's "Open in
 * Workspace" deep links to RESTORE context (client, run, question, filters)
 * rather than land on a blank dashboard. Empty object when no params are
 * present, so the existing localStorage flow is untouched for normal visits.
 */
export interface UrlContext {
  clientId?: string;
  runId?: string;
  question?: string;
  /** Opaque filter payload carried on the link (JSON string as sent). */
  filters?: string;
}

/** Parse the Workspace-restoring params off the current URL. Safe on SSR. */
export function readUrlContext(): UrlContext {
  if (typeof window === "undefined" || !window.location?.search) return {};
  try {
    const p = new URLSearchParams(window.location.search);
    const ctx: UrlContext = {};
    const client = p.get("client");
    const run = p.get("run");
    const q = p.get("q");
    const filters = p.get("filters");
    if (client) ctx.clientId = client;
    if (run) ctx.runId = run;
    if (q) ctx.question = q;
    if (filters) ctx.filters = filters;
    return ctx;
  } catch {
    return {};
  }
}

export function loadState(): PersistedState | null {
  if (typeof localStorage === "undefined") return null;
  try {
    const raw = localStorage.getItem(KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as PersistedState;
    if (!Array.isArray(parsed.messages)) return null;
    // Drop any in-flight pending messages from a previous session.
    parsed.messages = parsed.messages.filter((m) => !m.pending);
    return parsed;
  } catch {
    return null;
  }
}

export function saveState(state: PersistedState): void {
  if (typeof localStorage === "undefined") return;
  try {
    localStorage.setItem(KEY, JSON.stringify(state));
  } catch {
    /* quota / private mode — non-fatal */
  }
}

export function clearState(): void {
  if (typeof localStorage === "undefined") return;
  try {
    localStorage.removeItem(KEY);
  } catch {
    /* non-fatal */
  }
}
