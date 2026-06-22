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
