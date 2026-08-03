/**
 * Investor PPTX deck discovery — mirrors `mi_agent_api.decks.list_decks`.
 * UI-safe: reporting-period labels only, never raw blob/file paths.
 */

/** One dated reporting-period deck available for download. */
export interface DeckPeriod {
  period: string;
}

/** The latest published deck pointer (the default download target). */
export interface DeckLatest {
  period?: string | null;
  generatedAt?: string | null;
}

/** Decks available for a client (the download menu's data source). */
export interface DeckIndex {
  available: boolean;
  latest: DeckLatest | null;
  decks: DeckPeriod[];
  client_id: string;
  error?: string;
}

/**
 * A requested deck generation — mirrors `mi_agent_api.deck_generation`.
 *
 * `blocked` is deliberately distinct from `failed`: the pack was built and the
 * publication gates refused to release it, so nothing was published and the
 * previously published deck is untouched. Presenting that as either a success
 * or a crash would misdescribe what happened.
 */
export type DeckGenerationState =
  | "queued" | "generating" | "completed" | "blocked" | "failed";

export interface DeckGenerationJob {
  ok: boolean;
  jobId: string;
  state: DeckGenerationState;
  portfolioId: string;
  portfolioContext?: string | null;
  reportingPeriod?: string | null;
  runId?: string | null;
  requestedAt: string;
  startedAt?: string | null;
  completedAt?: string | null;
  /** Publication gates that refused, when `state === "blocked"`. */
  failedGates: string[];
  gateCount?: number | null;
  /** Investor-safe explanation for a blocked or failed generation. */
  message?: string | null;
  errorCode?: string | null;
}

/** What the caller asks for. The tenant is never sent — it is authenticated. */
export interface DeckGenerationRequest {
  portfolioId?: string;
  portfolioContext?: string | null;
  period?: string | null;
  idempotencyKey?: string;
}
