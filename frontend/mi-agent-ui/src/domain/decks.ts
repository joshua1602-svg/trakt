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
