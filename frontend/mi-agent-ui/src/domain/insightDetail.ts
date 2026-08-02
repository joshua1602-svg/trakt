/**
 * Phase 2A — the optional week-on-week movement detail.
 *
 * Mirrors the payload built by `mi_agent_api/movement_detail.py`. Deliberately
 * small: it explains ONE point on ONE weekly chart. It is not an insight model,
 * and it carries no case-level rows.
 *
 * Every field is optional-safe on read: the detail layer is additive, so a UI
 * that receives an older or degraded payload must render less, never break.
 */

/** The movement decompositions the backend can build. */
export const DETAIL_PIPELINE = "PIPELINE_WEEKLY_MOVEMENT";
export const DETAIL_COMPLETIONS = "COMPLETIONS_WEEKLY_MOVEMENT";

export type MovementDetailType =
  | typeof DETAIL_PIPELINE
  | typeof DETAIL_COMPLETIONS;

/** One ranked contributor to the CHANGE — a delta, not a current balance. */
export interface MovementContributor {
  name: string;
  amount: number;
  /** Share of the total movement. Null when nothing moved (never Infinity). */
  share_of_change_pct: number | null;
  case_count: number;
}

/** How much of the movement each component accounts for. */
export interface MovementComponent {
  amount: number;
  cases: number;
}

/**
 * The component keys, in the order they are presented.
 *
 * These are never merged: `progressed_out` (a case that left the ACTIVE
 * pipeline without leaving the extract) is a different event from `removed`
 * (a case that left the extract entirely), and showing them as one would be a
 * different metric.
 */
export const MOVEMENT_COMPONENTS = [
  "new", "increased", "decreased", "progressed_out", "removed", "unchanged",
] as const;

export type MovementComponentKey = typeof MOVEMENT_COMPONENTS[number];

export const COMPONENT_LABEL: Record<MovementComponentKey, string> = {
  new: "New cases",
  increased: "Increased value",
  decreased: "Decreased value",
  progressed_out: "Left active pipeline",
  removed: "Removed cases",
  unchanged: "Unchanged",
};

export interface MovementMethodology {
  metric_definition?: string;
  /** "net" — both movements are the difference between two stock levels. */
  movement_basis?: string;
  /** The single deterministic attribution convention, stated by the backend. */
  attribution?: string;
  version?: string;
  /** How many cases changed each dimension — i.e. how load-bearing the
   * convention actually was for this week. */
  dimension_reassignments?: Record<string, number>;
  unmatched_current?: { cases: number; amount: number };
  unmatched_comparison?: { cases: number; amount: number };
  duplicate_case_identifiers?: { current: number; comparison: number };
}

export interface MovementDetail {
  detail_type: MovementDetailType;
  portfolio_id: string;
  scope: string;
  run_id?: string | null;
  as_of_date: string | null;
  comparison_date: string | null;
  available: boolean;
  /** Present only when `available` is false. */
  reason?: string;
  headline_metric: {
    label: string;
    value: number;
    change: number;
    change_pct: number | null;
  } | null;
  counts: { current: number; comparison: number; change: number } | null;
  contributors: {
    brokers?: MovementContributor[];
    regions?: MovementContributor[];
  };
  components: Record<MovementComponentKey, MovementComponent> | null;
  methodology?: MovementMethodology;
  source_dates?: {
    pipeline_as_of?: string | null;
    pipeline_comparison?: string | null;
    /** Always null on a weekly pipeline detail — a weekly movement must never
     * imply that monthly funded actuals were refreshed. */
    funded_as_of?: string | null;
    forecast_generated_at?: string | null;
  };
  sources?: { current?: string | null; comparison?: string | null };
  portfolioScope?: Record<string, unknown>;
}
