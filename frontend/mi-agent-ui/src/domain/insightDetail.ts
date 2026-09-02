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

/* -------------------------------------------------------------------------- *
 * Sprint 2 — the governed GROSS stage-transition detail.
 *
 * Mirrors `build_stage_transition_detail` in `mi_agent_api/movement_detail.py`,
 * field for field. It is served by the SAME `/mi/insight/movement-detail`
 * route as the two NET detail types above, under its own `detailType` — the
 * extension point that route already had. There is no second endpoint, no
 * second client and no parallel payload model.
 *
 * Nothing here is recomputed in the browser. The panel that renders this
 * ORDERS, LABELS and FORMATS; every count, amount, residual and outcome is the
 * engine's. A consumer that re-derived a transition could disagree with the
 * deck rendering the same window.
 * -------------------------------------------------------------------------- */

export const DETAIL_STAGE_TRANSITION = "PIPELINE_STAGE_TRANSITION";

/**
 * Why a transition answer is not available.
 *
 * The engine decides this, not the UI: a duplicate identifier is a governed
 * refusal, and a panel that made its own availability call could show an empty
 * matrix where the engine refused — which reads as "nothing moved".
 */
export type StageTransitionReasonCode =
  | "no_prior_snapshot"
  | "missing_case_identifier"
  | "duplicate_case_identifiers"
  | "no_governed_cases";

/** One source → destination cell. Both stages are real governed stages. */
export interface StageTransitionRow {
  source_stage: string;
  destination_stage: string;
  case_count: number;
  prior_amount: number;
  latest_amount: number;
  amount_change: number;
}

/**
 * A case present only in the latest snapshot.
 *
 * There is deliberately NO `source_stage`: it never had one. A UI may label the
 * origin "New", but must never render it as `KFI → KFI` or any other real stage.
 */
export interface StageArrivalRow {
  destination_stage: string;
  case_count: number;
  latest_amount: number;
}

/** A case in both snapshots at the same stage — including one whose amount was
 * amended. An amendment does not make it a departure plus an arrival. */
export interface StageStayerRow {
  stage: string;
  case_count: number;
  prior_amount: number;
  latest_amount: number;
  amount_change: number;
}

/**
 * A case present only in the prior snapshot.
 *
 * `governed_outcome` is a canonical terminal stage ONLY where the prior extract
 * evidenced one; otherwise it is `unclassified_departure` and must be presented
 * as unresolved. The consumer never infers a reason.
 */
export interface StageDepartureRow {
  source_stage: string;
  governed_outcome: string;
  outcome_evidence: string;
  case_count: number;
  prior_amount: number;
}

/** The engine's sentinel for a departure the data cannot explain. */
export const UNCLASSIFIED_DEPARTURE = "unclassified_departure";

export interface StageEventTotal {
  case_count: number;
  prior_amount: number;
  latest_amount: number;
}

/** Opening → closing for one stage, in cases and in value, with residuals. */
export interface StageReconciliationRow {
  stage: string;
  opening_case_count: number;
  new_arrivals: number;
  transitions_in: number;
  transitions_out: number;
  departures: number;
  stayers: number;
  closing_case_count: number;
  count_reconciliation_residual: number;
  opening_amount: number;
  new_arrival_amount: number;
  transferred_in_latest_amount: number;
  transferred_out_prior_amount: number;
  departure_prior_amount: number;
  stayer_amount_change: number;
  closing_amount: number;
  amount_reconciliation_residual: number;
}

export interface StageTransitionDetail {
  detail_type: typeof DETAIL_STAGE_TRANSITION;
  portfolio_id: string;
  scope: string;
  run_id?: string | null;
  as_of_date: string | null;
  comparison_date: string | null;
  available: boolean;
  reason?: string | null;
  reason_code?: StageTransitionReasonCode | null;
  /** The governed natural key cases were matched on. */
  identifier?: string;
  measure?: string;
  stage_field?: string;
  counts: { current: number; comparison: number; change: number } | null;
  transitions: StageTransitionRow[];
  new_arrivals: StageArrivalRow[];
  stayers: StageStayerRow[];
  departures: StageDepartureRow[];
  event_totals: Record<string, StageEventTotal> | null;
  reconciliation: {
    by_stage: StageReconciliationRow[];
    count_reconciliation_residual: number;
    amount_reconciliation_residual: number;
    global: Record<string, number>;
    amount_tolerance: number;
    count_identity?: string;
    amount_identity?: string;
  } | null;
  methodology?: {
    capability_definition?: string;
    /** "gross" — the whole point; the two other detail types are "net". */
    movement_basis?: string;
    identity_basis?: string;
    identity_note?: string;
    stage_vocabulary?: string;
    terminal_stages?: string[];
    departure_outcome_basis?: string;
    version?: string;
    unmatched_current?: { cases: number; amount: number };
    unmatched_comparison?: { cases: number; amount: number };
    duplicate_case_identifiers?: { current: number; comparison: number } | null;
  };
  source_dates?: {
    pipeline_as_of?: string | null;
    pipeline_comparison?: string | null;
    funded_as_of?: string | null;
    forecast_generated_at?: string | null;
  };
  sources?: { current?: string | null; comparison?: string | null };
  portfolioScope?: Record<string, unknown>;
}
