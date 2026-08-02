/**
 * Phase 3A — the Weekly Portfolio Brief.
 *
 * Mirrors `mi_agent_api/insight_contract.py`. Every field is optional-safe on
 * read: the brief is additive, so a UI that receives an older or degraded
 * payload must render less, never break.
 */

export const INSIGHT_TYPES = [
  "PIPELINE_MOVEMENT", "TICKET_SIZE", "WEIGHTED_LTV", "TICKET_MIX_SHIFT",
  "LTV_MIX_SHIFT", "COMPLETIONS_MOVEMENT", "CONVERSION_CONTEXT",
  "CONCENTRATION_PROXIMITY", "DATA_QUALITY",
] as const;

export type InsightType = typeof INSIGHT_TYPES[number];

/** What the reader is told. Distinct from `priority`, which only orders. */
export type InsightSeverity = "info" | "attention" | "concern";

export interface InsightContributor {
  name: string;
  amount: number;
  share_of_change_pct: number | null;
  case_count: number;
}

export interface Insight {
  insight_id: string;
  insight_type: InsightType;
  version: string;
  tenant_id: string;
  portfolio_id: string;
  portfolio_context: string;
  run_id?: string | null;
  as_of_date: string | null;
  comparison_date: string | null;
  severity: InsightSeverity;
  /** Ordering weight, descending. Set by the selector, not the generator. */
  priority: number;
  headline: string;
  summary: string;
  metrics?: Record<string, unknown>;
  contributors?: {
    brokers?: InsightContributor[];
    regions?: InsightContributor[];
  };
  components?: Record<string, { amount: number; cases: number }>;
  methodology?: Record<string, unknown>;
  source_dates?: Record<string, string | null | undefined>;
  data_quality?: Record<string, unknown>;
  /** Which existing dashboard section this insight is about. */
  deep_link?: string | null;
  /** Whether a future channel SHOULD push this unprompted. Nothing acts on it. */
  notification_eligible?: boolean;
}

/**
 * Why an insight the brief would normally carry is absent.
 *
 * `immaterial` is the system working — a change below its threshold.
 * `unavailable` means the data or methodology is missing.
 * `capped` means it was crowded out of the eight.
 * `error` means generation failed for that type alone.
 */
export type OmissionCategory = "immaterial" | "unavailable" | "capped" | "error";

export interface InsightOmission {
  insight_type: string;
  reason: string;
  category: OmissionCategory;
}

export interface WeeklyBrief {
  brief_version: string;
  /** `partial` means a generator failed; `unavailable` means no governed source. */
  status: "success" | "partial" | "unavailable";
  reason?: string | null;
  tenant_id: string;
  portfolio_id: string;
  portfolio_context: string;
  run_id?: string | null;
  as_of_date: string | null;
  comparison_date: string | null;
  generated_from?: string;
  config_source?: string | null;
  insight_count: number;
  insights: Insight[];
  omitted: InsightOmission[];
  source_dates?: Record<string, string | null | undefined>;
  portfolioScope?: Record<string, unknown>;
}
