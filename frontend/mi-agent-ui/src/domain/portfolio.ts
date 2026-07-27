/**
 * Governed portfolio contract — the React mirror of `trakt_core.portfolio`.
 *
 * These types are a WIRE MIRROR, not a model. Every value in them is decided by
 * the governed backend (`GET /mi/portfolio-context`): which portfolios exist,
 * what the hierarchy is, which analyses are valid for a selection, which
 * portfolios contributed to an answer and which were excluded.
 *
 * The UI's job is presentation only. It may format a label, choose an icon or
 * decide where a disclosure sentence appears. It must NOT:
 *   - decide portfolio scope, or consolidate portfolios;
 *   - decide whether an analysis applies to a portfolio;
 *   - infer applicability from a portfolio name, id prefix or type string;
 *   - infer field availability from the presence or nullness of a value;
 *   - compute a portfolio metric or re-interpret a governed result.
 *
 * If a rule is not expressed in one of these structures, the answer is to add
 * it to the backend contract — never to reimplement it here.
 */

/** How a selectable context relates to the hierarchy. */
export type PortfolioContextKind = "total" | "type" | "portfolio";

/** Governed capability ids (mirrors `trakt_core.portfolio.CAPABILITIES`). */
export type PortfolioCapabilityId =
  | "funded"
  | "risk"
  | "evolution"
  | "cohorts"
  | "pipeline"
  | "origination_forecast"
  | "runoff_forecast"
  | "consolidated_forecast"
  | "drill_through"
  | "export";

/** Whether one analysis is valid for a scope, and exactly who contributes. */
export interface PortfolioCapability {
  capability: PortfolioCapabilityId;
  enabled: boolean;
  /** Machine-readable reason a capability is off (e.g. `NON_ORIGINATING`). */
  reason_code: string | null;
  /** Backend-authored business explanation, safe to show verbatim. */
  detail: string | null;
  contributing_portfolios: string[];
  excluded_portfolios: string[];
  /** True when only some portfolios in scope contribute to this analysis. */
  partial: boolean;
}

export type PortfolioCapabilities = Partial<
  Record<PortfolioCapabilityId, PortfolioCapability>
>;

/** One governed source portfolio and its business metadata. */
export interface PortfolioRecord {
  portfolio_id: string;
  portfolio_type: string | null;
  label: string;
  originates: boolean;
  pipeline_data_available: boolean;
  forecast_treatment: "origination" | "runoff" | "hold_constant" | string;
  runoff_profile_id: string | null;
  has_runoff_profile: boolean;
  reporting_dates: string[];
  row_count: number | null;
  present_in_data: boolean;
}

/** One node of the hierarchical selector (Total → type group → portfolio). */
export interface PortfolioContextOption {
  context_id: string;
  context_kind: PortfolioContextKind;
  label: string;
  parent_id: string | null;
  /** 0 = Total, 1 = a type group, 2 = an individual portfolio. */
  depth: number;
  portfolio_ids: string[];
  portfolio_types: string[];
  capabilities: PortfolioCapabilities;
}

/** `GET /mi/portfolio-context` — the whole governed contract in one payload. */
export interface PortfolioContextIndex {
  available: boolean;
  client_id: string | null;
  default_context_id: string | null;
  contexts: PortfolioContextOption[];
  portfolios: PortfolioRecord[];
  portfolio_types: string[];
  /** Portfolios with governed pipeline data; `null` when not discoverable. */
  pipeline_portfolios: string[] | null;
  error?: string;
}

/** The resolved scope a response was computed over. */
export interface PortfolioScope {
  context_id: string;
  context_kind: PortfolioContextKind;
  label: string;
  portfolio_ids: string[];
  portfolio_types: string[];
  /** True when the requested context was unknown and Total was substituted. */
  fell_back_to_total?: boolean;
  requested_context_id?: string | null;
}

/** Per-field availability, by portfolio, for one answer. */
export interface FieldCoverage {
  available_in: string[];
  unavailable_in: string[];
}

/**
 * The governed disclosure facts attached to a portfolio-scoped response.
 * `disclosure` is written by the backend; render it, do not re-derive it.
 */
export interface PortfolioCoverage {
  scope_requested: {
    context_id: string | null;
    context_kind: PortfolioContextKind | null;
    label: string | null;
  };
  portfolios_in_scope: string[];
  portfolios_used: string[];
  portfolios_excluded: Record<string, { reason_code?: string; detail?: string }>;
  is_fully_consolidated: boolean;
  field_coverage: Record<string, FieldCoverage>;
  disclosure: string | null;
  /** Present on responses that also carry the capability set for the scope. */
  capabilities?: PortfolioCapabilities;
  /** Present on responses that echo the resolved scope alongside coverage. */
  scope?: PortfolioScope;
}

/** Availability state of one funded chart dimension (backend-decided). */
export type ChartAvailability =
  | "available"
  | "present_but_all_null"
  | "not_supplied"
  | "partially_available";

/** Empty context index used before the first fetch resolves. */
export const EMPTY_PORTFOLIO_CONTEXT: PortfolioContextIndex = {
  available: false,
  client_id: null,
  default_context_id: null,
  contexts: [],
  portfolios: [],
  portfolio_types: [],
  pipeline_portfolios: null,
};
