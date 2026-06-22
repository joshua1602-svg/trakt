/**
 * MI catalogue — STATIC FALLBACK only.
 *
 * The authoritative catalogue is served at runtime by `GET /mi/catalogue`
 * (mi_agent_api), which projects the real `mi_semantics_field_registry.yaml`
 * (43 dimensions, 37 measures) and the MIQuerySpec enums. The keys below are a
 * registry-accurate subset used for offline/demo mode and the portfolio
 * selector; prefer the live catalogue when the backend is reachable.
 */

import type { DimensionDef, MeasureDef } from "@/domain";

// NOTE: portfolios and reporting dates are NO LONGER hardcoded here. They are
// discovered at runtime from real onboarding output via `GET /mi/snapshots`
// (see `domain/snapshot.ts` and `useWorkspace`), so the dropdowns only ever show
// portfolios / reporting runs that actually exist. The dimension / measure
// catalogue below remains a registry-accurate static fallback for offline mode.

// Keys below are verified against mi_semantics_field_registry.yaml.
export const DIMENSIONS: DimensionDef[] = [
  { key: "geographic_region_obligor", label: "Region (Obligor)", bucketed: false, group: "geography" },
  { key: "geographic_region_collateral", label: "Region (Collateral)", bucketed: false, group: "geography" },
  { key: "broker_channel", label: "Broker", bucketed: false, group: "channel" },
  { key: "origination_channel", label: "Origination Channel", bucketed: false, group: "channel" },
  { key: "erm_product_type", label: "Product Type", bucketed: false, group: "product" },
  { key: "amortisation_type", label: "Amortisation Type", bucketed: false, group: "product" },
  { key: "interest_rate_type", label: "Rate Type", bucketed: false, group: "product" },
  { key: "account_status", label: "Account Status", bucketed: false, group: "performance" },
  { key: "pipeline_stage", label: "Pipeline Stage", bucketed: false, group: "performance" },
  { key: "internal_risk_grade", label: "Risk Grade", bucketed: false, group: "risk" },
  { key: "ifrs9_stage", label: "IFRS 9 Stage", bucketed: false, group: "risk" },
  { key: "ltv_bucket", label: "LTV Band", bucketed: true, group: "risk" },
  { key: "age_bucket", label: "Borrower Age Band", bucketed: true, group: "structure" },
  { key: "pd_bucket", label: "PD Band", bucketed: true, group: "risk" },
  { key: "ead_bucket", label: "EAD Band", bucketed: true, group: "risk" },
  { key: "ticket_bucket", label: "Ticket Size Band", bucketed: true, group: "structure" },
  { key: "term_bucket", label: "Term Band", bucketed: true, group: "performance" },
  { key: "vintage_year", label: "Vintage", bucketed: false, group: "vintage" },
  { key: "portfolio_id", label: "Portfolio", bucketed: false, group: "segmentation" },
  { key: "spv_id", label: "SPV", bucketed: false, group: "segmentation" },
];

export const MEASURES: MeasureDef[] = [
  { key: "current_outstanding_balance", label: "Balance", format: "gbp", defaultAggregation: "sum" },
  { key: "current_principal_balance", label: "Principal Balance", format: "gbp", defaultAggregation: "sum" },
  { key: "current_valuation_amount", label: "Valuation", format: "gbp", defaultAggregation: "sum" },
  { key: "forecast_funded_balance", label: "Forecast Funded", format: "gbp", defaultAggregation: "sum" },
  { key: "current_loan_to_value", label: "Current LTV", format: "pct", defaultAggregation: "weighted_avg" },
  { key: "current_interest_rate", label: "Coupon", format: "pct", defaultAggregation: "weighted_avg" },
  { key: "youngest_borrower_age", label: "Borrower Age", format: "number", defaultAggregation: "avg" },
  { key: "arrears_balance", label: "Arrears", format: "gbp", defaultAggregation: "sum" },
  { key: "default_amount", label: "Default Amount", format: "gbp", defaultAggregation: "sum" },
  { key: "redemptions_received_in_period", label: "Redemptions", format: "gbp", defaultAggregation: "sum" },
];
