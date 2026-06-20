/**
 * MI catalogue — dimensions, measures and portfolio context, mirroring the
 * Python semantic registry and stratification catalogue. Drives portfolio
 * selectors today and the future query-builder UI.
 */

import type { DimensionDef, MeasureDef } from "@/domain";
import type { PortfolioContext } from "@/domain";

export const PORTFOLIOS: PortfolioContext[] = [
  { id: "erm-uk-master", name: "ERM UK — Master", entity: "Trakt SPV I" },
  { id: "erm-uk-warehouse", name: "ERM UK — Warehouse", entity: "Warehouse Co" },
  { id: "erm-uk-fwd", name: "ERM UK — Forward Flow", entity: "Origination" },
];

export const REPORTING_DATES = ["2026-05-31", "2026-04-30", "2026-03-31", "2026-02-28"];

/** Prior-period lookup for movement comparisons. */
export const PRIOR_PERIOD: Record<string, string> = {
  "2026-05-31": "2026-04-30",
  "2026-04-30": "2026-03-31",
  "2026-03-31": "2026-02-28",
  "2026-02-28": "2026-01-31",
};

export const DIMENSIONS: DimensionDef[] = [
  { key: "geographic_region_obligor", label: "Region", bucketed: false, group: "geography" },
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
  { key: "youngest_borrower_age", label: "Borrower Age Band", bucketed: true, group: "structure" },
  { key: "interest_rate_bucket", label: "Interest Rate Band", bucketed: true, group: "product" },
  { key: "pd_bucket", label: "PD Band", bucketed: true, group: "risk" },
  { key: "balance_band", label: "Balance Band", bucketed: true, group: "structure" },
  { key: "time_on_book", label: "Time on Book", bucketed: true, group: "performance" },
  { key: "vintage_year", label: "Vintage", bucketed: false, group: "vintage" },
  { key: "portfolio_id", label: "Portfolio", bucketed: false, group: "segmentation" },
  { key: "spv_id", label: "SPV", bucketed: false, group: "segmentation" },
];

export const MEASURES: MeasureDef[] = [
  { key: "current_outstanding_balance", label: "Balance", format: "gbp", defaultAggregation: "sum" },
  { key: "forecast_funded_balance", label: "Forecast Funded", format: "gbp", defaultAggregation: "sum" },
  { key: "loan_count", label: "Loan Count", format: "number", defaultAggregation: "count" },
  { key: "current_loan_to_value", label: "Current LTV", format: "pct", defaultAggregation: "weighted_avg" },
  { key: "current_interest_rate", label: "Coupon", format: "pct", defaultAggregation: "weighted_avg" },
  { key: "youngest_borrower_age", label: "Borrower Age", format: "number", defaultAggregation: "weighted_avg" },
  { key: "arrears_balance", label: "Arrears", format: "gbp", defaultAggregation: "sum" },
  { key: "redemptions_received_in_period", label: "Redemptions", format: "gbp", defaultAggregation: "sum" },
  { key: "expected_nneg_loss", label: "NNEG Loss", format: "gbp", defaultAggregation: "sum" },
];

export const DEFAULT_PORTFOLIO = PORTFOLIOS[0];
export const DEFAULT_REPORTING_DATE = REPORTING_DATES[0];
