/** Shared presentation helpers for the Risk Limits workspace. Display only —
 *  every number comes from the governed evaluation service. */

import type { ConcentrationTestStatus } from "@/domain";

export const STATUS_TONE: Record<
  ConcentrationTestStatus,
  "mint" | "amber" | "rose" | "navy" | "neutral"
> = {
  pass: "mint",
  warning: "amber",
  breach: "rose",
  unavailable: "neutral",
  insufficient_data: "neutral",
  pending_effective_date: "navy",
  expired: "neutral",
};

export const STATUS_LABEL: Record<ConcentrationTestStatus, string> = {
  pass: "Pass",
  warning: "Warning",
  breach: "Breach",
  unavailable: "Unavailable",
  insufficient_data: "No data",
  pending_effective_date: "Pending",
  expired: "Expired",
};

/** Non-colour status cue rendered inside the badge next to the label. */
export const STATUS_GLYPH: Record<ConcentrationTestStatus, string> = {
  pass: "✓",
  warning: "!",
  breach: "✕",
  unavailable: "–",
  insufficient_data: "–",
  pending_effective_date: "…",
  expired: "–",
};

export const CATEGORY_LABEL: Record<string, string> = {
  geography: "Geography",
  property_value: "Property value",
  loan_balance: "Loan balance",
  borrower: "Borrower",
  rate_product: "Rate & product",
  ltv: "LTV",
  performance: "Performance",
  composition: "Composition",
  external_index: "External index",
  primitive: "Composed",
  // Legacy extracted categories, shown during migration.
  geographic_concentration: "Geography",
  broker_concentration: "Broker",
  large_loan_concentration: "Loan balance",
  ltv_limit: "LTV",
  interest_rate_limit: "Rate & product",
  borrower_concentration: "Borrower",
  joint_borrower_limit: "Borrower",
  age_limit: "Borrower",
  property_value_concentration: "Property value",
  other: "Other",
};

export function categoryLabel(category: string): string {
  return CATEGORY_LABEL[category] ?? category.replace(/_/g, " ");
}

export function formatValue(
  value: number | null | undefined,
  unit: string | null | undefined,
): string {
  if (value === null || value === undefined) return "—";
  if (unit === "percent") return `${value.toFixed(2)}%`;
  if (unit === "count") return `${Math.round(value)}`;
  return value.toLocaleString("en-GB", { maximumFractionDigits: 0 });
}

export function formatChange(
  value: number | null | undefined,
  unit: string | null | undefined,
): string {
  if (value === null || value === undefined) return "—";
  const sign = value > 0 ? "+" : "";
  if (unit === "percent") return `${sign}${value.toFixed(2)}pp`;
  if (unit === "count") return `${sign}${Math.round(value)}`;
  return `${sign}${value.toLocaleString("en-GB", { maximumFractionDigits: 0 })}`;
}

export function operatorGlyph(operator: string): string {
  return operator === "min" ? "≥" : "≤";
}

export function formatDate(value: string | null | undefined): string {
  return value ? value.slice(0, 10) : "—";
}
