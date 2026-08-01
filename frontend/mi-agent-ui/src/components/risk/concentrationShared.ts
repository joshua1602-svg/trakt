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

// ---------------------------------------------------------------------------
// Three-state presentation: risk classification chips (mapped from the
// service's emerging-risk categories — the UI never re-ranks or re-derives).
// ---------------------------------------------------------------------------
import type { ConcentrationTest, EmergingRisk } from "@/domain";

export const RISK_CHIP: Record<
  string,
  { label: string; tone: "mint" | "amber" | "rose" | "navy" | "neutral" }
> = {
  current_breach: { label: "BREACH", tone: "rose" },
  expected_breach: { label: "EXPECTED BREACH", tone: "rose" },
  expected_warning_low_headroom: { label: "LOW HEADROOM", tone: "amber" },
  material_deterioration: { label: "DETERIORATION", tone: "amber" },
  full_pipeline_only_breach: { label: "STRESS ONLY", tone: "navy" },
  data_or_methodology_limitation: { label: "LIMITED", tone: "neutral" },
  ok: { label: "OK", tone: "mint" },
};

/** testId → the service's risk category (one per test; absent = ok). */
export function riskCategoryByTest(
  risks: EmergingRisk[] | undefined,
): Map<string, string> {
  const map = new Map<string, string>();
  for (const r of risks ?? []) {
    if (r.testId && !map.has(r.testId)) map.set(r.testId, r.category);
  }
  return map;
}

/** Sort weight for "expected risk" ordering: service rank, then ascending
 *  expected headroom, then name. Rows without a risk sort last. */
export function riskSortKey(
  t: ConcentrationTest,
  categoryByTest: Map<string, string>,
): [number, number, string] {
  const rankByCategory: Record<string, number> = {
    current_breach: 1,
    expected_breach: 2,
    expected_warning_low_headroom: 3,
    material_deterioration: 4,
    full_pipeline_only_breach: 5,
    data_or_methodology_limitation: 6,
  };
  const category = categoryByTest.get(t.testId);
  const rank = category ? (rankByCategory[category] ?? 7) : 7;
  const headroom = t.expected?.headroom ?? Number.POSITIVE_INFINITY;
  return [rank, headroom, t.displayName];
}

/** One state cell's text: value + headroom/over phrase. */
export function statePhrase(
  block: { value: number | null; status: string | null; headroom: number | null } | null | undefined,
  unit: string | null | undefined,
): { value: string; sub: string } {
  if (!block) return { value: "—", sub: "" };
  if (block.status === "indicative_only") {
    return { value: "—", sub: "indicative only" };
  }
  if (block.value === null) return { value: "—", sub: "" };
  const hd = block.headroom;
  const unitSuffix = unit === "percent" ? "pp" : "";
  const sub =
    hd === null
      ? ""
      : hd < 0
        ? `over ${Math.abs(hd).toFixed(unit === "percent" ? 1 : 0)}${unitSuffix}`
        : `hd ${hd.toFixed(unit === "percent" ? 1 : 0)}${unitSuffix}`;
  return { value: formatValue(block.value, unit), sub };
}
