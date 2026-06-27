/**
 * Typed-predicate formatting for the Query Logic / audit panel.
 *
 * Backend filters are STRUCTURED objects ({op, value}) or plain category strings —
 * never display strings. Rendering one directly produced "[object Object]". This
 * formatter turns a predicate into a human label, e.g.
 *   current_loan_to_value: {op:"gt", value:40}  ->  "Current LTV > 40"
 *   broker_channel: {op:"not_in", value:[...]}   ->  "Broker Channel not in [Alpha, …]"
 */
import { formatUiTitle } from "@/lib/utils";

const OP_SYMBOL: Record<string, string> = {
  gt: ">", ">": ">", more_than: ">", over: ">", above: ">",
  ge: "≥", ">=": "≥", at_least: "≥",
  lt: "<", "<": "<", less_than: "<", under: "<", below: "<",
  le: "≤", "<=": "≤", at_most: "≤",
  eq: "=", "=": "=", is: "=",
  ne: "≠", "!=": "≠",
  between: "between", in: "in", not_in: "not in", "not in": "not in",
};

function fmtScalar(v: unknown): string {
  if (typeof v === "number") return v.toLocaleString("en-GB");
  return String(v);
}

/** Human-readable predicate for one filter entry. Never returns "[object Object]". */
export function formatPredicate(field: string, value: unknown): string {
  const label = formatUiTitle(field);
  if (value && typeof value === "object" && !Array.isArray(value)) {
    const v = value as { op?: string; value?: unknown };
    const op = OP_SYMBOL[String(v.op ?? "=").toLowerCase()] ?? String(v.op ?? "=");
    if (op === "between" && Array.isArray(v.value)) {
      return `${label} between ${fmtScalar(v.value[0])} and ${fmtScalar(v.value[1])}`;
    }
    if ((op === "in" || op === "not in") && Array.isArray(v.value)) {
      const vals = v.value as unknown[];
      const head = vals.slice(0, 3).map(fmtScalar).join(", ");
      const more = vals.length > 3 ? `, +${vals.length - 3} more` : "";
      return `${label} ${op} [${head}${more}]`;
    }
    return `${label} ${op} ${fmtScalar(v.value)}`;
  }
  if (Array.isArray(value)) {
    return `${label} in [${(value as unknown[]).map(fmtScalar).join(", ")}]`;
  }
  return `${label} = ${fmtScalar(value)}`;
}

/** Join all filters in a spec into one readable string. */
export function formatFilters(filters?: Record<string, unknown> | null): string {
  if (!filters) return "";
  return Object.entries(filters).map(([k, v]) => formatPredicate(k, v)).join(" · ");
}
