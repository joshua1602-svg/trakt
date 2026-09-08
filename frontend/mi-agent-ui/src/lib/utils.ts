import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

/** Tailwind-aware className combiner. */
export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

/**
 * The reporting currency in force, as told to us by the API.
 *
 * The browser does NOT decide the economic currency. The governed client
 * configuration owns it (`portfolio.base_currency`), the API resolves it once
 * per request and states it in the response envelope, and this module holds the
 * answer so the money formatters agree with the server-formatted KPI strings.
 * The GBP default only applies until the first envelope arrives.
 */
let displayCurrency = "GBP";

/** Adopt the reporting currency an API response declared. Display-only. */
export function setDisplayCurrency(code?: string | null): void {
  if (typeof code === "string" && /^[A-Z]{3}$/.test(code.trim().toUpperCase())) {
    displayCurrency = code.trim().toUpperCase();
  }
}

export function getDisplayCurrency(): string {
  return displayCurrency;
}

/** Symbol for the reporting currency; falls back to the code when unknown. */
function currencySymbol(): string {
  const symbols: Record<string, string> = {
    GBP: "£", EUR: "€", USD: "$", JPY: "¥", CHF: "CHF ",
    AUD: "A$", CAD: "C$", NZD: "NZ$", SEK: "kr ", NOK: "kr ", DKK: "kr ",
  };
  return symbols[displayCurrency] ?? `${displayCurrency} `;
}

/**
 * Format a monetary amount in the reporting currency, compacting large
 * magnitudes (e.g. £124.6MM). Named for its original GBP-only behaviour, which
 * it still produces for a GBP book.
 */
export function formatGBP(value: number, opts?: { compact?: boolean }): string {
  const symbol = currencySymbol();
  if (opts?.compact ?? true) {
    const abs = Math.abs(value);
    if (abs >= 1e9) return `${symbol}${(value / 1e9).toFixed(2)}BN`;
    if (abs >= 1e6) return `${symbol}${(value / 1e6).toFixed(1)}MM`;
    if (abs >= 1e3) return `${symbol}${(value / 1e3).toFixed(0)}K`;
  }
  return new Intl.NumberFormat("en-GB", {
    style: "currency",
    currency: displayCurrency,
    maximumFractionDigits: 0,
  }).format(value);
}

/** Preferred name for new call sites; `formatGBP` is the historical alias. */
export const formatMoney = formatGBP;

export function formatPct(value: number, dp = 1): string {
  return `${value >= 0 ? "" : ""}${value.toFixed(dp)}%`;
}

export function formatSignedPct(value: number, dp = 1): string {
  return `${value >= 0 ? "+" : ""}${value.toFixed(dp)}%`;
}

/** Domain acronyms that must stay fully capitalised in generated UI titles. */
const TITLE_ACRONYMS: Record<string, string> = {
  ltv: "LTV",
  wa: "WA",
  nneg: "NNEG",
  abs: "ABS",
  spv: "SPV",
  uk: "UK",
  id: "ID",
  irr: "IRR",
  moic: "MOIC",
  dscr: "DSCR",
  cpr: "CPR",
  rag: "RAG",
  kpi: "KPI",
  ifrs9: "IFRS9",
  esma: "ESMA",
  nuts: "NUTS",
};

/**
 * Polish a raw measure/dimension key into a presentation title.
 * `average_ltv by region by age_bucket` → `Average LTV By Region By Age Bucket`.
 * Snake_case becomes spaced Title Case; known acronyms stay capitalised.
 */
export function formatUiTitle(input?: string): string {
  if (!input) return "";
  return input
    .replace(/_/g, " ")
    .replace(/\s+/g, " ")
    .trim()
    .split(" ")
    .map((word) => {
      const lower = word.toLowerCase();
      return TITLE_ACRONYMS[lower] ?? lower.charAt(0).toUpperCase() + lower.slice(1);
    })
    .join(" ");
}

/**
 * Polish a heading/label that MIGHT already be human-written prose. Only
 * transforms strings that still look like a raw key (contain an underscore),
 * leaving curated titles like "Pipeline Bridge to £100MM" untouched.
 */
export function formatHeading(input?: string): string {
  if (!input) return "";
  return input.includes("_") ? formatUiTitle(input) : input;
}

/**
 * Slugify a title into a download-filename stem (snake_case, ascii-safe).
 * `Average LTV By Region` → `average_ltv_by_region`.
 */
export function toFilenameStem(input?: string): string {
  if (!input) return "export";
  const stem = input
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "_")
    .replace(/^_+|_+$/g, "");
  return stem || "export";
}

/** Short, deterministic id for mock records. */
export function uid(prefix = "id"): string {
  return `${prefix}_${Math.random().toString(36).slice(2, 9)}`;
}

export function formatTime(iso: string): string {
  return new Date(iso).toLocaleTimeString("en-GB", {
    hour: "2-digit",
    minute: "2-digit",
  });
}

export function formatDate(iso: string): string {
  return new Date(iso).toLocaleDateString("en-GB", {
    day: "2-digit",
    month: "short",
    year: "numeric",
  });
}

/** Storage scale of a percent value, from the API dataset contract. */
export type PercentScale = "percent_fraction" | "percent_points" | null | undefined;

/**
 * Convert a stored percent to display points using the contract scale. A
 * fraction (0.51) becomes 51; points (51) stay 51. Internal values are never
 * mutated — this is display-only.
 */
export function toPercentPoints(value: number, scale: PercentScale): number {
  return scale === "percent_fraction" ? value * 100 : value;
}

/**
 * Normalise a percent of UNKNOWN storage scale to a fraction (0–1), used when
 * the dataset contract did NOT tag the scale. Heuristic: a magnitude above 1.5
 * is read as whole percentage points (56 → 0.56); otherwise it is already a
 * fraction (0.56 → 0.56). This is the fallback for the contract-aware path and
 * fixes the "0.6% for a 56% LTV" bug when no scale is supplied.
 */
export function normalisePercentValue(value: unknown): number | null {
  const n = Number(value);
  if (!Number.isFinite(n)) return null;
  if (Math.abs(n) > 1.5) return n / 100;
  return n;
}

/**
 * Format a percent of UNKNOWN scale for display. 0.56 and 56 both render as
 * "56.0%"; non-numeric input renders as "N/A". When the dataset contract scale
 * IS known, prefer `formatValue(v, "pct", scale)` / `toPercentPoints`.
 */
export function formatPercent(value: unknown, decimals = 1): string {
  const n = normalisePercentValue(value);
  if (n === null) return "N/A";
  return `${(n * 100).toFixed(decimals)}%`;
}

/**
 * Format a value by a domain ValueFormat tag, honouring the percent storage
 * scale from the dataset contract (so 0.51 displays as 51.0%, not 0.5%).
 */
export function formatValue(
  value: string | number,
  format?: "gbp" | "pct" | "number" | "decimal" | "text" | "date",
  scale?: PercentScale,
): string {
  if (typeof value !== "number") {
    return format === "date" && value ? formatDate(String(value)) : String(value);
  }
  switch (format) {
    case "gbp":
      return formatGBP(value);
    case "pct":
      // Contract scale is authoritative; without it, fall back to the heuristic
      // so a fraction (0.56) doesn't render as 0.6%.
      return scale === "percent_fraction" || scale === "percent_points"
        ? `${toPercentPoints(value, scale).toFixed(1)}%`
        : formatPercent(value, 1);
    case "decimal":
      return value.toFixed(2);
    case "number":
      return value.toLocaleString("en-GB");
    default:
      return value.toLocaleString("en-GB");
  }
}

/**
 * Single source of truth for chart AXIS tick labels (and the tooltips beside
 * them) — every chart renderer must call this rather than hand-rolling its
 * own `£${v}` or `${v}%`. Chart data reaches the frontend under two different
 * conventions and this is the one place that reconciles them: the live API
 * sends a raw absolute value (4_000_000) with no unit, which this compacts
 * (formatGBP's own default); the demo dataset instead pre-scales rows to
 * millions and tags them with an explicit `unit` ("MM"), which is already
 * exactly what should appear on the axis and is left alone. Without this
 * distinction, a raw absolute value renders as "£4000000" (the reported
 * defect) or a pre-scaled demo value gets compacted a second time.
 */
export function formatAxisValue(
  value: number,
  format?: "gbp" | "pct" | "number" | "decimal" | "text" | "date",
  opts?: { unit?: string; scale?: PercentScale },
): string {
  if (typeof value !== "number") return String(value);
  switch (format) {
    case "gbp":
      return opts?.unit ? `£${value}${opts.unit}` : formatGBP(value, { compact: true });
    case "pct":
      return opts?.scale === "percent_fraction" || opts?.scale === "percent_points"
        ? `${toPercentPoints(value, opts.scale).toFixed(1)}%`
        : formatPercent(value, 1);
    case "decimal":
      return value.toFixed(2);
    case "number":
    default:
      return value.toLocaleString("en-GB");
  }
}
