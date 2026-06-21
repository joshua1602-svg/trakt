import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

/** Tailwind-aware className combiner. */
export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

/** Format a number as GBP, compacting large magnitudes (e.g. £124.6MM). */
export function formatGBP(value: number, opts?: { compact?: boolean }): string {
  if (opts?.compact ?? true) {
    const abs = Math.abs(value);
    if (abs >= 1e9) return `£${(value / 1e9).toFixed(2)}BN`;
    if (abs >= 1e6) return `£${(value / 1e6).toFixed(1)}MM`;
    if (abs >= 1e3) return `£${(value / 1e3).toFixed(0)}K`;
  }
  return new Intl.NumberFormat("en-GB", {
    style: "currency",
    currency: "GBP",
    maximumFractionDigits: 0,
  }).format(value);
}

export function formatPct(value: number, dp = 1): string {
  return `${value >= 0 ? "" : ""}${value.toFixed(dp)}%`;
}

export function formatSignedPct(value: number, dp = 1): string {
  return `${value >= 0 ? "+" : ""}${value.toFixed(dp)}%`;
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

/**
 * Format a value by a domain ValueFormat tag.
 *
 * `scale` is a display multiplier applied to percentages only, so a value stored
 * as a 0..1 fraction (LTV 0.51) renders as `51.0%`; plotting keeps the raw value.
 */
export function formatValue(
  value: string | number,
  format?: "gbp" | "pct" | "number" | "decimal" | "text" | "date",
  scale = 1,
): string {
  if (typeof value !== "number") {
    return format === "date" && value ? formatDate(String(value)) : String(value);
  }
  switch (format) {
    case "gbp":
      return formatGBP(value);
    case "pct":
      return `${(value * (scale || 1)).toFixed(1)}%`;
    case "decimal":
      return value.toFixed(2);
    case "number":
      return value.toLocaleString("en-GB");
    default:
      return value.toLocaleString("en-GB");
  }
}
