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
