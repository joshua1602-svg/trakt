import type { DemoColumn, DemoRow } from "@/types/demo";

/**
 * Display formatting for values the Trakt engine produced.
 *
 * This module formats; it never computes. Every number it receives has already
 * been calculated by the deterministic engine, and the format/scale contract
 * comes from the engine's own column metadata.
 */

const gbp = new Intl.NumberFormat("en-GB", {
  style: "currency",
  currency: "GBP",
  maximumFractionDigits: 0,
});

const gbpCompact = new Intl.NumberFormat("en-GB", {
  style: "currency",
  currency: "GBP",
  notation: "compact",
  maximumFractionDigits: 1,
});

const plain = new Intl.NumberFormat("en-GB", { maximumFractionDigits: 1 });

export function formatValue(
  value: DemoRow[string],
  format: string | undefined,
  options: { compact?: boolean } = {},
): string {
  if (value === null || value === undefined || value === "") return "—";
  if (typeof value === "boolean") return value ? "Yes" : "No";

  if (typeof value === "number") {
    switch (format) {
      case "gbp":
        return options.compact ? gbpCompact.format(value) : gbp.format(value);
      case "pct":
        return `${plain.format(value)}%`;
      case "number":
        return plain.format(value);
      default:
        return plain.format(value);
    }
  }

  return String(value);
}

export function columnFormat(columns: DemoColumn[], key: string): string {
  return columns.find((c) => c.key === key)?.format ?? "text";
}

export function columnLabel(columns: DemoColumn[], key: string): string {
  return (
    columns.find((c) => c.key === key)?.label ??
    key.replace(/_/g, " ").replace(/\b\w/g, (m) => m.toUpperCase())
  );
}

