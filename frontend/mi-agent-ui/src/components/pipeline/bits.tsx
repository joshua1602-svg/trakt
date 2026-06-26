/** Small shared building blocks for the pipeline + forecast landing-page sections. */
import type { ReactNode } from "react";
import { ArrowDownRight, ArrowUpRight, Minus } from "lucide-react";
import { cn, formatGBP } from "@/lib/utils";

export type Severity = "blocker" | "warning" | "info";
export type DeltaIntent = "positive" | "negative" | "neutral";

export function severityTone(sev: Severity): "rose" | "amber" | "navy" {
  return sev === "blocker" ? "rose" : sev === "warning" ? "amber" : "navy";
}

function deltaColour(intent: DeltaIntent) {
  return intent === "positive" ? "text-mint-400" : intent === "negative" ? "text-rose-400" : "text-ink-500";
}

/** A compact KPI tile (matches the funded-snapshot tile look). */
export function StatTile({
  label,
  value,
  hint,
  dim,
  delta,
  deltaIntent = "neutral",
}: {
  label: string;
  value: string;
  hint?: ReactNode;
  dim?: boolean;
  /** Week-on-week movement, e.g. "+156 vs prior week" or "No prior week". */
  delta?: ReactNode;
  deltaIntent?: DeltaIntent;
}) {
  const Icon = deltaIntent === "positive" ? ArrowUpRight : deltaIntent === "negative" ? ArrowDownRight : Minus;
  return (
    <div className={cn("rounded-lg border border-[var(--color-line-soft)] bg-navy-850/60 p-3.5", dim && "opacity-60")}>
      <div className="text-[11px] font-medium uppercase tracking-wider text-ink-400">{label}</div>
      <div className="mt-1.5 font-mono text-2xl font-semibold tabular-nums text-ink-100">{value}</div>
      {delta != null && (
        <div className={cn("mt-1.5 inline-flex items-center gap-0.5 text-xs font-medium", deltaColour(deltaIntent))}>
          {deltaIntent !== "neutral" && <Icon size={13} strokeWidth={2.5} />}
          {delta}
        </div>
      )}
      {hint && <div className="mt-1.5 text-[11px] text-ink-500">{hint}</div>}
    </div>
  );
}

export interface BarDatum {
  label: string;
  value: number;
  /** Optional secondary value rendered to the right (e.g. weighted £). */
  secondary?: string;
  count?: number;
}

/**
 * A deterministic horizontal bar list (no chart dependency). Values are
 * formatted as GBP by default; bar widths are proportional to the max value.
 */
export function BarList({
  data,
  format = "gbp",
  emptyLabel = "No data",
}: {
  data: BarDatum[];
  format?: "gbp" | "count";
  emptyLabel?: string;
}) {
  if (data.length === 0) {
    return <p className="text-[11px] text-ink-500">{emptyLabel}</p>;
  }
  const max = Math.max(...data.map((d) => d.value), 1);
  return (
    <div className="space-y-1.5">
      {data.map((d) => (
        <div key={d.label} className="grid grid-cols-[7rem_1fr_auto] items-center gap-2">
          <span className="truncate text-[11px] text-ink-300" title={d.label}>
            {d.label}
          </span>
          <div className="h-3.5 overflow-hidden rounded-sm bg-navy-800/70">
            <div
              className="h-full rounded-sm bg-peri-400/70"
              style={{ width: `${Math.max(2, (d.value / max) * 100)}%` }}
            />
          </div>
          <span className="text-right font-mono text-[11px] tabular-nums text-ink-200">
            {format === "gbp" ? formatGBP(d.value) : d.value.toLocaleString("en-GB")}
            {d.count != null && format === "gbp" && (
              <span className="ml-1 text-ink-500">· {d.count}</span>
            )}
            {d.secondary && <span className="ml-1 text-ink-500">{d.secondary}</span>}
          </span>
        </div>
      ))}
    </div>
  );
}
