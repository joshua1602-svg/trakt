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
    // Elevated tile: a step lighter than the panel behind it, with a visible
    // border + top highlight so the KPI grid reads as raised cards.
    <div
      className={cn(
        "rounded-lg border border-navy-600/70 bg-navy-800/80 p-3.5 shadow-sm",
        "shadow-[inset_0_1px_0_rgba(255,255,255,0.05)]",
        dim && "opacity-60",
      )}
    >
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
 *
 * All rows share ONE grid, so the value column resolves to a single width
 * across the whole list and every bar track is identical — bars stay
 * apples-to-apples regardless of how long each row's value text is.
 */
/** Which already-computed measure a bar list displays. Presentation only: every
 *  measure named here is a field the deterministic engine already returned in
 *  the same payload — switching never re-aggregates anything in the browser. */
export type BarMeasure = "balance" | "share" | "count";

export const BAR_MEASURE_LABEL: Record<BarMeasure, string> = {
  balance: "Balance",
  share: "% of book",
  count: "Count",
};

/** The render format each measure needs. */
export const BAR_MEASURE_FORMAT: Record<BarMeasure, "gbp" | "pct" | "count"> = {
  balance: "gbp",
  share: "pct",
  count: "count",
};

/**
 * A measure switch for bar lists. ``measures`` names only what the payload
 * actually carries, so a breakdown without a count simply never offers one.
 */
export function MeasureToggle({
  measures,
  active,
  onChange,
  label = "Measure",
  testIdPrefix,
}: {
  measures: readonly BarMeasure[];
  active: BarMeasure;
  onChange: (m: BarMeasure) => void;
  label?: string;
  testIdPrefix: string;
}) {
  if (measures.length < 2) return null;
  return (
    <div role="group" aria-label={label}
      className="inline-flex overflow-hidden rounded-md border border-navy-600/70">
      {measures.map((m) => (
        <button
          key={m}
          type="button"
          aria-pressed={active === m}
          data-testid={`${testIdPrefix}-${m}`}
          onClick={() => onChange(m)}
          className={cn(
            "px-2.5 py-1 text-[10px] font-medium transition-colors",
            active === m ? "bg-peri-400/20 text-peri-200" : "text-ink-400 hover:text-ink-200",
          )}
        >
          {BAR_MEASURE_LABEL[m]}
        </button>
      ))}
    </div>
  );
}


export function BarList({
  data,
  format = "gbp",
  emptyLabel = "No data",
  onSelect,
  selectTitle,
}: {
  data: BarDatum[];
  format?: "gbp" | "count" | "pct";
  emptyLabel?: string;
  /** Selecting a bar. Presentation-only: the handler decides what a selection
   *  means, and no measure is recomputed here. */
  onSelect?: (label: string) => void;
  selectTitle?: (label: string) => string;
}) {
  if (data.length === 0) {
    return <p className="text-[11px] text-ink-500">{emptyLabel}</p>;
  }
  const max = Math.max(...data.map((d) => d.value), 1);
  const render = (v: number) =>
    format === "gbp" ? formatGBP(v)
      : format === "pct" ? `${v.toFixed(1)}%`
      : v.toLocaleString("en-GB");
  return (
    <div className="grid grid-cols-[7rem_1fr_auto] items-center gap-x-2 gap-y-1.5">
      {data.map((d) => {
        const cells = (
          <>
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
              {render(d.value)}
              {d.count != null && format === "gbp" && (
                <span className="ml-1 text-ink-500">· {d.count}</span>
              )}
              {d.secondary && <span className="ml-1 text-ink-500">{d.secondary}</span>}
            </span>
          </>
        );
        if (!onSelect) return <div key={d.label} className="contents">{cells}</div>;
        return (
          <button
            key={d.label}
            type="button"
            onClick={() => onSelect(d.label)}
            title={selectTitle?.(d.label)}
            className="col-span-3 grid grid-cols-[7rem_1fr_auto] items-center gap-x-2 rounded-sm
                       px-1 py-0.5 text-left hover:bg-navy-700/50 focus-visible:bg-navy-700/50"
          >
            {cells}
          </button>
        );
      })}
    </div>
  );
}
