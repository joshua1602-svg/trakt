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
    // A discrete panel, not a cell in a grid. Three things make it read that
    // way: it sits a full surface step above the panel behind it, it is lit
    // along its top edge and cast below, and the movement rail on its leading
    // edge takes a colour ONLY where there is a direction to report — so the
    // group resolves into four distinct panels the moment the eye lands.
    <div
      className={cn(
        "rounded-lg border-l-2 bg-navy-800 p-4 shadow-[var(--elev-card)]",
        deltaIntent === "positive" ? "border-l-mint-400/60"
          : deltaIntent === "negative" ? "border-l-rose-400/60"
          : "border-l-[var(--color-line-strong)]",
        dim && "opacity-60",
      )}
    >
      {/* Three separated typographic steps: tracked uppercase caption, then
          the figure at 26px mono, then the movement and its footnote. */}
      <div className="t-label">{label}</div>
      <div className="t-figure mt-[var(--gap-tight)]">{value}</div>
      {delta != null && (
        <div
          className={cn(
            "mt-[var(--gap-tight)] inline-flex items-center gap-1 text-[var(--fs-label)] font-semibold",
            deltaColour(deltaIntent),
          )}
        >
          {deltaIntent !== "neutral" && <Icon size={13} strokeWidth={2.75} />}
          {delta}
        </div>
      )}
      {hint && <div className="t-micro mt-[var(--gap-hair)]">{hint}</div>}
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
    // Tier 3 — which unit the bars are drawn in. The same `nav-unit` treatment
    // as the stage-movement Cases/Value switch, because it is the same rank of
    // control: it governs the figures in one section, not the screen.
    //
    // `aria-selected` is what the `nav-unit-item` styling keys off, and
    // `aria-pressed` is what a toggle button owes assistive technology. Both,
    // deliberately — dropping either loses the active state for one audience.
    <div role="group" aria-label={label} className="nav-unit">
      {measures.map((m) => (
        <button
          key={m}
          type="button"
          aria-pressed={active === m}
          aria-selected={active === m}
          data-testid={`${testIdPrefix}-${m}`}
          onClick={() => onChange(m)}
          className={cn("nav-unit-item", active !== m && "cursor-pointer")}
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
    return <p className="t-micro">{emptyLabel}</p>;
  }
  const max = Math.max(...data.map((d) => d.value), 1);
  const render = (v: number) =>
    format === "gbp" ? formatGBP(v)
      : format === "pct" ? `${v.toFixed(1)}%`
      : v.toLocaleString("en-GB");
  return (
    <div className="grid grid-cols-[7rem_1fr_auto] items-center gap-x-3 gap-y-2">
      {data.map((d) => {
        const cells = (
          <>
            <span className="t-meta truncate" title={d.label}>
              {d.label}
            </span>
            <div className="h-3 overflow-hidden rounded-[2px] bg-navy-950">
              <div
                className="h-full rounded-[2px] bg-peri-500"
                style={{ width: `${Math.max(2, (d.value / max) * 100)}%` }}
              />
            </div>
            {/* The figure is the point of the row; the label and any suffix
                are context, so they sit a full ink step behind it. */}
            <span className="t-num text-right text-[var(--fs-label)] font-semibold text-ink-100">
              {render(d.value)}
              {d.count != null && format === "gbp" && (
                <span className="ml-1.5 font-normal text-ink-500">· {d.count}</span>
              )}
              {d.secondary && <span className="ml-1.5 font-normal text-ink-500">{d.secondary}</span>}
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
            className="col-span-3 grid grid-cols-[7rem_1fr_auto] items-center gap-x-3 rounded-sm
                       px-1 py-0.5 text-left transition-colors hover:bg-navy-700/60"
          >
            {cells}
          </button>
        );
      })}
    </div>
  );
}
