import { useMemo, useState } from "react";
import { ChevronRight, Layers } from "lucide-react";
import { aggregateSelection, buildDrillModel, type DrillArtifact, type DrillMeasure } from "@/lib/drill";
import { formatValue } from "@/lib/utils";

/** Format one aggregated measure value, degrading missing data to N/A. */
function measureText(value: number | null, m: DrillMeasure): string {
  if (value === null || !Number.isFinite(value)) return "N/A";
  return formatValue(value, m.format, m.scale);
}

/**
 * Dynamic drill-through for a chart/table MI result. Reuses the artifact's own
 * rows: pick a dimension value (region, broker, year, SPV, product…) and see
 * the detailed metrics already present in the payload, plus a derived share of
 * total. Missing measures show N/A rather than erroring.
 *
 * The simplest robust pattern (a "Drill into" selector) works uniformly across
 * bar/line/heatmap/treemap charts and tables — no chart-library click wiring.
 */
export function DrillThroughPanel({ artifact }: { artifact: DrillArtifact }) {
  const model = useMemo(() => buildDrillModel(artifact), [artifact]);
  const [selected, setSelected] = useState<string>("");

  if (!model) return null;

  const agg = selected ? aggregateSelection(model, selected) : null;
  const primary = model.primary;
  const primaryTotal = primary ? model.totals[primary.key] : 0;
  const primarySelected = primary && agg ? agg.values[primary.key] : null;
  const share =
    primary && primarySelected !== null && primaryTotal
      ? `${((primarySelected / primaryTotal) * 100).toFixed(1)}%`
      : null;

  return (
    <div className="mt-3 rounded-lg border border-[var(--color-line-soft)] bg-navy-900/40 p-3">
      <div className="flex flex-wrap items-center gap-2">
        <Layers size={13} className="text-peri-300" />
        <span className="text-[11px] font-medium uppercase tracking-wider text-ink-400">
          Drill into {model.dimensionLabel}
        </span>
        <select
          aria-label={`Drill into ${model.dimensionLabel}`}
          value={selected}
          onChange={(e) => setSelected(e.target.value)}
          className="ml-auto rounded-md border border-[var(--color-line)] bg-navy-950/60 px-2 py-1 text-[11px] text-ink-200 focus:border-peri-400/50 focus:outline-none"
        >
          <option value="">Select a value…</option>
          {model.values.map((v) => (
            <option key={v} value={v}>
              {v}
            </option>
          ))}
        </select>
      </div>

      {agg && (
        <div className="mt-3">
          <div className="flex items-center gap-1.5 text-[12px] font-semibold text-ink-100">
            <ChevronRight size={13} className="text-peri-300" />
            {selected}
            {agg.records > 1 && (
              <span className="text-[11px] font-normal text-ink-500">· {agg.records} rows</span>
            )}
          </div>
          <div className="mt-2 grid grid-cols-2 gap-2 sm:grid-cols-3">
            {model.measures.map((m) => (
              <Metric key={m.key} label={m.label} value={measureText(agg.values[m.key], m)} />
            ))}
            {share && primary && (
              <Metric label={`Share of ${primary.label}`} value={share} accent />
            )}
          </div>
        </div>
      )}
    </div>
  );
}

function Metric({ label, value, accent }: { label: string; value: string; accent?: boolean }) {
  return (
    <div className="rounded-md border border-[var(--color-line-soft)] bg-navy-850/60 px-2.5 py-2">
      <div className="truncate text-[10px] font-medium uppercase tracking-wider text-ink-500" title={label}>
        {label}
      </div>
      <div className={`mt-0.5 font-mono text-sm tabular-nums ${accent ? "text-peri-200" : "text-ink-100"}`}>
        {value}
      </div>
    </div>
  );
}
