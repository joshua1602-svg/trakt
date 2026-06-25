import { useState } from "react";
import { ChevronDown, Info } from "lucide-react";
import type { ViewLineage } from "@/domain";
import { cn } from "@/lib/utils";

/**
 * Compact "How calculated" panel for a view: the headline source/metric line is
 * always visible; the full lineage (dates, probability basis, formula) is behind
 * an expandable technical-details toggle so the main card stays clean.
 */
export function LineagePanel({ lineage }: { lineage: ViewLineage | null | undefined }) {
  const [open, setOpen] = useState(false);
  if (!lineage) return null;

  const headline =
    lineage.view === "forecast"
      ? lineage.formula
      : `Source: ${lineage.source ?? "—"} · Metric: ${lineage.metric ?? "—"}`;

  const rows: [string, string | null | undefined][] = [
    ["Source", lineage.source],
    ["Metric", lineage.metric],
    ["Weighted metric", lineage.weightedMetric],
    ["Forecast basis", lineage.formula],
    ["Funded reporting date", lineage.fundedReportingDate],
    ["Pipeline as-of", lineage.pipelineAsOfDate],
    ["Probability basis", lineage.completionProbabilityBasis],
  ];

  return (
    <div className="rounded-lg border border-[var(--color-line-soft)] bg-navy-900/40 px-3.5 py-2">
      <div className="flex items-center justify-between gap-2">
        <div className="flex min-w-0 items-center gap-2 text-[11px] text-ink-400">
          <Info size={13} className="shrink-0 text-peri-300" />
          <span className="truncate" title={headline ?? undefined}>{headline}</span>
        </div>
        <button
          type="button"
          onClick={() => setOpen((s) => !s)}
          className="inline-flex shrink-0 items-center gap-1 text-[11px] font-medium text-ink-500 hover:text-ink-300"
        >
          How calculated
          <ChevronDown size={13} className={cn("transition-transform", !open && "-rotate-90")} />
        </button>
      </div>
      {open && (
        <dl className="mt-2 grid grid-cols-1 gap-x-6 gap-y-1 border-t border-[var(--color-line-soft)] pt-2 text-[11px] sm:grid-cols-2">
          {rows
            .filter(([, v]) => v != null && v !== "")
            .map(([k, v]) => (
              <div key={k} className="flex justify-between gap-2">
                <dt className="text-ink-500">{k}</dt>
                <dd className="text-right font-medium text-ink-300">{v}</dd>
              </div>
            ))}
          {lineage.explanation && (
            <p className="col-span-full mt-1 text-ink-500">{lineage.explanation}</p>
          )}
        </dl>
      )}
    </div>
  );
}
