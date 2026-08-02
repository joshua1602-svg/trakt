/**
 * Phase 2A — the optional enhanced chart tooltip.
 *
 * Composition, not replacement. It renders the ORIGINAL tooltip content first
 * and appends the movement detail beneath it, so the numbers the chart has
 * always shown are still the numbers on top and are unaffected by anything
 * here. When there is no detail — feature off, still loading, nothing to
 * compare against — this renders exactly the original tooltip.
 *
 * Deliberately restrained: headline, movement, two short contributor lists,
 * case count, and a hint that more is available. Anything longer belongs in the
 * drill panel, not in a tooltip that follows the pointer.
 */

import type { MovementDetail } from "@/domain";
import { ContributorList } from "./ContributorList";
import { cn, formatGBP, formatSignedPct } from "@/lib/utils";

export interface EnhancedMetricTooltipProps {
  /** The chart's own tooltip content — always rendered, always first. */
  base: React.ReactNode;
  detail: MovementDetail | null;
  loading?: boolean;
  unavailable?: boolean;
  /** Shown only when a drill affordance is actually wired up. */
  showDrillHint?: boolean;
}

function signed(v: number): string {
  return `${v >= 0 ? "+" : "−"}${formatGBP(Math.abs(v), { compact: true })}`;
}

export function EnhancedMetricTooltip({
  base, detail, loading = false, unavailable = false, showDrillHint = false,
}: EnhancedMetricTooltipProps) {
  const head = detail?.available ? detail.headline_metric : null;

  return (
    <div className="max-w-[280px] rounded-md border border-[#23304d] bg-[#0f1626] p-2 text-[12px]"
      data-testid="enhanced-metric-tooltip">
      {base}

      {loading && (
        <div className="mt-2 border-t border-[#23304d] pt-2 text-[11px] text-ink-500"
          data-testid="movement-detail-loading">
          Loading movement detail…
        </div>
      )}

      {!loading && unavailable && (
        <div className="mt-2 border-t border-[#23304d] pt-2 text-[11px] text-ink-500"
          data-testid="movement-detail-unavailable">
          {detail?.reason ?? "No movement detail for this point."}
        </div>
      )}

      {!loading && head && (
        <div className="mt-2 border-t border-[#23304d] pt-2"
          data-testid="movement-detail-body">
          <div className="flex items-baseline justify-between gap-3">
            <span className="text-[11px] text-ink-400">{head.label}</span>
            <span className="text-[12px] font-semibold tabular-nums text-ink-100">
              {formatGBP(head.value, { compact: true })}
            </span>
          </div>
          <div className={cn("text-[11px] tabular-nums",
            head.change < 0 ? "text-[#eb6f6f]" : "text-[#5ec6b8]")}>
            {signed(head.change)}
            {head.change_pct != null && ` / ${formatSignedPct(head.change_pct)}`}
            <span className="text-ink-500"> vs prior week</span>
          </div>

          <ContributorList title="Top broker contributors"
            rows={detail?.contributors?.brokers} />
          <ContributorList title="Top regional contributors"
            rows={detail?.contributors?.regions} />

          {detail?.counts && (
            <div className="mt-2 text-[11px] text-ink-400 tabular-nums">
              {detail.counts.current.toLocaleString("en-GB")} cases
              <span className="text-ink-500">
                {" "}({detail.counts.change >= 0 ? "+" : "−"}
                {Math.abs(detail.counts.change).toLocaleString("en-GB")} vs prior week)
              </span>
            </div>
          )}

          {/* Dates are always explicit: a weekly pipeline movement must never be
              read as a refresh of the monthly funded actuals. */}
          {detail?.comparison_date && (
            <div className="mt-1 text-[10px] text-ink-500">
              Pipeline {detail.as_of_date} vs {detail.comparison_date}
              {detail.methodology?.movement_basis === "net" && " · net movement"}
            </div>
          )}

          {showDrillHint && (
            <div className="mt-1 text-[10px] text-ink-400">
              Click the chart for details
            </div>
          )}
        </div>
      )}
    </div>
  );
}
