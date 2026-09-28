/**
 * Phase 2A — the governed evidence behind a conversion number.
 *
 * Presentation only: every field is already on the wire from
 * `/mi/evolution/funnel`. No endpoint, no calculation, no new methodology.
 *
 * Deliberately NOT here: contributor attribution. The numerator is a five-week
 * average FLOW and the denominator is a KFI STOCK lagged by the median
 * KFI→completion time. Splitting that by broker or region would need per-broker
 * lagged cohorts — a new methodology, not a new view — so this stays at the
 * governed aggregate and says so, rather than showing a plausible split that
 * nothing supports.
 *
 * A prior-period comparison rate is likewise absent from the governed payload,
 * so the movement-in-percentage-points the brief asks for is reported as
 * unavailable rather than derived from two numbers that were never intended to
 * be differenced.
 */

import type { FunnelConversion } from "@/domain";

function pct1(v: number | null | undefined): string {
  return v == null ? "—" : `${v.toFixed(1)}%`;
}

function num(v: number | null | undefined): string {
  return v == null ? "—" : v.toLocaleString("en-GB");
}

function money(v: number | null | undefined): string {
  return v == null ? "—" : `£${(v / 1e6).toFixed(1)}MM`;
}

function Row({ label, value, hint }: {
  label: string; value: string; hint?: string;
}) {
  return (
    <div className="flex items-baseline justify-between gap-3">
      <span className="text-[9px] text-ink-500">
        {label}
        {hint && <span className="text-ink-600"> · {hint}</span>}
      </span>
      <span className="text-[10px] tabular-nums text-ink-300">{value}</span>
    </div>
  );
}

export function ConversionContext({ stage, conversion, cohortPct, cohortSize, latestWeek }: {
  stage: string;
  conversion: FunnelConversion;
  /** The canonical cumulative cohort conversion for this stage. */
  cohortPct: number | null;
  /** Cases in the KFI cohort — the denominator of ``cohortPct``. */
  cohortSize?: number | null;
  /** The most recent governed weekly extract date. */
  latestWeek?: string | null;
}) {
  const lag = conversion.lagApplied && conversion.lagWeeks != null
    ? `${conversion.lagWeeks}w lag applied`
    : "no lag applied";
  // The cohort figure's own numerator: the percentage was computed from whole
  // cases over ``cohortSize``, so the count is recovered exactly by rounding.
  const reached = cohortPct != null && cohortSize
    ? Math.round((cohortPct / 100) * cohortSize)
    : null;

  return (
    <div className="mt-2 border-t border-[var(--color-line-soft)] pt-2"
      data-testid={`conversion-context-${stage}`}>
      <div className="text-[9px] uppercase tracking-wide text-ink-400">
        Governed evidence · cohort conversion
      </div>

      {/* The rows that reproduce the headline: reached ÷ cohort = %. */}
      <div className="mt-1 space-y-0.5" data-testid={`conversion-context-cohort-${stage}`}>
        <Row label="Definition"
          value="cases reaching this stage ÷ every case seen at KFI or later" />
        <Row label="Numerator" hint="cases reached, to date" value={`${num(reached)} cases`} />
        <Row label="Denominator" hint="KFI cohort" value={`${num(cohortSize)} cases`} />
        <Row label="Cumulative cohort conversion" value={pct1(cohortPct)} />
        {latestWeek && <Row label="Latest extract" value={latestWeek} />}
      </div>
      <p className="mt-1 text-[9px] leading-snug text-ink-500">
        By case count. The cohort includes recent KFIs that have not yet had
        time to convert, so the rate reads low for a young book.
      </p>

      <div className="mt-2 text-[9px] uppercase tracking-wide text-ink-400">
        Weekly velocity · forecast input, not conversion
      </div>
      <div className="mt-1 space-y-0.5" data-testid={`conversion-context-velocity-${stage}`}>
        <Row label="Basis" value={conversion.basis ?? "—"} />
        <Row label="Weekly velocity (value)" value={`${pct1(conversion.weeklyRateValue)}/wk`} />
        <Row label="Weekly velocity (count)" value={`${pct1(conversion.weeklyRateCount)}/wk`} />
        <Row label="Numerator" hint="avg weekly flow, 5wk"
          value={`${money(conversion.avgWeeklyFlowValue)} · ${num(conversion.avgWeeklyFlowCount)} cases`} />
        <Row label="Denominator" hint={lag}
          value={`${money(conversion.kfiStockValue)} · ${num(conversion.kfiStockCount)} KFIs`} />
        <Row label="Observation window"
          value={`${conversion.weeksInWindow ?? "—"} weeks (minimum ${conversion.minWeeks ?? "—"})`} />
        <Row label="Denominator week" value={conversion.denominatorWeek ?? "—"} />
      </div>

      {!conversion.sufficient && (
        <div className="mt-1 text-[9px] font-medium text-amber-300"
          data-testid={`conversion-context-provisional-${stage}`}>
          Provisional — {conversion.weeksInWindow} of {conversion.minWeeks}+ weeks observed.
        </div>
      )}

      <div className="mt-1.5 space-y-1 text-[9px] leading-snug text-ink-500">
        <div data-testid={`conversion-context-no-prior-${stage}`}>
          No prior-period comparison rate is published for this metric, so the
          movement in percentage points is not shown rather than derived.
        </div>
        <div data-testid={`conversion-context-no-attribution-${stage}`}>
          Broker and regional attribution is not available for conversion: the
          numerator is a trailing average flow and the denominator a lagged KFI
          stock, which cannot be split by dimension without a cohort methodology
          this release does not define.
        </div>
      </div>
    </div>
  );
}
