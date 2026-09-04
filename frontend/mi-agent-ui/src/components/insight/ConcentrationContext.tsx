/**
 * Phase 2A — concentration proximity, with the three governed states kept apart.
 *
 * Everything here is already on the wire from `/mi/concentration-tests`; this
 * adds NO calculation and NO endpoint. Its whole job is separation and dating:
 *
 *   Current funded position   the monthly funded actuals — dated to the FUNDED
 *                             reporting date, never to the weekly pipeline
 *   Expected forecast         the completion-trend forecast, with its horizon
 *                             and generation provenance
 *   All pipeline converts     a STRESS scenario, labelled hypothetical, shown
 *                             only because it already exists as a governed
 *                             output — never merged into the expected forecast
 *
 * The one thing this must never do is let a weekly pipeline number read as a
 * refresh of the monthly funded actuals, so every block states its own date.
 */

import type { ConcentrationTest, ForecastMethodology } from "@/domain";
import { cn } from "@/lib/utils";

// FIXED: `currentValue` / `threshold` / `utilization` / `headroom` are
// ALREADY percent numbers on the wire (7.63 means 7.63%), the same
// convention `concentrationShared.formatValue`/`formatChange` and the
// server's own `_fmt` (mi_agent_api/concentration_query.py) use for every
// other rendering of these same fields — confirmed by both agreeing on
// "%.2f%%" with no scaling. This block used to multiply by 100 on top of
// that, so a test the table row showed as "7.63%" and "15.00%" opened here
// as "763.0%" and "1500.0%" — the reported defect. `warningFraction` is the
// one genuinely fractional field in this payload (used elsewhere as
// `warningFraction * 100`); it is not touched by either helper below.
function pct(v: number | null | undefined, dp = 1): string {
  return v == null ? "—" : `${v.toFixed(dp)}%`;
}

function pp(v: number | null | undefined, dp = 1): string {
  return v == null ? "—" : `${v.toFixed(dp)}pp`;
}

const TONE: Record<string, string> = {
  pass: "text-[#5ec6b8]", warning: "text-[#e0a458]",
  breach: "text-[#eb6f6f]", unavailable: "text-ink-500",
};

function Row({ label, value, tone }: {
  label: string; value: string; tone?: string;
}) {
  return (
    <div className="flex items-baseline justify-between gap-3">
      <span className="text-[10px] text-ink-500">{label}</span>
      <span className={cn("text-[11px] tabular-nums text-ink-200", tone)}>{value}</span>
    </div>
  );
}

function Block({ title, subtitle, hypothetical, children }: {
  title: string; subtitle?: string | null; hypothetical?: boolean;
  children: React.ReactNode;
}) {
  return (
    <div className={cn(
      "rounded-md border p-2",
      hypothetical
        ? "border-dashed border-[var(--color-line-soft)] bg-navy-900/30"
        : "border-[var(--color-line-soft)] bg-navy-900/50")}>
      <div className="mb-1 flex items-baseline justify-between gap-2">
        <span className="text-[10px] font-semibold uppercase tracking-wide text-ink-300">
          {title}
        </span>
        {hypothetical && (
          <span className="rounded bg-[#e0a458]/15 px-1 text-[9px] uppercase tracking-wide text-[#e0a458]">
            Hypothetical
          </span>
        )}
      </div>
      {subtitle && <div className="mb-1 text-[9px] text-ink-500">{subtitle}</div>}
      <div className="space-y-0.5">{children}</div>
    </div>
  );
}

export function ConcentrationContext({
  test, forecast, fundedReportingDate, statesAvailable, statesReason,
}: {
  test: ConcentrationTest;
  forecast?: ForecastMethodology;
  /** The FUNDED reporting date — a monthly cut, not the weekly pipeline date. */
  fundedReportingDate?: string | null;
  statesAvailable?: boolean;
  statesReason?: string | null;
}) {
  const expected = test.expected ?? null;
  const full = test.fullPipeline ?? null;
  const horizon = test.expectedBreachHorizon ?? null;

  return (
    <div className="mt-3 space-y-2" data-testid="concentration-context">
      <div className="text-[10px] uppercase tracking-wide text-ink-500">
        Position by state
      </div>

      <Block title="Current funded position"
        subtitle={fundedReportingDate
          ? `Funded actuals as of ${fundedReportingDate}`
          : "Funded actuals — reporting date unavailable"}>
        <Row label="Actual concentration" value={pct(test.currentValue)} />
        <Row label="Contractual limit" value={pct(test.threshold)} />
        <Row label="Utilisation" value={pct(test.utilization)} />
        <Row label="Headroom" value={pp(test.headroom)} />
        <Row label="Status" value={test.status ?? "—"}
          tone={TONE[test.status ?? "unavailable"]} />
      </Block>

      {statesAvailable && expected ? (
        <Block title="Expected forecast position"
          subtitle={[
            forecast?.methodology,
            forecast?.observationWindowEnd
              && `model observed to ${forecast.observationWindowEnd}`,
          ].filter(Boolean).join(" · ") || null}>
          <Row label="Forecast concentration" value={pct(expected.value)} />
          <Row label="Forecast utilisation" value={pct(expected.utilization)} />
          <Row label="Forecast headroom" value={pp(expected.headroom)} />
          <Row label="Forecast status" value={expected.status ?? "—"}
            tone={TONE[expected.status ?? "unavailable"]} />
          <Row label="Projected breach"
            value={horizon?.available && horizon.period
              ? horizon.period
              : (horizon?.reason ? "not projected" : "—")} />
        </Block>
      ) : (
        <div className="rounded-md border border-[var(--color-line-soft)] bg-navy-900/30 p-2 text-[10px] text-ink-500"
          data-testid="concentration-context-no-forecast">
          {statesReason
            ?? "No governed forecast state is available for this test."}
        </div>
      )}

      {statesAvailable && full && (
        <Block title="All pipeline converts" hypothetical
          subtitle="Stress scenario — assumes 100% of active pipeline completes. Not a forecast.">
          <Row label="Stressed concentration" value={pct(full.value)} />
          <Row label="Stressed utilisation" value={pct(full.utilization)} />
          <Row label="Stressed headroom" value={pp(full.headroom)} />
          <Row label="Stressed status" value={full.status ?? "—"}
            tone={TONE[full.status ?? "unavailable"]} />
        </Block>
      )}
    </div>
  );
}
