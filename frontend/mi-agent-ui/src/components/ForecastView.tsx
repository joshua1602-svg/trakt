import type { DimensionBucket, ForecastSnapshot } from "@/domain";
import { ForecastBridgeCard } from "@/components/ForecastBridgeCard";
import { TimingDisclosureBanner } from "@/components/TimingDisclosureBanner";
import { PipelineWatchlist } from "@/components/PipelineWatchlist";
import { LineagePanel } from "@/components/LineagePanel";
import { BarList, type BarDatum } from "@/components/pipeline/bits";

/**
 * Forecast view: the deterministic funded + pipeline bridge, forecast-by-dimension
 * breakdowns (funded actual + weighted pipeline, derived backend-side), and the
 * forecast watchlist. All numbers are backend-derived.
 */
export function ForecastView({
  forecast,
  loading,
}: {
  forecast: ForecastSnapshot | null;
  loading?: boolean;
}) {
  if (loading && !forecast) {
    return (
      <section className="rounded-xl border border-[var(--color-line)] bg-navy-900/40 p-5">
        <div className="h-4 w-48 animate-pulse rounded bg-navy-700/60" />
        <div className="mt-4 h-24 animate-pulse rounded-lg bg-navy-800/50" />
      </section>
    );
  }
  const bridge = forecast?.forecastBridge ?? null;
  const breakdowns = forecast?.forecastBreakdowns;

  // A forecast bar is drawn as its PARTS where the payload carries them: the
  // funded exposure that exists today, and the weighted pipeline expected to
  // arrive. Those are facts of different certainty and a funder is buying one
  // of them. Both come from the engine — nothing is derived here.
  const stacked = (r: DimensionBucket): BarDatum => {
    const funded = r.fundedAmount;
    const expected = r.weightedExpectedFundedAmount ?? 0;
    return {
      label: r.key,
      value: r.pipelineAmount,
      parts: funded == null ? undefined : [
        { label: "Current funded", value: funded, className: "bg-peri-400/70" },
        { label: "Expected additions", value: expected, className: "bg-mint-400/80" },
      ],
    };
  };
  const byRegion: BarDatum[] = (breakdowns?.byRegionCapped ?? []).map(stacked);
  const byLtv: BarDatum[] = (breakdowns?.byLtvBucketCapped ?? []).map(stacked);
  const byMonth: BarDatum[] = (breakdowns?.byCompletionMonth ?? []).map((m) => ({
    label: m.month,
    value: m.weightedExpectedFundedAmount,
  }));

  return (
    <div className="space-y-4">
      <TimingDisclosureBanner timing={forecast?.pipelineTiming} />
      <ForecastBridgeCard bridge={bridge} />
      <LineagePanel lineage={forecast?.lineage} />
      {(byRegion.length > 0 || byLtv.length > 0 || byMonth.length > 0) && (
        <section className="rounded-xl border border-[var(--color-line)] bg-navy-900/40 p-5">
          <h3 className="text-sm font-semibold text-ink-100">Forecast funded balance breakdowns</h3>
          <p className="mt-0.5 text-[11px] text-ink-400">
            Funded actual exposure + probability-weighted pipeline (derived).
          </p>
          <div className="mt-2 flex flex-wrap items-center gap-4 text-[10px] text-ink-400">
            <span className="flex items-center gap-1.5">
              <span className="inline-block h-2 w-3 rounded-sm bg-peri-400/70" />
              Current funded
            </span>
            <span className="flex items-center gap-1.5">
              <span className="inline-block h-2 w-3 rounded-sm bg-mint-400/80" />
              Expected additions
            </span>
          </div>
          <div className="mt-4 grid grid-cols-1 gap-4 lg:grid-cols-2">
            {byRegion.length > 0 && (
              <Panel title="Forecast balance by region">
                <BarList data={byRegion} format="gbp" />
              </Panel>
            )}
            {byLtv.length > 0 && (
              <Panel title="Forecast balance by LTV bucket">
                <BarList data={byLtv} format="gbp" />
              </Panel>
            )}
            {byMonth.length > 0 && (
              <Panel title="Forecast contribution by completion month">
                <BarList data={byMonth} format="gbp" />
              </Panel>
            )}
          </div>
        </section>
      )}
      <PipelineWatchlist items={forecast?.watchlist ?? []} />
    </div>
  );
}

function Panel({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="rounded-lg border border-[var(--color-line-soft)] bg-navy-900/50 p-3.5">
      <div className="mb-2.5 text-[11px] font-medium uppercase tracking-wider text-ink-400">{title}</div>
      {children}
    </div>
  );
}
