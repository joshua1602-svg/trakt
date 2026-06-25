import type { ForecastSnapshot } from "@/domain";
import { ForecastBridgeCard } from "@/components/ForecastBridgeCard";
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

  const byRegion: BarDatum[] = (breakdowns?.byRegionCapped ?? []).map((r) => ({
    label: r.key,
    value: r.pipelineAmount,
  }));
  const byLtv: BarDatum[] = (breakdowns?.byLtvBucketCapped ?? []).map((r) => ({
    label: r.key,
    value: r.pipelineAmount,
  }));
  const byMonth: BarDatum[] = (breakdowns?.byCompletionMonth ?? []).map((m) => ({
    label: m.month,
    value: m.weightedExpectedFundedAmount,
  }));

  return (
    <div className="space-y-4">
      <ForecastBridgeCard bridge={bridge} />
      <LineagePanel lineage={forecast?.lineage} />
      {(byRegion.length > 0 || byLtv.length > 0 || byMonth.length > 0) && (
        <section className="rounded-xl border border-[var(--color-line)] bg-navy-900/40 p-5">
          <h3 className="text-sm font-semibold text-ink-100">Forecast funded balance breakdowns</h3>
          <p className="mt-0.5 text-[11px] text-ink-400">
            Funded actual exposure + probability-weighted pipeline (derived).
          </p>
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
