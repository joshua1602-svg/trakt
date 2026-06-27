import { useEffect, useMemo, useState } from "react";
import {
  CartesianGrid, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip,
  XAxis, YAxis, Legend,
} from "recharts";
import { TrendingUp } from "lucide-react";
import type { AgentClient } from "@/api";
import type {
  ForecastExtrapolation, RunRateForecast, KfiConversionForecast,
} from "@/domain";
import { Card } from "@/components/ui";
import { cn, formatGBP } from "@/lib/utils";

type ModelKey = "weighted" | "run_rate" | "kfi";

function gbpC(v: number): string {
  return formatGBP(v, { compact: true });
}

const MODELS: { key: ModelKey; label: string }[] = [
  { key: "weighted", label: "Current weighted pipeline" },
  { key: "run_rate", label: "Completion run-rate" },
  { key: "kfi", label: "KFI run-rate × conversion" },
];

function activeModel(data: ForecastExtrapolation, key: ModelKey):
  RunRateForecast | KfiConversionForecast | null {
  if (key === "run_rate") return data.completionRunRateForecast;
  if (key === "kfi") return data.kfiConversionForecast;
  return null;
}

/**
 * Scale-up extrapolation: model selector, projected funded-balance curve with
 * downside/base/upside bands + threshold markers, milestone table, and an
 * assumptions panel. Distinct from (and complementary to) the point-in-time
 * weighted-pipeline bridge.
 */
export function ForecastExtrapolationPanel({
  client, portfolioId,
}: {
  client: AgentClient;
  portfolioId: string;
}) {
  const [data, setData] = useState<ForecastExtrapolation | null>(null);
  const [model, setModel] = useState<ModelKey>("run_rate");
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    client
      .getForecastExtrapolation(portfolioId)
      .then((d) => { if (!cancelled) setData(d); })
      .catch(() => { /* keep prior */ })
      .finally(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; };
  }, [client, portfolioId]);

  const current = data ? activeModel(data, model) : null;
  const curve = useMemo(
    () => (current?.projectedBalances ?? []).map((p) => ({
      month: p.month, downside: p.downside, base: p.base, upside: p.upside,
    })),
    [current],
  );

  if (loading && !data) {
    return <p className="text-[12px] text-ink-500" data-testid="forecast-extrapolation-loading">Loading scale-up forecast…</p>;
  }
  if (!data) return null;

  const weighted = data.currentWeightedPipelineForecast;
  const insufficient = !current || current.available === false;

  return (
    <section className="space-y-4" data-testid="forecast-extrapolation-panel">
      <div className="flex items-center gap-2 text-sm font-semibold text-ink-100">
        <TrendingUp size={16} className="text-peri-300" /> Scale-up forecast — when does the book reach scale?
      </div>

      {/* Model selector */}
      <div role="tablist" aria-label="Forecast model"
        className="inline-flex flex-wrap items-center gap-1 rounded-lg border border-[var(--color-line)] bg-navy-900/60 p-1">
        {MODELS.map((m) => (
          <button key={m.key} type="button" role="tab" aria-selected={model === m.key}
            onClick={() => setModel(m.key)}
            className={cn(
              "rounded-md px-3 py-1 text-[12px] font-medium transition-colors",
              model === m.key ? "bg-navy-700/80 text-ink-100" : "text-ink-400 hover:text-ink-200",
            )}>
            {m.label}
          </button>
        ))}
      </div>

      {model === "weighted" && (
        <Card className="p-4" testId="weighted-pipeline-forecast">
          <div className="text-[12px] font-semibold text-ink-200">{weighted.label}</div>
          <p className="mt-1 text-[11px] text-amber-300/80">{weighted.note}</p>
          <div className="mt-3 grid grid-cols-3 gap-3 text-[12px]">
            <div><div className="text-ink-500">Current funded balance</div><div className="text-ink-100">{gbpC(weighted.fundedBalance)}</div></div>
            <div><div className="text-ink-500">Weighted expected pipeline</div><div className="text-ink-100">{weighted.weightedExpectedPipeline != null ? gbpC(weighted.weightedExpectedPipeline) : "—"}</div></div>
            <div><div className="text-ink-500">Point-in-time forecast</div><div className="text-ink-100">{weighted.forecastFundedBalance != null ? gbpC(weighted.forecastFundedBalance) : "—"}</div></div>
          </div>
        </Card>
      )}

      {model !== "weighted" && insufficient && (
        <Card className="p-4" testId="forecast-insufficient">
          <div className="text-[13px] font-medium text-amber-300">Insufficient history for this model</div>
          <p className="mt-1 text-[12px] text-ink-400">
            {(current && "caveat" in current && current.caveat) ||
              "Not enough governed history to extrapolate. Showing the available signal only."}
          </p>
        </Card>
      )}

      {model !== "weighted" && current && current.available && (
        <>
          {/* Scale-up curve */}
          <Card className="p-4">
            <div className="mb-2 text-[12px] font-semibold text-ink-200">
              Projected funded balance — downside / base / upside
            </div>
            <div style={{ width: "100%", height: 240 }}>
              <ResponsiveContainer>
                <LineChart data={curve} margin={{ top: 6, right: 16, bottom: 4, left: 6 }}>
                  <CartesianGrid stroke="#23304d" strokeDasharray="3 3" />
                  <XAxis dataKey="month" tick={{ fill: "#8a97ad", fontSize: 10 }} minTickGap={24} />
                  <YAxis tickFormatter={gbpC} tick={{ fill: "#8a97ad", fontSize: 10 }} width={60} />
                  <Tooltip formatter={(v: number) => gbpC(Number(v))}
                    contentStyle={{ background: "#0f1626", border: "1px solid #23304d", fontSize: 12 }} />
                  <Legend wrapperStyle={{ fontSize: 11 }} />
                  {data.thresholds.map((thr) => (
                    <ReferenceLine key={thr} y={thr} stroke="#3a4a66" strokeDasharray="2 4"
                      label={{ value: `£${thr / 1_000_000}m`, fill: "#6b7890", fontSize: 9, position: "right" }} />
                  ))}
                  <Line type="monotone" dataKey="downside" name="Downside" stroke="#eb6f6f" strokeWidth={2} dot={false} />
                  <Line type="monotone" dataKey="base" name="Base" stroke="#7c9cf0" strokeWidth={2} dot={false} />
                  <Line type="monotone" dataKey="upside" name="Upside" stroke="#5ec6b8" strokeWidth={2} dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>
            <p className="mt-1 text-[10px] text-ink-500">
              Scenario bands are indicative ranges, not statistical confidence intervals.
            </p>
          </Card>

          {/* Milestone table */}
          <Card className="p-4" testId="milestone-table">
            <div className="mb-2 text-[12px] font-semibold text-ink-200">Milestone dates to funding thresholds</div>
            <div className="grid grid-cols-4 gap-2 px-1 pb-1 text-[10px] uppercase tracking-wider text-ink-500">
              <span>Threshold</span><span>Downside</span><span>Base</span><span>Upside</span>
            </div>
            {(current.milestones ?? []).map((m) => (
              <div key={m.threshold}
                className="grid grid-cols-4 gap-2 border-b border-[var(--color-line-soft)] px-1 py-1.5 text-[12px] last:border-0">
                <span className="text-ink-200">{m.thresholdLabel}</span>
                <span className="text-ink-300">{m.reached ? "Reached" : (m.downsideDate ?? "—")}</span>
                <span className="text-ink-100">{m.reached ? "Reached" : (m.baseDate ?? "—")}</span>
                <span className="text-ink-300">{m.reached ? "Reached" : (m.upsideDate ?? "—")}</span>
              </div>
            ))}
          </Card>

          {/* Assumptions panel */}
          <Card className="p-4" testId="assumptions-panel">
            <div className="mb-2 text-[12px] font-semibold text-ink-200">Assumptions &amp; caveats</div>
            <dl className="grid grid-cols-2 gap-x-4 gap-y-1 text-[11px] sm:grid-cols-3">
              {"baseMonthlyRunRate" in current && current.baseMonthlyRunRate != null && (
                <div><dt className="text-ink-500">Base monthly run-rate</dt><dd className="text-ink-300">{gbpC(current.baseMonthlyRunRate)}</dd></div>
              )}
              {"annualisedRunRate" in current && current.annualisedRunRate != null && (
                <div><dt className="text-ink-500">Annualised run-rate</dt><dd className="text-ink-300">{gbpC(current.annualisedRunRate)}</dd></div>
              )}
              {"conversionRate" in current && current.conversionRate != null && (
                <div><dt className="text-ink-500">KFI→completion rate</dt><dd className="text-ink-300">{(current.conversionRate * 100).toFixed(1)}%</dd></div>
              )}
              {"lagMonths" in current && current.lagMonths != null && (
                <div><dt className="text-ink-500">Lag (months)</dt><dd className="text-ink-300">{current.lagMonths}</dd></div>
              )}
              {"observedMonths" in current && current.observedMonths != null && (
                <div><dt className="text-ink-500">Observed months</dt><dd className="text-ink-300">{current.observedMonths}</dd></div>
              )}
              {"observedWeeks" in current && current.observedWeeks != null && (
                <div><dt className="text-ink-500">Observed weeks</dt><dd className="text-ink-300">{current.observedWeeks}</dd></div>
              )}
              {"scenarioBasis" in current && current.scenarioBasis && (
                <div className="col-span-2 sm:col-span-3"><dt className="text-ink-500">Scenario basis</dt><dd className="text-ink-300">{current.scenarioBasis}</dd></div>
              )}
            </dl>
            {(current.caveats ?? []).length > 0 && (
              <ul className="mt-2 space-y-1 text-[11px] text-amber-300/80">
                {current.caveats!.map((c, i) => <li key={i}>• {c}</li>)}
              </ul>
            )}
          </Card>
        </>
      )}
    </section>
  );
}
