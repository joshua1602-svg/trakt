import { useEffect, useMemo, useState } from "react";
import {
  CartesianGrid, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip,
  XAxis, YAxis, Legend,
} from "recharts";
import { TrendingUp } from "lucide-react";
import type { AgentClient } from "@/api";
import type { ForecastExtrapolation } from "@/domain";
import { Card } from "@/components/ui";
import { formatGBP } from "@/lib/utils";

function gbpC(v: number): string {
  return formatGBP(v, { compact: true });
}

/**
 * View ii — Scale-up run-rate. The forward projection of the funded book based on
 * the recent COMPLETION run-rate (net month-on-month funded growth), with
 * downside / base / upside scenario bands and milestone dates to funding
 * thresholds. This is deliberately the ONLY scale-up model shown:
 *   - the "current weighted pipeline" figure is the point-in-time bridge (View i,
 *     the Funded + Pipeline Forecast card above) — not repeated here;
 *   - the KFI run-rate × conversion model is withdrawn (it omitted KFI→completion
 *     attrition and over-stated scale), so it is not presented.
 */
export function ForecastExtrapolationPanel({
  client, portfolioId,
}: {
  client: AgentClient;
  portfolioId: string;
}) {
  const [data, setData] = useState<ForecastExtrapolation | null>(null);
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

  const current = data?.completionRunRateForecast ?? null;
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

  const insufficient = !current || current.available === false;

  return (
    <section className="space-y-4" data-testid="forecast-extrapolation-panel">
      <div className="flex items-center gap-2 text-sm font-semibold text-ink-100">
        <TrendingUp size={16} className="text-peri-300" /> Scale-up run-rate — when does the book reach scale?
      </div>
      <p className="-mt-2 text-[11px] text-ink-500">
        Forward projection from the recent completion run-rate (net funded growth). The point-in-time
        figure — if today&apos;s pipeline simply converts — is the Funded + Pipeline Forecast above.
      </p>

      {insufficient && (
        <Card className="p-4" testId="forecast-insufficient">
          <div className="text-[13px] font-medium text-amber-300">Insufficient history for the scale-up run-rate</div>
          <p className="mt-1 text-[12px] text-ink-400">
            {(current && "caveat" in current && current.caveat) ||
              "Not enough governed completion history to extrapolate. Showing the available signal only."}
          </p>
        </Card>
      )}

      {current && current.available && (
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
              {"observedMonths" in current && current.observedMonths != null && (
                <div><dt className="text-ink-500">Observed months</dt><dd className="text-ink-300">{current.observedMonths}</dd></div>
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
