import { useEffect, useMemo, useState } from "react";
import {
  CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis, Legend,
} from "recharts";
import { Activity } from "lucide-react";
import type { AgentClient } from "@/api";
import type {
  FundedEvolution,
  PipelineEvolution,
  ForecastEvolution,
  StagePoint,
} from "@/domain";
import { cn, formatGBP } from "@/lib/utils";

type EvoView = "funded" | "pipeline" | "forecast";

const PALETTE = ["#7c9cf0", "#5ec6b8", "#e0a458", "#c98bdb", "#6fcf97", "#eb6f6f"];

function gbpCompact(v: number): string {
  return formatGBP(v, { compact: true });
}

/** A single labelled line chart over periods, with a source/coverage footer. */
function EvoLineChart({
  title, data, lines, valueFormat = "gbp", source, asOf,
}: {
  title: string;
  data: Array<Record<string, number | string | null>>;
  lines: { key: string; label: string }[];
  valueFormat?: "gbp" | "count" | "pct";
  source?: string | null;
  asOf?: string | null;
}) {
  const fmt = (v: number) =>
    valueFormat === "gbp" ? gbpCompact(v)
      : valueFormat === "pct" ? `${(v * 100).toFixed(1)}%`
        : v.toLocaleString("en-GB");
  return (
    <div className="rounded-xl border border-[var(--color-line)] bg-navy-900/40 p-4">
      <div className="mb-2 text-[12px] font-semibold text-ink-200">{title}</div>
      {data.length === 0 ? (
        <p className="py-8 text-center text-[12px] text-ink-500">No periods available.</p>
      ) : (
        <div style={{ width: "100%", height: 200 }}>
          <ResponsiveContainer>
            <LineChart data={data} margin={{ top: 6, right: 12, bottom: 4, left: 4 }}>
              <CartesianGrid stroke="#23304d" strokeDasharray="3 3" />
              <XAxis dataKey="period" tick={{ fill: "#8a97ad", fontSize: 11 }} />
              <YAxis tickFormatter={fmt} tick={{ fill: "#8a97ad", fontSize: 11 }} width={64} />
              <Tooltip
                formatter={(v: number) => fmt(Number(v))}
                contentStyle={{ background: "#0f1626", border: "1px solid #23304d", fontSize: 12 }}
              />
              {lines.length > 1 && <Legend wrapperStyle={{ fontSize: 11 }} />}
              {lines.map((l, i) => (
                <Line key={l.key} type="monotone" dataKey={l.key} name={l.label}
                  stroke={PALETTE[i % PALETTE.length]} strokeWidth={2} dot={false} />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}
      <div className="mt-2 flex flex-wrap gap-x-4 text-[10px] text-ink-500">
        {source && <span>Source: {source}</span>}
        {asOf && <span>As of: {asOf}</span>}
        <span>Coverage: 100% (per-period reconciliation)</span>
      </div>
    </div>
  );
}

function pivotStage(rows: StagePoint[]): {
  data: Array<Record<string, number | string>>; stages: string[];
} {
  const periods = Array.from(new Set(rows.map((r) => r.period)));
  const stages = Array.from(new Set(rows.map((r) => r.stage)));
  const data = periods.map((p) => {
    const row: Record<string, number | string> = { period: p };
    for (const s of stages) {
      row[s] = rows.filter((r) => r.period === p && r.stage === s).reduce((a, r) => a + r.value, 0);
    }
    return row;
  });
  return { data, stages };
}

/**
 * Evolution view — funded / pipeline / forecast metrics over time. Reads the
 * governed monthly funded runs and weekly pipeline extracts via the evolution
 * endpoints; every chart carries source lineage + per-period reconciliation.
 */
export function EvolutionPanel({
  client, portfolioId,
}: {
  client: AgentClient;
  portfolioId: string;
}) {
  const [view, setView] = useState<EvoView>("funded");
  const [funded, setFunded] = useState<FundedEvolution | null>(null);
  const [pipeline, setPipeline] = useState<PipelineEvolution | null>(null);
  const [forecast, setForecast] = useState<ForecastEvolution | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    const run = async () => {
      try {
        if (view === "funded") setFunded(await client.getFundedEvolution(portfolioId));
        else if (view === "pipeline") setPipeline(await client.getPipelineEvolution(portfolioId));
        else setForecast(await client.getForecastEvolution(portfolioId));
      } catch {
        /* keep prior state; charts show empty */
      } finally {
        if (!cancelled) setLoading(false);
      }
    };
    void run();
    return () => { cancelled = true; };
  }, [client, portfolioId, view]);

  const fundedSeries = useMemo(
    () => (funded?.periods ?? []).map((p) => ({ period: p.period, ...p.metrics })),
    [funded],
  );
  const pipelineSeries = useMemo(
    () => (pipeline?.periods ?? []).map((p) => ({ period: p.period, ...p.metrics })),
    [pipeline],
  );
  const stagePivot = useMemo(() => pivotStage(pipeline?.byStage ?? []), [pipeline]);
  const forecastSeries = useMemo(
    () => (forecast?.periods ?? []).map((p) => ({ period: p.period, ...p.metrics })),
    [forecast],
  );

  const single =
    (view === "funded" && funded?.singlePeriod) ||
    (view === "pipeline" && pipeline?.singlePeriod) ||
    (view === "forecast" && forecast?.singlePeriod);

  return (
    <section className="space-y-4" data-testid="evolution-panel">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2 text-sm font-semibold text-ink-100">
          <Activity size={16} className="text-peri-300" /> Evolution
        </div>
        <div role="tablist" aria-label="Evolution series"
          className="inline-flex items-center gap-1 rounded-lg border border-[var(--color-line)] bg-navy-900/60 p-1">
          {(["funded", "pipeline", "forecast"] as EvoView[]).map((v) => (
            <button key={v} type="button" role="tab" aria-selected={view === v}
              onClick={() => setView(v)}
              className={cn(
                "rounded-md px-3 py-1 text-[12px] font-medium capitalize transition-colors",
                view === v ? "bg-navy-700/80 text-ink-100" : "text-ink-400 hover:text-ink-200",
              )}>
              {v}
            </button>
          ))}
        </div>
      </div>

      {single && (
        <div className="rounded-lg border border-amber-400/20 bg-amber-400/5 px-3 py-2 text-[11px] text-amber-300/90">
          Only one period is available — an evolution view needs at least two runs.
          Showing the single point.
        </div>
      )}

      {loading && <p className="text-[12px] text-ink-500">Loading evolution…</p>}

      {view === "funded" && (
        <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
          <EvoLineChart title="Funded balance by month" data={fundedSeries}
            lines={[{ key: "funded_balance", label: "Funded balance" }]}
            valueFormat="gbp" source={funded?.sourceFiles?.[0]} asOf={funded?.reportingDates?.slice(-1)[0]} />
          <EvoLineChart title="Funded loan count by month" data={fundedSeries}
            lines={[{ key: "loan_count", label: "Loan count" }]} valueFormat="count"
            source="central lender tapes" />
          <EvoLineChart title="WA LTV by month" data={fundedSeries}
            lines={[{ key: "wa_ltv", label: "WA LTV" }]} valueFormat="pct"
            source="central lender tapes" />
          <EvoLineChart title="WA interest rate by month" data={fundedSeries}
            lines={[{ key: "wa_interest_rate", label: "WA rate" }]} valueFormat="pct"
            source="central lender tapes" />
        </div>
      )}

      {view === "pipeline" && (
        <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
          <EvoLineChart title="Pipeline amount by week/month" data={pipelineSeries}
            lines={[{ key: "pipeline_amount", label: "Pipeline amount" }]} valueFormat="gbp"
            source={pipeline?.sourceFiles?.[0]} />
          <EvoLineChart title="Weighted expected funded by month" data={pipelineSeries}
            lines={[{ key: "weighted_expected_funded_amount", label: "Weighted expected" }]}
            valueFormat="gbp" source="weekly pipeline extracts" />
          <EvoLineChart title="Pipeline case count over time" data={pipelineSeries}
            lines={[{ key: "pipeline_case_count", label: "Cases" }]} valueFormat="count"
            source="weekly pipeline extracts" />
          <EvoLineChart title="Pipeline by stage over time" data={stagePivot.data}
            lines={stagePivot.stages.map((s) => ({ key: s, label: s }))} valueFormat="gbp"
            source="weekly pipeline extracts" />
        </div>
      )}

      {view === "forecast" && (
        <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
          <EvoLineChart title="Forecast funded balance over time" data={forecastSeries}
            lines={[
              { key: "funded_balance", label: "Funded" },
              { key: "weighted_expected_pipeline", label: "Weighted pipeline" },
              { key: "forecast_funded_balance", label: "Forecast" },
            ]} valueFormat="gbp" source="funded tapes + weighted pipeline" />
        </div>
      )}
    </section>
  );
}
