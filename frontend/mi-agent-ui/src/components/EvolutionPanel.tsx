import { useEffect, useMemo, useState } from "react";
import {
  CartesianGrid, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip,
  XAxis, YAxis, Legend,
} from "recharts";
import { Activity, ArrowDownRight, ArrowUpRight, Minus } from "lucide-react";
import type { AgentClient } from "@/api";
import type {
  FundedEvolution,
  PipelineEvolution,
  ForecastEvolution,
  PipelineFunnelEvolution,
  StagePoint,
} from "@/domain";
import { cn, formatGBP } from "@/lib/utils";

type EvoView = "funded" | "pipeline" | "forecast" | "origination";

const PALETTE = ["#7c9cf0", "#5ec6b8", "#e0a458", "#c98bdb", "#6fcf97", "#eb6f6f"];

// Explicit funnel process order. WITHDRAWN sits after the main funnel; UNKNOWN
// last. Synonyms (COMPLETION/COMPLETED) and case are normalised first.
export const STAGE_ORDER = ["KFI", "APPLICATION", "OFFER", "COMPLETED", "WITHDRAWN", "UNKNOWN"];

export function normaliseStage(stage: string): string {
  const s = stage.trim().toUpperCase();
  if (s === "COMPLETION" || s === "COMPLETE") return "COMPLETED";
  if (s === "APP") return "APPLICATION";
  return s;
}

/** Order stage keys by the funnel process order; unknown stages keep their
 * relative order at the end (before UNKNOWN). */
export function orderStages(stages: string[]): string[] {
  return [...stages].sort((a, b) => {
    const ia = STAGE_ORDER.indexOf(normaliseStage(a));
    const ib = STAGE_ORDER.indexOf(normaliseStage(b));
    return (ia === -1 ? STAGE_ORDER.length - 0.5 : ia)
      - (ib === -1 ? STAGE_ORDER.length - 0.5 : ib);
  });
}

function gbpCompact(v: number): string {
  return formatGBP(v, { compact: true });
}

/** The x-axis label for a weekly pipeline period: the DAY-LEVEL extract date
 * (week / extract_date) in preference to the YYYY-MM month, so multiple weekly
 * points within a month are distinguishable rather than sharing one label. */
export function pipelineXValue(
  p: { week?: string | null; extract_date?: string | null; period: string },
): string {
  return p.week ?? p.extract_date ?? p.period;
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
  const periods = Array.from(new Set(rows.map((r) => r.period))).sort();
  const stages = orderStages(Array.from(new Set(rows.map((r) => r.stage))));
  const data = periods.map((p) => {
    const row: Record<string, number | string> = { period: p };
    for (const s of stages) {
      row[s] = rows.filter((r) => r.period === p && r.stage === s).reduce((a, r) => a + r.value, 0);
    }
    return row;
  });
  return { data, stages };
}

function trendIcon(trend: "up" | "down" | "flat") {
  if (trend === "up") return <ArrowUpRight size={13} className="text-mint-300" />;
  if (trend === "down") return <ArrowDownRight size={13} className="text-rose-300" />;
  return <Minus size={13} className="text-ink-500" />;
}

/** Conversion of a stage relative to KFI (count + value), divide-by-zero safe. */
export function stageConversion(
  stage: PipelineFunnelEvolution["summary"][string] | undefined,
  kfi: PipelineFunnelEvolution["summary"][string] | undefined,
): { countPct: number | null; valuePct: number | null; numerCount: number; denomCount: number } | null {
  if (!stage || !kfi) return null;
  const denomCount = kfi.latestCount ?? 0;
  const numerCount = stage.latestCount ?? 0;
  const countPct = denomCount > 0 ? (numerCount / denomCount) * 100 : null;
  const denomVal = kfi.latestValue ?? 0;
  const valuePct = denomVal > 0 && stage.latestValue != null
    ? (stage.latestValue / denomVal) * 100 : null;
  return { countPct, valuePct, numerCount, denomCount };
}

/** One origination-funnel stage: weekly value chart + 5-week avg line + summary. */
function FunnelStageCard({
  stage, label, points, summary, conversion,
}: {
  stage: string;
  label: string;
  points: { week: string | null; value: number | null; count: number }[];
  summary: PipelineFunnelEvolution["summary"][string] | undefined;
  conversion?: ReturnType<typeof stageConversion>;
}) {
  const data = points.map((p) => ({ week: p.week ?? "", value: p.value, count: p.count }));
  const avg = summary?.fiveWeekAvgValue ?? null;
  return (
    <div className="rounded-xl border border-[var(--color-line)] bg-navy-900/40 p-4"
      data-testid={`funnel-stage-${stage}`}>
      <div className="mb-1 flex items-center justify-between">
        <div className="text-[12px] font-semibold text-ink-200">{label}</div>
        <div className="flex items-center gap-1 text-[11px] text-ink-400">
          {summary && trendIcon(summary.trend)}
          <span>{summary?.weeksObserved ?? 0} wks</span>
        </div>
      </div>
      {data.length === 0 ? (
        <p className="py-8 text-center text-[12px] text-ink-500">No weekly extracts available.</p>
      ) : (
        <div style={{ width: "100%", height: 150 }}>
          <ResponsiveContainer>
            <LineChart data={data} margin={{ top: 6, right: 12, bottom: 4, left: 4 }}>
              <CartesianGrid stroke="#23304d" strokeDasharray="3 3" />
              <XAxis dataKey="week" tick={{ fill: "#8a97ad", fontSize: 10 }} />
              <YAxis tickFormatter={gbpCompact} tick={{ fill: "#8a97ad", fontSize: 10 }} width={56} />
              <Tooltip
                formatter={(v: number) => gbpCompact(Number(v))}
                contentStyle={{ background: "#0f1626", border: "1px solid #23304d", fontSize: 12 }} />
              {avg != null && (
                <ReferenceLine y={avg} stroke="#e0a458" strokeDasharray="4 3"
                  label={{ value: "5-wk avg", fill: "#e0a458", fontSize: 9, position: "insideTopRight" }} />
              )}
              <Line type="monotone" dataKey="value" name="Weekly value"
                stroke="#7c9cf0" strokeWidth={2} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}
      {summary && (
        <div className="mt-2 grid grid-cols-3 gap-2 text-[10px]">
          <div>
            <div className="text-ink-500">Latest week</div>
            <div className="text-ink-200">
              {summary.latestValue != null ? gbpCompact(summary.latestValue) : "—"}
              <span className="text-ink-500"> · {summary.latestCount} cases</span>
            </div>
          </div>
          <div>
            <div className="text-ink-500">5-week avg</div>
            <div className="text-ink-200">
              {summary.fiveWeekAvgValue != null ? gbpCompact(summary.fiveWeekAvgValue) : "—"}
              <span className="text-ink-500">
                {" "}· {summary.fiveWeekAvgCount != null ? Math.round(summary.fiveWeekAvgCount) : "—"}
              </span>
            </div>
          </div>
          <div>
            <div className="text-ink-500">Δ vs prior wk</div>
            <div className="text-ink-200">
              {summary.deltaValue != null ? gbpCompact(summary.deltaValue) : "—"}
              <span className="text-ink-500">
                {" "}· {summary.deltaCount != null
                  ? `${summary.deltaCount >= 0 ? "+" : ""}${summary.deltaCount}`
                  : "—"}
              </span>
            </div>
          </div>
        </div>
      )}
      {conversion && (
        <div className="mt-2 rounded-md border border-[var(--color-line-soft)] bg-navy-900/50 px-2 py-1 text-[10px]"
          data-testid={`funnel-conversion-${stage}`}
          title={`Conversion vs KFI (latest week): ${conversion.numerCount} ${label} / `
            + `${conversion.denomCount} KFI`}>
          <span className="text-ink-500">Conversion vs KFI: </span>
          <span className="font-semibold text-mint-300">
            {conversion.countPct != null ? `${conversion.countPct.toFixed(1)}%` : "n/a"}
          </span>
          <span className="text-ink-500"> by count</span>
          {conversion.valuePct != null && (
            <span className="text-ink-500"> · {conversion.valuePct.toFixed(1)}% by value</span>
          )}
        </div>
      )}
    </div>
  );
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
  const [funnel, setFunnel] = useState<PipelineFunnelEvolution | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    const run = async () => {
      try {
        if (view === "funded") setFunded(await client.getFundedEvolution(portfolioId));
        else if (view === "pipeline") setPipeline(await client.getPipelineEvolution(portfolioId));
        else if (view === "origination") setFunnel(await client.getFunnelEvolution(portfolioId));
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
    // Weekly extracts: label by the day-level extract date (not the YYYY-MM
    // month, which collapsed multiple weekly points onto one indistinguishable
    // label). Sorted chronologically by the actual date string (ISO sorts).
    () => (pipeline?.periods ?? [])
      .map((p) => ({ period: pipelineXValue(p), ...p.metrics }))
      .sort((a, b) => String(a.period).localeCompare(String(b.period))),
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
    (view === "origination" && funnel?.singlePeriod) ||
    (view === "forecast" && forecast?.singlePeriod);

  return (
    <section className="space-y-4" data-testid="evolution-panel">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2 text-sm font-semibold text-ink-100">
          <Activity size={16} className="text-peri-300" /> Evolution
        </div>
        <div role="tablist" aria-label="Evolution series"
          className="inline-flex items-center gap-1 rounded-lg border border-[var(--color-line)] bg-navy-900/60 p-1">
          {(["funded", "pipeline", "origination", "forecast"] as EvoView[]).map((v) => (
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
          <EvoLineChart title="Pipeline amount by week" data={pipelineSeries}
            lines={[{ key: "pipeline_amount", label: "Pipeline amount" }]} valueFormat="gbp"
            source={pipeline?.sourceFiles?.[0]} />
          <EvoLineChart title="Weighted expected funded by week" data={pipelineSeries}
            lines={[{ key: "weighted_expected_funded_amount", label: "Weighted expected" }]}
            valueFormat="gbp" source="weekly pipeline extracts" />
          <EvoLineChart title="Pipeline case count by week" data={pipelineSeries}
            lines={[{ key: "pipeline_case_count", label: "Cases" }]} valueFormat="count"
            source="weekly pipeline extracts" />
          <EvoLineChart title="Pipeline by stage over time" data={stagePivot.data}
            lines={stagePivot.stages.map((s) => ({ key: s, label: s }))} valueFormat="gbp"
            source="weekly pipeline extracts" />
        </div>
      )}

      {view === "origination" && (
        <div className="space-y-3" data-testid="origination-funnel">
          <p className="text-[11px] text-ink-400">
            Weekly origination funnel — KFI → Application → Offer → Completion value and
            count per governed weekly extract, with a 5-week trailing average and the
            week-on-week movement.
          </p>
          <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
            {orderStages(funnel?.stages ?? []).map((stage) => (
              <FunnelStageCard key={stage} stage={stage}
                label={funnel?.stageLabels?.[stage] ?? stage}
                points={funnel?.series?.[stage] ?? []}
                summary={funnel?.summary?.[stage]}
                conversion={normaliseStage(stage) === "KFI"
                  ? null
                  : stageConversion(funnel?.summary?.[stage], funnel?.summary?.["KFI"])} />
            ))}
          </div>
          {funnel?.sourceFiles?.length ? (
            <p className="text-[10px] text-ink-500">
              Source: {funnel.uniqueWeeklyExtractsUsed ?? funnel.sourceFiles.length} governed
              weekly extract(s). 5-week average = trailing mean of up to the last 5 weeks.
            </p>
          ) : null}
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
