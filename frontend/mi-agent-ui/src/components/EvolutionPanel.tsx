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
import { TimingDisclosureBanner } from "@/components/TimingDisclosureBanner";
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

// Evolution sub-tab semantics (A3 / A10): make stock vs funnel-flow vs forecast
// unambiguous.
const EVO_SUBTITLES: Record<EvoView, string> = {
  funded: "Funded book actuals by reporting month (balance, loan count, WA LTV, WA rate).",
  pipeline: "Stock of open pipeline exposure per weekly extract — amount, case count and weighted expected funded balance.",
  origination: "Funnel flow per week — KFI → Application → Offer → Completion value/count, 5-week average and conversion vs KFI.",
  // D: distinct from the main Forecast tab (which is the forward projection from
  // the latest run). This is the HISTORY of the forecast across reporting runs.
  forecast: "Forecast Evolution — historical movement in forecast metrics across reporting runs (how the forecast changed over time, and actual funded vs the prior run's forecast). For the forward projection from the latest run, use the main Forecast tab.",
};

// Sub-tab button labels (the "forecast" view reads as "Forecast Evolution").
const EVO_TAB_LABEL: Record<EvoView, string> = {
  funded: "Funded", pipeline: "Pipeline", origination: "Origination",
  forecast: "Forecast Evolution",
};

/** Data-quality annotation for a weekly pipeline series (A2): flags a sharp
 * week-on-week discontinuity (|Δ| over the threshold) so a real drop is explained
 * rather than smoothed away. Missing weeks are absent (null), never zero-filled. */
export function pipelineDataQuality(
  series: Array<{ period: string; pipeline_amount?: number | null }>,
  thresholdPct = 30,
): { period: string; changePct: number } | null {
  let worst: { period: string; changePct: number } | null = null;
  for (let i = 1; i < series.length; i++) {
    const prev = series[i - 1]?.pipeline_amount;
    const cur = series[i]?.pipeline_amount;
    if (prev == null || cur == null || prev === 0) continue;
    const changePct = ((cur - prev) / Math.abs(prev)) * 100;
    if (Math.abs(changePct) >= thresholdPct
        && (!worst || Math.abs(changePct) > Math.abs(worst.changePct))) {
      worst = { period: series[i].period, changePct: Math.round(changePct * 10) / 10 };
    }
  }
  return worst;
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
  valueFormat?: "gbp" | "count" | "pct" | "pct_points";
  source?: string | null;
  asOf?: string | null;
}) {
  const fmt = (v: number) =>
    valueFormat === "gbp" ? gbpCompact(v)
      : valueFormat === "pct" ? `${(v * 100).toFixed(1)}%`
        : valueFormat === "pct_points" ? `${v.toFixed(1)}%`
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

export type StageMetric = "value" | "count";
export type StageViewMode = "amount" | "count" | "conversion";

// The "Withdrawn" / "Unknown" stages are NOT part of the linear KFI→Completion
// funnel — kept out of the main funnel lines and conversion, shown separately.
const MAIN_FUNNEL = ["KFI", "APPLICATION", "OFFER", "COMPLETED"];
const STAGE_LABEL: Record<string, string> = {
  KFI: "KFIs", APPLICATION: "Applications", OFFER: "Offers",
  COMPLETED: "Completions", WITHDRAWN: "Withdrawn", UNKNOWN: "Unknown",
};

/** Pivot raw stage points to one row per extract date with a column per stage,
 * summing the chosen metric (`value` = £ amount, `count` = cases). */
export function pivotStage(rows: StagePoint[], metric: StageMetric = "value"): {
  data: Array<Record<string, number | string>>; stages: string[];
} {
  const periods = Array.from(new Set(rows.map((r) => r.period))).sort();
  const stages = orderStages(Array.from(new Set(rows.map((r) => r.stage))));
  const pick = (r: StagePoint) =>
    metric === "count" ? (r.count ?? 0) : (typeof r.value === "number" ? r.value : 0);
  const data = periods.map((p) => {
    const row: Record<string, number | string> = { period: p };
    for (const s of stages) {
      row[s] = rows.filter((r) => r.period === p && r.stage === s).reduce((a, r) => a + pick(r), 0);
    }
    return row;
  });
  return { data, stages };
}

/** The funnel-stage line keys for the amount/count views, in process order.
 * `includeKfi=false` drops KFI so the smaller downstream stages are readable. */
export function funnelLineStages(stages: string[], includeKfi: boolean): string[] {
  return orderStages(stages.filter((s) =>
    MAIN_FUNNEL.includes(normaliseStage(s)) && (includeKfi || normaliseStage(s) !== "KFI")));
}

/** Conversion-vs-KFI series (COUNT based): Application/Offer/Completion only —
 * KFI is the denominator (KFI/KFI = 100% is dropped). Divide-by-zero safe. */
export function stageConversionSeries(rows: StagePoint[]): {
  data: Array<Record<string, number | string>>; stages: string[];
} {
  const { data: countPivot } = pivotStage(rows, "count");
  const present = orderStages(Array.from(new Set(rows.map((r) => r.stage))));
  const stages = present.filter((s) =>
    ["APPLICATION", "OFFER", "COMPLETED"].includes(normaliseStage(s)));
  const data = countPivot.map((r) => {
    const out: Record<string, number | string> = { period: r.period };
    const kfi = (r.KFI as number) || 0;
    for (const s of stages) {
      const v = r[s] as number;
      out[s] = kfi ? Math.round((v / kfi) * 1000) / 10 : 0;
    }
    return out;
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
  const [stageMode, setStageMode] = useState<StageViewMode>("amount");
  const [includeKfi, setIncludeKfi] = useState(true);

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
  // By-stage chart data for the selected mode. Conversion uses count ratios vs
  // KFI (Application/Offer/Completion only); amount/count pivot the chosen metric
  // and chart the main funnel stages (KFI optional), Withdrawn/Unknown excluded.
  const stageChart = useMemo(() => {
    const rows = pipeline?.byStage ?? [];
    if (stageMode === "conversion") {
      const conv = stageConversionSeries(rows);
      return { data: conv.data, lines: conv.stages, format: "pct" as const };
    }
    const piv = pivotStage(rows, stageMode === "count" ? "count" : "value");
    const lines = funnelLineStages(piv.stages, includeKfi);
    return { data: piv.data, lines,
      format: (stageMode === "count" ? "count" : "gbp") as "count" | "gbp" };
  }, [pipeline, stageMode, includeKfi]);
  const hasWithdrawn = useMemo(
    () => (pipeline?.byStage ?? []).some((r) => normaliseStage(r.stage) === "WITHDRAWN"),
    [pipeline],
  );
  const forecastSeries = useMemo(
    () => (forecast?.periods ?? []).map((p) => ({ period: p.period, ...p.metrics })),
    [forecast],
  );
  // D: actual funded THIS run vs the PRIOR run's forecast — "did the forecast hold?".
  const forecastVariance = useMemo(() => {
    const periods = forecast?.periods ?? [];
    return periods.map((p, i) => ({
      period: p.period,
      actual_funded: (p.metrics?.funded_balance as number) ?? null,
      prior_forecast: i > 0
        ? ((periods[i - 1].metrics?.forecast_funded_balance as number) ?? null) : null,
    })).filter((r) => r.prior_forecast != null);
  }, [forecast],
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
                "rounded-md px-3 py-1 text-[12px] font-medium transition-colors",
                view === v ? "bg-navy-700/80 text-ink-100" : "text-ink-400 hover:text-ink-200",
              )}>
              {EVO_TAB_LABEL[v]}
            </button>
          ))}
        </div>
      </div>

      <p className="text-[11px] text-ink-500" data-testid="evo-subtitle">
        {EVO_SUBTITLES[view]}
      </p>

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

      {view === "pipeline" && (() => {
        const dq = pipelineDataQuality(pipelineSeries as Array<{ period: string; pipeline_amount?: number | null }>);
        return dq ? (
          <div className="rounded-lg border border-amber-400/30 bg-amber-400/5 px-3 py-2 text-[11px] text-amber-300/90"
            data-testid="pipeline-dq-badge">
            ⚠ Sharp movement: {dq.changePct > 0 ? "+" : ""}{dq.changePct}% in pipeline amount at {dq.period}.
            Verify weekly extract completeness — missing weeks are shown as gaps (not zero), so a real
            drop is not smoothed away.
          </div>
        ) : null;
      })()}

      {view === "pipeline" && (
        <TimingDisclosureBanner timing={pipeline?.pipelineTiming} />
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
          <div className="rounded-xl border border-[var(--color-line)] bg-navy-900/40 p-4 lg:col-span-2"
            data-testid="pipeline-by-stage">
            <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
              <div className="text-[12px] font-semibold text-ink-200">Pipeline by stage over time</div>
              <div className="flex items-center gap-2">
                {stageMode !== "conversion" && (
                  <label className="flex items-center gap-1 text-[10px] text-ink-400">
                    <input type="checkbox" checked={includeKfi}
                      onChange={(e) => setIncludeKfi(e.target.checked)}
                      aria-label="Include KFI" />
                    Include KFI
                  </label>
                )}
                <div role="tablist" aria-label="Stage view mode"
                  className="inline-flex items-center gap-0.5 rounded-lg border border-[var(--color-line)] bg-navy-900/60 p-0.5">
                  {([["amount", "Amount"], ["count", "Count"], ["conversion", "Conversion"]] as const).map(
                    ([m, label]) => (
                      <button key={m} type="button" role="tab" aria-selected={stageMode === m}
                        onClick={() => setStageMode(m)}
                        className={cn("rounded-md px-2 py-0.5 text-[10px] font-medium transition-colors",
                          stageMode === m ? "bg-navy-700/80 text-ink-100" : "text-ink-400 hover:text-ink-200")}>
                        {label}
                      </button>
                    ))}
                </div>
              </div>
            </div>
            <EvoLineChart
              title={stageMode === "amount" ? "Pipeline amount by stage (£)"
                : stageMode === "count" ? "Pipeline case count by stage"
                : "Conversion vs KFI (Application / Offer / Completion)"}
              data={stageChart.data}
              lines={stageChart.lines.map((s) => ({ key: s, label: STAGE_LABEL[normaliseStage(s)] ?? s }))}
              valueFormat={stageChart.format === "pct" ? "pct_points" : stageChart.format}
              source="weekly pipeline extracts" />
            <p className="mt-1 text-[10px] text-ink-500" data-testid="stage-mode-note">
              {stageMode === "conversion"
                ? "Conversion = each stage as a % of KFIs in the same week (count). KFI is the denominator and is not charted."
                : `${stageMode === "amount" ? "Amount (£)" : "Case count"} for the main funnel`
                  + ` (KFI → Application → Offer → Completion).${hasWithdrawn ? " Withdrawn is tracked separately, not in the funnel." : ""}`
                  + " Toggle 'Include KFI' off to read the smaller downstream stages."}
            </p>
          </div>
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
        <div className="space-y-3" data-testid="forecast-evolution">
          {forecastSeries.length < 2 && (
            <div className="rounded-lg border border-amber-400/20 bg-amber-400/5 px-3 py-2 text-[11px] text-amber-300/90"
              data-testid="forecast-evolution-insufficient">
              Only {forecastSeries.length} reporting run available — forecast history needs at
              least two runs to show how the forecast has changed. Showing what is available.
            </div>
          )}
          <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
            <EvoLineChart title="Forecast funded balance by reporting run" data={forecastSeries}
              lines={[
                { key: "funded_balance", label: "Funded actual" },
                { key: "weighted_expected_pipeline", label: "Weighted pipeline" },
                { key: "forecast_funded_balance", label: "Forecast (funded + pipeline)" },
              ]} valueFormat="gbp" source="funded tapes + weighted pipeline" />
            {forecastVariance.length > 0 && (
              <EvoLineChart title="Actual funded vs prior-run forecast" data={forecastVariance}
                lines={[
                  { key: "prior_forecast", label: "Prior-run forecast" },
                  { key: "actual_funded", label: "Actual funded" },
                ]} valueFormat="gbp" source="this run's actual vs the prior run's forecast" />
            )}
          </div>
          <p className="text-[10px] text-ink-500" data-testid="forecast-evolution-lineage">
            Runs: {(forecast?.periods ?? []).map((p) => p.run_id ?? p.period).join(", ") || "—"}.
            {" "}Forecast basis: funded balance + Σ(weighted expected pipeline) per run.
            {" "}Actual-vs-forecast: {forecastVariance.length > 0 ? "available" : "needs a prior run"}.
          </p>
        </div>
      )}
    </section>
  );
}
