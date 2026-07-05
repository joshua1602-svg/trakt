import { useEffect, useMemo, useState, type ReactNode } from "react";
import {
  Bar, CartesianGrid, ComposedChart, Line, LineChart, ReferenceLine,
  ResponsiveContainer, Tooltip, XAxis, YAxis, Legend,
} from "recharts";
import { Activity, ArrowDownRight, ArrowUpRight, ChevronDown, Maximize2, Minus, X } from "lucide-react";
import type { AgentClient } from "@/api";
import type {
  FundedEvolution,
  PipelineEvolution,
  ForecastEvolution,
  PipelineFunnelEvolution,
  FunnelConversion,
  FunnelFlowPoint,
  FunnelPoint,
  CohortAnalysis,
  CohortDimension,
  CohortGrain,
  CohortProgression,
  SourcePortfolioLens,
  StagePoint,
} from "@/domain";
import { TimingDisclosureBanner } from "@/components/TimingDisclosureBanner";
import { cn, formatGBP } from "@/lib/utils";

type EvoView = "funded" | "pipeline" | "forecast" | "origination" | "cohorts";

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
  cohorts: "Cohorts — funded book by a selectable lens (vintage, borrower age, LTV band or origination channel): balance, loan count, book share and balance-weighted LTV / rate / months-on-book per cohort, as of the selected reporting date.",
};

// Sub-tab button labels (the "forecast" view reads as "Forecast Evolution").
const EVO_TAB_LABEL: Record<EvoView, string> = {
  funded: "Funded", pipeline: "Pipeline", origination: "Origination",
  forecast: "Forecast Evolution", cohorts: "Cohorts",
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
 * KFI is the denominator (KFI/KFI = 100% is dropped). Divide-by-zero safe.
 *
 * `lagWeeks` shifts the KFI denominator back by the KFI→completion timeline so
 * each week's stage count is measured against the KFI book those cases came
 * from — not the current (still-growing) book. When null the series is unlagged
 * (same-week). The lag is in periods; the by-stage extracts are weekly, matching
 * the funnel's `conversionLagWeeks`. */
export function stageConversionSeries(rows: StagePoint[], lagWeeks?: number | null): {
  data: Array<Record<string, number | string>>; stages: string[];
} {
  const { data: countPivot } = pivotStage(rows, "count");
  const present = orderStages(Array.from(new Set(rows.map((r) => r.stage))));
  const stages = present.filter((s) =>
    ["APPLICATION", "OFFER", "COMPLETED"].includes(normaliseStage(s)));
  const lag = Math.max(0, Math.round(lagWeeks ?? 0));
  const data = countPivot.map((r, i) => {
    const out: Record<string, number | string> = { period: r.period };
    // Denominator: KFI stock `lag` periods earlier (clamped to the first week).
    const denomRow = countPivot[Math.max(0, i - lag)];
    const kfi = (denomRow.KFI as number) || 0;
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

function pct1(v: number | null | undefined): string {
  return v == null ? "n/a" : `${v.toFixed(1)}%`;
}

/** Collapsed-by-default conversion disclosure. Leads with the CANONICAL metric —
 * cumulative cohort conversion (% of the original KFI cohort that has reached
 * this milestone to date) — and shows the weekly completion velocity below it as
 * a labelled operational/forecast input, NOT as "conversion". Hidden until
 * expanded to keep the card calm. */
function ConversionDisclosure({ stage, conversion, cohortPct }: {
  stage: string;
  conversion: FunnelConversion;
  cohortPct: number | null;
}) {
  const [open, setOpen] = useState(false);
  const lagLabel = conversion.lagApplied && conversion.lagWeeks != null
    ? `KFI stock lagged ${conversion.lagWeeks}w`
    : "KFI stock unlagged";
  return (
    <div className="mt-2 rounded-md border border-[var(--color-line-soft)] bg-navy-900/50 text-[10px]"
      data-testid={`funnel-conversion-${stage}`}>
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        aria-expanded={open}
        className="flex w-full items-center justify-between gap-1.5 px-2 py-1.5 text-left font-medium uppercase tracking-wide text-ink-400 hover:text-ink-200"
      >
        <span className="flex items-center gap-1.5">
          <ChevronDown size={12} className={cn("transition-transform", !open && "-rotate-90")} />
          Cohort conversion
        </span>
        {cohortPct != null && <span className="font-semibold normal-case text-mint-300">{pct1(cohortPct)}</span>}
      </button>
      {open && (
        <div className="px-2 pb-2" data-testid={`funnel-conversion-body-${stage}`}>
          <div className="text-[10px] leading-snug">
            <span className="font-semibold text-mint-300">{pct1(cohortPct)}</span>
            <span className="text-ink-500"> of the KFI cohort has reached this milestone to date (cumulative, cohort-tracked).</span>
          </div>
          <div className="mt-1.5 border-t border-[var(--color-line-soft)] pt-1.5 text-[9px] leading-snug text-ink-500">
            <span className="uppercase tracking-wide text-ink-400">Weekly velocity</span> — forecast input, not conversion:{" "}
            <span className="text-ink-300">{pct1(conversion.weeklyRateValue)}/wk by value · {pct1(conversion.weeklyRateCount)}/wk by count</span>{" "}
            (avg weekly flow, last 5 wks ÷ {lagLabel}).
          </div>
          {!conversion.sufficient && (
            <div className="mt-1 text-[9px] font-medium leading-snug text-amber-300"
              data-testid={`funnel-conversion-provisional-${stage}`}>
              Velocity provisional — {conversion.weeksInWindow} of {conversion.minWeeks}+ weeks; too few to forecast off yet.
            </div>
          )}
        </div>
      )}
    </div>
  );
}

/** One origination-funnel stage: weekly-FLOW bars (default) with an optional
 * stock line, a 5-week trailing average of the WEEKLY FLOW, the Δ vs prior week
 * (flow − prior flow), and a collapsed conversion-vs-KFI disclosure. Renders
 * compact in the 2×2 grid and larger inside the focus modal (``large``). */
function FunnelStageCard({
  stage, label, points, flowPoints, summary, conversion, cohortPct, showCumulative, large, onExpand,
}: {
  stage: string;
  label: string;
  points: FunnelPoint[];
  flowPoints: FunnelFlowPoint[];
  summary: PipelineFunnelEvolution["summary"][string] | undefined;
  conversion: FunnelConversion | null;
  /** Latest cumulative cohort % for this stage (the canonical conversion). */
  cohortPct: number | null;
  showCumulative: boolean;
  /** Larger chart for the focus modal. */
  large?: boolean;
  /** Open this stage in the focus modal (compact card only). */
  onExpand?: () => void;
}) {
  // Join the weekly-flow bars with the stock level per week.
  const data = flowPoints.map((f, i) => ({
    week: f.week ?? "",
    flow: f.flowValue,
    stock: points[i]?.value ?? null,
  }));
  const avgFlow = summary?.fiveWeekAvgFlowValue ?? null;
  const hasFlow = data.some((d) => d.flow != null);
  return (
    <div className={cn("rounded-xl border border-[var(--color-line)] bg-navy-900/40 p-4", large && "h-full")}
      data-testid={`funnel-stage-${stage}`}>
      <div className="mb-1 flex items-center justify-between">
        <div className="text-[12px] font-semibold text-ink-200">
          {label} <span className="font-normal text-ink-500">· weekly flow</span>
        </div>
        <div className="flex items-center gap-1.5 text-[11px] text-ink-400">
          {summary && trendIcon(summary.trend)}
          <span>{summary?.weeksObserved ?? 0} wks</span>
          {onExpand && (
            <button
              type="button"
              onClick={onExpand}
              aria-label={`Enlarge ${label} chart`}
              title="View larger"
              data-testid={`funnel-expand-${stage}`}
              className="inline-flex h-6 w-6 items-center justify-center rounded-md text-ink-400 hover:bg-navy-800 hover:text-ink-100"
            >
              <Maximize2 size={13} />
            </button>
          )}
        </div>
      </div>
      {!hasFlow ? (
        <p className="py-8 text-center text-[12px] text-ink-500">
          Need ≥2 weekly extracts to show weekly flow.
        </p>
      ) : (
        <div style={{ width: "100%", height: large ? 380 : 160 }}>
          <ResponsiveContainer>
            <ComposedChart data={data} margin={{ top: 6, right: 12, bottom: 4, left: 4 }}>
              <CartesianGrid stroke="#23304d" strokeDasharray="3 3" />
              <XAxis dataKey="week" tick={{ fill: "#8a97ad", fontSize: large ? 11 : 10 }} />
              <YAxis yAxisId="flow" tickFormatter={gbpCompact}
                tick={{ fill: "#8a97ad", fontSize: large ? 11 : 10 }} width={56} />
              {showCumulative && (
                <YAxis yAxisId="stock" orientation="right" tickFormatter={gbpCompact}
                  tick={{ fill: "#6f7b91", fontSize: large ? 11 : 10 }} width={56} />
              )}
              <Tooltip
                formatter={(v: number, name: string) => [gbpCompact(Number(v)), name]}
                contentStyle={{ background: "#0f1626", border: "1px solid #23304d", fontSize: 12 }} />
              {avgFlow != null && (
                <ReferenceLine yAxisId="flow" y={avgFlow} stroke="#e0a458" strokeDasharray="4 3"
                  label={{ value: "5-wk avg flow", fill: "#e0a458", fontSize: 9, position: "insideTopRight" }} />
              )}
              <Bar yAxisId="flow" dataKey="flow" name="Weekly flow (£)"
                fill="#7c9cf0" radius={[2, 2, 0, 0]} />
              {showCumulative && (
                <Line yAxisId="stock" type="monotone" dataKey="stock" name="Stock (£)"
                  stroke="#5ec6b8" strokeWidth={2} dot={false} />
              )}
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      )}
      {summary && (
        <div className="mt-2 grid grid-cols-3 gap-2 text-[10px]">
          <div>
            <div className="text-ink-500">Latest weekly flow</div>
            <div className="text-ink-200">
              {summary.latestFlowValue != null ? gbpCompact(summary.latestFlowValue) : "—"}
              <span className="text-ink-500">
                {" "}· {summary.latestFlowCount != null
                  ? `${summary.latestFlowCount >= 0 ? "+" : ""}${summary.latestFlowCount}`
                  : "—"} cases
              </span>
            </div>
          </div>
          <div>
            <div className="text-ink-500" title="Trailing mean of the last 5 weeks of weekly flow">
              5-wk avg flow
            </div>
            <div className="text-ink-200">
              {summary.fiveWeekAvgFlowValue != null ? gbpCompact(summary.fiveWeekAvgFlowValue) : "—"}
              <span className="text-ink-500">
                {" "}· {summary.fiveWeekAvgFlowCount != null ? Math.round(summary.fiveWeekAvgFlowCount) : "—"}
              </span>
            </div>
          </div>
          <div>
            <div className="text-ink-500" title="Latest weekly flow minus prior weekly flow">
              Δ vs prior wk
            </div>
            <div className="text-ink-200">
              {summary.deltaFlowValue != null ? gbpCompact(summary.deltaFlowValue) : "—"}
              <span className="text-ink-500">
                {" "}· {summary.deltaFlowCount != null
                  ? `${summary.deltaFlowCount >= 0 ? "+" : ""}${summary.deltaFlowCount}`
                  : "—"}
              </span>
            </div>
          </div>
        </div>
      )}
      {conversion && <ConversionDisclosure stage={stage} conversion={conversion} cohortPct={cohortPct} />}
    </div>
  );
}

/** Lightweight focus modal for a single enlarged chart. Escape / backdrop /
 * close button all dismiss it; only one chart is enlarged at a time. */
function ChartFocusModal({ title, onClose, children }: {
  title: string;
  onClose: () => void;
  children: ReactNode;
}) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") onClose(); };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [onClose]);
  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-label={title}
      data-testid="funnel-focus-modal"
      className="fixed inset-0 z-50 flex items-center justify-center bg-navy-950/70 p-4 backdrop-blur-sm"
      onMouseDown={(e) => { if (e.target === e.currentTarget) onClose(); }}
    >
      <div className="flex max-h-[90vh] w-full max-w-3xl flex-col overflow-hidden rounded-2xl border border-[var(--color-line)] bg-navy-900 shadow-2xl">
        <div className="flex items-center justify-between gap-3 border-b border-[var(--color-line)] px-4 py-3">
          <h3 className="text-sm font-semibold text-ink-100">{title}</h3>
          <button
            type="button"
            onClick={onClose}
            aria-label="Close enlarged chart"
            data-testid="funnel-focus-close"
            className="inline-flex h-7 w-7 items-center justify-center rounded-md text-ink-400 hover:bg-navy-800 hover:text-ink-100"
          >
            <X size={16} />
          </button>
        </div>
        <div className="min-h-0 flex-1 overflow-y-auto p-4">{children}</div>
      </div>
    </div>
  );
}

/** Short, collapsed methodology note (replaces the long inline paragraph). */
function MethodologyDisclosure() {
  const [open, setOpen] = useState(false);
  return (
    <div className="relative">
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        aria-expanded={open}
        data-testid="origination-methodology-toggle"
        className="inline-flex items-center gap-1.5 rounded-md border border-[var(--color-line)] bg-navy-900/60 px-2 py-1 text-[10px] text-ink-400 hover:text-ink-200"
      >
        <ChevronDown size={11} className={cn("transition-transform", !open && "-rotate-90")} />
        Methodology
      </button>
      {open && (
        <div className="absolute right-0 z-20 mt-1 w-72 rounded-lg border border-[var(--color-line)] bg-navy-900 px-3 py-2 text-[11px] leading-relaxed text-ink-300 shadow-xl"
          data-testid="origination-methodology-body">
          Bars show <span className="text-ink-100">weekly flow</span> — the week-on-week change in each
          stage. The 5-week average and Δ-vs-prior-week are on this weekly-flow basis. Toggle
          “Show stock line” to overlay the stock level.
        </div>
      )}
    </div>
  );
}

function pctFmt(v: number | null | undefined): string {
  return v == null ? "—" : `${(v * 100).toFixed(1)}%`;
}
function mobFmt(v: number | null | undefined): string {
  return v == null ? "—" : `${Math.round(v)} mo`;
}

/** A compact labelled dropdown used by the cohort selector row. */
function CohortSelect({ label, value, onChange, options, testId }: {
  label: string; value: string; onChange: (v: string) => void;
  options: { value: string; label: string }[]; testId?: string;
}) {
  return (
    <label className="flex flex-col gap-0.5">
      <span className="text-[9px] font-medium uppercase tracking-wider text-ink-500">{label}</span>
      <select
        data-testid={testId}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="rounded-md border border-[var(--color-line)] bg-navy-900 px-2 py-1 text-[11px] text-ink-100 focus:border-peri-400/50 focus:outline-none"
      >
        {options.map((o) => <option key={o.value} value={o.value}>{o.label}</option>)}
      </select>
    </label>
  );
}

const _PROG_METRICS: { key: string; label: string; fmt: "gbp" | "pct" | "count"; nneg?: boolean }[] = [
  { key: "funded_balance", label: "Funded balance", fmt: "gbp" },
  { key: "wa_ltv", label: "WA LTV", fmt: "pct" },
  { key: "wa_interest_rate", label: "WA rate", fmt: "pct" },
  { key: "nneg_headroom_pct", label: "NNEG headroom", fmt: "pct", nneg: true },
  { key: "loan_count", label: "Loan count", fmt: "count" },
];

/** Funded cohort view: a static-pool PROGRESSION (how a cohort — a source
 * portfolio ± origination vintage — seasons across reporting periods) plus a
 * point-in-time cohort composition table across a selectable lens (vintage,
 * borrower age, LTV band or origination channel). Replaces the old
 * single-snapshot vintage bar (redundant with the table's balance column). */
function CohortView({ client, portfolioId }: { client: AgentClient; portfolioId: string }) {
  const [grain, setGrain] = useState<CohortGrain>("Y");
  const [lens, setLens] = useState("total");
  const [vintage, setVintage] = useState("");
  const [metric, setMetric] = useState("funded_balance");
  const [dimension, setDimension] = useState<CohortDimension>("vintage");
  const [lenses, setLenses] = useState<SourcePortfolioLens[]>([]);
  const [cohorts, setCohorts] = useState<CohortAnalysis | null>(null);
  const [composition, setComposition] = useState<CohortAnalysis | null>(null);
  const [prog, setProg] = useState<CohortProgression | null>(null);

  useEffect(() => {
    let c = false;
    Promise.resolve(client.getSourcePortfolios())
      .then((r) => { if (!c && r) setLenses(r.lenses ?? []); }).catch(() => {});
    return () => { c = true; };
  }, [client]);

  // Vintage cohorts feed the progression's vintage selector (always vintage).
  useEffect(() => {
    let c = false;
    Promise.resolve(client.getCohorts(portfolioId, grain))
      .then((r) => { if (!c) setCohorts(r); }).catch(() => {});
    return () => { c = true; };
  }, [client, portfolioId, grain]);

  // The composition table follows the selected cohort dimension (deduped with
  // the vintage fetch above when dimension === "vintage").
  useEffect(() => {
    let c = false;
    Promise.resolve(client.getCohorts(portfolioId, grain, dimension))
      .then((r) => { if (!c) setComposition(r); }).catch(() => {});
    return () => { c = true; };
  }, [client, portfolioId, grain, dimension]);

  useEffect(() => {
    let c = false;
    Promise.resolve(client.getCohortProgression(portfolioId,
      { lens, vintage: vintage || undefined, grain }))
      .then((r) => { if (!c) setProg(r); }).catch(() => {});
    return () => { c = true; };
  }, [client, portfolioId, lens, vintage, grain]);

  // Progression's vintage options come from the vintage fetch; the composition
  // table follows the selected dimension.
  const vintageOpts = (cohorts?.cohorts ?? [])
    .map((r) => r.vintage ?? r.cohort).filter((v): v is string => !!v && v !== "Unknown");
  const rows = composition?.cohorts ?? [];
  const cmetrics = new Set(composition?.metricsAvailable ?? []);
  const dimLabel = composition?.dimensionLabel ?? "Vintage";
  const DIM_LABELS: Record<CohortDimension, string> = {
    vintage: "Vintage", age: "Borrower age", ltv: "LTV band", channel: "Origination channel",
  };
  const dimOptions = (composition?.availableDimensions ?? cohorts?.availableDimensions ?? ["vintage"])
    .map((d) => ({ value: d, label: DIM_LABELS[d] }));
  const hasNneg = (prog?.metricsAvailable ?? []).includes("nneg_headroom_pct");
  const metricOpts = _PROG_METRICS.filter((m) => !m.nneg || hasNneg);
  const pm = metricOpts.find((m) => m.key === metric) ?? metricOpts[0];
  const progData = (prog?.periods ?? []).map((p) => ({ period: p.period, value: p.metrics?.[pm.key] ?? null }));
  const scope = (prog?.lens ?? "Total") + (vintage ? `, ${vintage} vintage` : "");
  const portfolioOptions = [
    { value: "total", label: "Consolidated (Total)" },
    ...lenses.filter((l) => l.id !== "total").map((l) => ({ value: l.id, label: l.label })),
  ];

  return (
    <div className="space-y-3" data-testid="cohorts-view">
      {/* Cohort selector: source portfolio × vintage × grain × metric. */}
      <div className="flex flex-wrap items-end gap-3 rounded-xl border border-[var(--color-line)] bg-navy-900/40 px-3 py-2.5">
        <CohortSelect label="Portfolio" value={lens} onChange={setLens} options={portfolioOptions} testId="cohort-lens" />
        <CohortSelect label="Vintage" value={vintage} onChange={setVintage}
          options={[{ value: "", label: "All vintages" }, ...vintageOpts.map((v) => ({ value: v, label: v }))]}
          testId="cohort-vintage" />
        <CohortSelect label="Grain" value={grain} onChange={(v) => setGrain(v as CohortGrain)}
          options={[{ value: "Y", label: "Year" }, { value: "Q", label: "Quarter" }, { value: "M", label: "Month" }]}
          testId="cohort-grain" />
        <CohortSelect label="Metric" value={pm.key} onChange={setMetric}
          options={metricOpts.map((m) => ({ value: m.key, label: m.label }))} testId="cohort-metric" />
      </div>

      {/* Static-pool progression (the hero visual). */}
      {prog && !prog.available ? (
        <div className="rounded-lg border border-amber-400/20 bg-amber-400/5 px-3 py-2 text-[11px] text-amber-300/90"
          data-testid="cohort-progression-unavailable">
          No progression for {scope}{prog.reason ? ` — ${prog.reason}` : ""}. A static pool needs
          at least two reporting periods of matching loans.
        </div>
      ) : (
        <EvoLineChart title={`${pm.label} — ${scope} (static pool)`} data={progData}
          lines={[{ key: "value", label: pm.label }]} valueFormat={pm.fmt}
          source={prog?.lineage?.source as string | undefined} />
      )}

      {/* Cohort metrics by reporting period. */}
      {prog?.available && prog.periods.length > 0 && (
        <div className="overflow-x-auto rounded-xl border border-[var(--color-line)] bg-navy-900/40">
          <table className="w-full text-[11px]" data-testid="cohort-progression-table">
            <thead>
              <tr className="border-b border-[var(--color-line)] text-ink-400">
                <th className="px-3 py-2 text-left font-medium">Period</th>
                <th className="px-3 py-2 text-right font-medium">Loans</th>
                <th className="px-3 py-2 text-right font-medium">Balance</th>
                <th className="px-3 py-2 text-right font-medium">WA LTV</th>
                <th className="px-3 py-2 text-right font-medium">WA rate</th>
                {hasNneg && <th className="px-3 py-2 text-right font-medium">NNEG headroom</th>}
              </tr>
            </thead>
            <tbody>
              {prog.periods.map((p) => (
                <tr key={p.period} className="border-b border-[var(--color-line-soft)] last:border-0">
                  <td className="px-3 py-1.5 text-left font-medium text-ink-100">{p.period}</td>
                  <td className="px-3 py-1.5 text-right text-ink-200">{(p.loanCount ?? 0).toLocaleString("en-GB")}</td>
                  <td className="px-3 py-1.5 text-right text-ink-200">{p.metrics?.funded_balance != null ? gbpCompact(p.metrics.funded_balance) : "—"}</td>
                  <td className="px-3 py-1.5 text-right text-ink-200">{pctFmt(p.metrics?.wa_ltv)}</td>
                  <td className="px-3 py-1.5 text-right text-ink-200">{pctFmt(p.metrics?.wa_interest_rate)}</td>
                  {hasNneg && <td className="px-3 py-1.5 text-right text-ink-200">{pctFmt(p.metrics?.nneg_headroom_pct)}</td>}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Point-in-time static-pool composition by the selected dimension. */}
      <div className="flex flex-wrap items-end justify-between gap-2">
        <div className="text-[11px] font-semibold text-ink-300">
          Cohort composition — by {dimLabel.toLowerCase()} (as of {composition?.reportingDate ?? "latest"})
        </div>
        <CohortSelect label="Cohort by" value={dimension}
          onChange={(v) => setDimension(v as CohortDimension)}
          options={dimOptions} testId="cohort-dimension" />
      </div>
      {composition && !composition.available ? (
        <div className="rounded-lg border border-amber-400/20 bg-amber-400/5 px-3 py-2 text-[11px] text-amber-300/90"
          data-testid="cohorts-unavailable">
          No {dimLabel.toLowerCase()} composition for this run{composition.reason ? ` — ${composition.reason}` : ""}.
        </div>
      ) : (
        <div className="overflow-x-auto rounded-xl border border-[var(--color-line)] bg-navy-900/40">
          <table className="w-full text-[11px]" data-testid="cohorts-table">
            <thead>
              <tr className="border-b border-[var(--color-line)] text-ink-400">
                <th className="px-3 py-2 text-left font-medium">{dimLabel}</th>
                <th className="px-3 py-2 text-right font-medium">Loans</th>
                {cmetrics.has("balance") && <th className="px-3 py-2 text-right font-medium">Balance</th>}
                {cmetrics.has("balance") && <th className="px-3 py-2 text-right font-medium">Book share</th>}
                {cmetrics.has("waLtv") && <th className="px-3 py-2 text-right font-medium">WA LTV</th>}
                {cmetrics.has("waRate") && <th className="px-3 py-2 text-right font-medium">WA rate</th>}
                {cmetrics.has("waMonthsOnBook") && <th className="px-3 py-2 text-right font-medium">WA MOB</th>}
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => (
                <tr key={r.cohort} className="border-b border-[var(--color-line-soft)] last:border-0">
                  <td className="px-3 py-1.5 text-left font-medium text-ink-100">{r.cohort}</td>
                  <td className="px-3 py-1.5 text-right text-ink-200">{r.loanCount.toLocaleString("en-GB")}</td>
                  {cmetrics.has("balance") && <td className="px-3 py-1.5 text-right text-ink-200">{r.balance != null ? gbpCompact(r.balance) : "—"}</td>}
                  {cmetrics.has("balance") && <td className="px-3 py-1.5 text-right text-ink-400">{r.sharePct != null ? `${r.sharePct.toFixed(1)}%` : "—"}</td>}
                  {cmetrics.has("waLtv") && <td className="px-3 py-1.5 text-right text-ink-200">{pctFmt(r.waLtv)}</td>}
                  {cmetrics.has("waRate") && <td className="px-3 py-1.5 text-right text-ink-200">{pctFmt(r.waRate)}</td>}
                  {cmetrics.has("waMonthsOnBook") && <td className="px-3 py-1.5 text-right text-ink-200">{mobFmt(r.waMonthsOnBook)}</td>}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      <p className="text-[10px] text-ink-500">
        Static-pool seasoning across governed reporting periods (balance / LTV / rate / NNEG) for the
        selected cohort, plus point-in-time cohort composition across the chosen lens. Source: governed
        funded central lender tape.
      </p>
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
  // Origination funnel: overlay the stock line on the weekly-flow bars.
  const [showCumulative, setShowCumulative] = useState(false);
  // Which origination stage (if any) is enlarged in the focus modal.
  const [expandedStage, setExpandedStage] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    const run = async () => {
      try {
        if (view === "funded") setFunded(await client.getFundedEvolution(portfolioId));
        else if (view === "pipeline") setPipeline(await client.getPipelineEvolution(portfolioId));
        else if (view === "origination") setFunnel(await client.getFunnelEvolution(portfolioId));
        else if (view === "cohorts") { /* CohortView is self-contained (owns its fetches) */ }
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
      // Canonical conversion: cumulative cohort progression — % of the original
      // KFI cohort reaching each milestone (KFI → Application → Offer → Funded)
      // by each week. A true cohort funnel, so the lines nest and show leakage.
      const prog = funnel?.cohortProgression;
      if (prog?.weeks?.length) {
        const data = prog.weeks.map((w, i) => {
          const row: Record<string, number | string> = { period: w };
          for (const st of prog.stages) row[st] = prog.series[st]?.[i] ?? 0;
          return row;
        });
        return { data, lines: prog.stages, format: "pct" as const };
      }
      // Fallback only when no cohort funnel is available (single/thin history).
      const conv = stageConversionSeries(rows, funnel?.conversionLagWeeks);
      return { data: conv.data, lines: conv.stages, format: "pct" as const };
    }
    const piv = pivotStage(rows, stageMode === "count" ? "count" : "value");
    const lines = funnelLineStages(piv.stages, includeKfi);
    return { data: piv.data, lines,
      format: (stageMode === "count" ? "count" : "gbp") as "count" | "gbp" };
  }, [pipeline, funnel, stageMode, includeKfi]);
  const hasWithdrawn = useMemo(
    () => (pipeline?.byStage ?? []).some((r) => normaliseStage(r.stage) === "WITHDRAWN"),
    [pipeline],
  );
  // Latest cumulative cohort % for a stage — the canonical conversion figure.
  const cohortPctFor = (stage: string): number | null => {
    const arr = funnel?.cohortProgression?.series?.[stage];
    return arr && arr.length ? arr[arr.length - 1] : null;
  };
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
          className="inline-flex items-center gap-1 rounded-lg border border-navy-600 bg-navy-950/80 p-1 ring-1 ring-inset ring-white/5">
          {(["funded", "pipeline", "origination", "cohorts", "forecast"] as EvoView[]).map((v) => (
            <button key={v} type="button" role="tab" aria-selected={view === v}
              onClick={() => setView(v)}
              className={cn(
                "rounded-md px-3 py-1 text-[12px] font-medium transition-all",
                view === v
                  ? "bg-peri-400/20 text-ink-100 ring-1 ring-inset ring-peri-400/50"
                  : "cursor-pointer bg-navy-800/70 text-ink-300 ring-1 ring-inset ring-white/5 hover:bg-navy-700 hover:text-ink-100",
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
                : "Cumulative cohort conversion (KFI → Application → Offer → Funded)"}
              data={stageChart.data}
              lines={stageChart.lines.map((s) => ({ key: s, label: STAGE_LABEL[normaliseStage(s)] ?? s }))}
              valueFormat={stageChart.format === "pct" ? "pct_points" : stageChart.format}
              source="weekly pipeline extracts" />
            <p className="mt-1 text-[10px] text-ink-500" data-testid="stage-mode-note">
              {stageMode === "conversion"
                ? (funnel?.cohortProgression?.weeks?.length
                    ? `Conversion = cumulative % of the original KFI cohort${funnel.cohortProgression.cohortSize ? ` (${funnel.cohortProgression.cohortSize.toLocaleString()} cases)` : ""} that has reached each milestone to date. Cohort-tracked, so the lines nest (Funded ≤ Offer ≤ Application ≤ KFI) and show where the pipeline leaks — not point-in-time stage stocks.`
                    : "Conversion = cumulative % of the KFI cohort reaching each milestone to date. Cohort history is thin for this book, so a lagged stock-ratio approximation is shown until more weeks accrue.")
                : `${stageMode === "amount" ? "Amount (£)" : "Case count"} for the main funnel`
                  + ` (KFI → Application → Offer → Completion).${hasWithdrawn ? " Withdrawn is tracked separately, not in the funnel." : ""}`
                  + " Toggle 'Include KFI' off to read the smaller downstream stages."}
            </p>
          </div>
        </div>
      )}

      {view === "origination" && (
        <div className="space-y-3" data-testid="origination-funnel">
          <div className="flex flex-wrap items-center justify-end gap-2">
            <MethodologyDisclosure />
            <label className="flex shrink-0 items-center gap-1.5 rounded-md border border-[var(--color-line)] bg-navy-900/60 px-2 py-1 text-[10px] text-ink-300">
              <input type="checkbox" checked={showCumulative}
                onChange={(e) => setShowCumulative(e.target.checked)}
                aria-label="Show stock line" />
              Show stock line
            </label>
          </div>
          <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
            {orderStages(funnel?.stages ?? []).map((stage) => (
              <FunnelStageCard key={stage} stage={stage}
                label={funnel?.stageLabels?.[stage] ?? stage}
                points={funnel?.series?.[stage] ?? []}
                flowPoints={funnel?.flowSeries?.[stage] ?? []}
                summary={funnel?.summary?.[stage]}
                conversion={funnel?.summary?.[stage]?.conversion ?? null}
                cohortPct={cohortPctFor(stage)}
                showCumulative={showCumulative}
                onExpand={() => setExpandedStage(stage)} />
            ))}
          </div>
          {funnel?.sourceFiles?.length ? (
            <p className="text-[10px] text-ink-500">
              Source: {funnel.uniqueWeeklyExtractsUsed ?? funnel.sourceFiles.length} governed
              weekly extract(s).
            </p>
          ) : null}
          {expandedStage && funnel && (
            <ChartFocusModal
              title={`${funnel.stageLabels?.[expandedStage] ?? expandedStage} · weekly origination flow`}
              onClose={() => setExpandedStage(null)}
            >
              <FunnelStageCard stage={expandedStage}
                label={funnel.stageLabels?.[expandedStage] ?? expandedStage}
                points={funnel.series?.[expandedStage] ?? []}
                flowPoints={funnel.flowSeries?.[expandedStage] ?? []}
                summary={funnel.summary?.[expandedStage]}
                conversion={funnel.summary?.[expandedStage]?.conversion ?? null}
                cohortPct={cohortPctFor(expandedStage)}
                showCumulative={showCumulative}
                large />
            </ChartFocusModal>
          )}
        </div>
      )}

      {view === "cohorts" && <CohortView client={client} portfolioId={portfolioId} />}

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
