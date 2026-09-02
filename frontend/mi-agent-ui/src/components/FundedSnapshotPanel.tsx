import { useState } from "react";
import {
  AlertTriangle,
  ArrowDownRight,
  ArrowUpRight,
  CalendarDays,
  ChevronDown,
  Landmark,
  Minus,
} from "lucide-react";
import type { FundedSnapshot, SnapshotKPI } from "@/domain";
import { BarList, MeasureToggle, BAR_MEASURE_LABEL, BAR_MEASURE_FORMAT,
  type BarDatum, type BarMeasure } from "@/components/pipeline/bits";
import { cleanBucketLabel, sortStratBars } from "@/lib/stratOrder";
import { cn, formatDate } from "@/lib/utils";

function deltaColour(intent?: SnapshotKPI["deltaIntent"]) {
  return intent === "positive"
    ? "text-mint-400"
    : intent === "negative"
      ? "text-rose-400"
      : "text-ink-400";
}

function KpiTile({ kpi }: { kpi: SnapshotKPI }) {
  const Icon = kpi.deltaIntent === "positive" ? ArrowUpRight : kpi.deltaIntent === "negative" ? ArrowDownRight : Minus;
  const dim = kpi.available === false;
  return (
    // Elevated tile: a step lighter than the panel behind it, with a visible
    // border + top highlight so the KPI grid reads as raised cards.
    <div
      className={cn(
        "rounded-lg border border-navy-600/70 bg-navy-800/80 p-3.5 shadow-sm",
        "shadow-[inset_0_1px_0_rgba(255,255,255,0.05)]",
        dim && "opacity-60",
      )}
    >
      <div className="text-[11px] font-medium uppercase tracking-wider text-ink-400">{kpi.label}</div>
      <div className="mt-1.5 font-mono text-2xl font-semibold tabular-nums text-ink-100">{kpi.value}</div>
      <div className="mt-1.5 flex items-center gap-1.5">
        {kpi.delta && (
          <span className={cn("inline-flex items-center gap-0.5 text-xs font-medium", deltaColour(kpi.deltaIntent))}>
            <Icon size={13} strokeWidth={2.5} />
            {kpi.delta}
          </span>
        )}
        {kpi.hint && <span className="text-[11px] text-ink-500">{kpi.hint}</span>}
      </div>
    </div>
  );
}

/**
 * The deterministic funded-portfolio snapshot shown on the landing page BEFORE
 * any AI query. Clearly labelled as funded-book MI (not the origination pipeline).
 */
export function FundedSnapshotPanel({
  snapshot,
  loading,
  onDrill,
}: {
  snapshot: FundedSnapshot | null;
  loading?: boolean;
  /** Selecting a band. The handler owns what a selection means; this panel
   *  never derives a population itself. */
  onDrill?: (dimension: string, band: string) => void;
}) {
  const [measure, setMeasure] = useState<BarMeasure>("balance");
  const [showDiagnostics, setShowDiagnostics] = useState(false);

  if (loading && !snapshot) {
    return (
      <section className="rounded-xl border border-[var(--color-line)] bg-navy-900/40 p-5">
        <div className="h-4 w-48 animate-pulse rounded bg-navy-700/60" />
        <div className="mt-4 grid grid-cols-2 gap-3 lg:grid-cols-3">
          {Array.from({ length: 6 }).map((_, i) => (
            <div key={i} className="h-20 animate-pulse rounded-lg bg-navy-800/50" />
          ))}
        </div>
      </section>
    );
  }

  if (!snapshot) return null;

  if (!snapshot.ok) {
    return (
      <section className="rounded-xl border border-amber-400/20 bg-amber-400/5 p-5 text-[13px] text-amber-300/90">
        <div className="flex items-center gap-2 font-medium">
          <AlertTriangle size={15} /> Funded Book Snapshot unavailable
        </div>
        <p className="mt-1 text-amber-300/70">{snapshot.error ?? "No funded data for this reporting date."}</p>
      </section>
    );
  }

  const { portfolio, prior } = snapshot;
  const reporting = portfolio.reporting_date ? formatDate(portfolio.reporting_date) : portfolio.run_id;

  // The headline Balance / Loans tiles already carry the month-on-month delta;
  // the separate "Monthly change" tiles then repeat the same numbers. Hide the
  // duplicates only when the headline tile really shows the delta, so a payload
  // without headline deltas keeps its explicit monthly-change tiles.
  const byId = new Map(snapshot.kpis.map((k) => [k.id, k]));
  const kpis = snapshot.kpis.filter((k) => {
    if (k.id === "mom_balance") return !byId.get("balance")?.delta;
    if (k.id === "mom_loans") return !byId.get("loans")?.delta;
    return true;
  });

  return (
    <section className="rounded-xl border border-[var(--color-line)] bg-navy-900/40 p-5">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="flex items-center gap-2.5">
          <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-navy-700/70 text-peri-300">
            <Landmark size={17} />
          </div>
          <div className="leading-tight">
            <h2 className="text-sm font-semibold text-ink-100">Funded Book Snapshot</h2>
            <p className="text-[11px] text-ink-400">
              Funded Portfolio · <span className="font-medium text-ink-300">{portfolio.label}</span>
            </p>
          </div>
        </div>
        <div className="flex flex-col items-end gap-1">
          <span className="inline-flex items-center gap-2">
            {/* Stale-while-loading: keep the current figures visible but say a
                new selection is being computed (backend runs can take a while). */}
            {loading && (
              <span
                data-testid="snapshot-updating"
                className="inline-flex items-center gap-1.5 rounded-md border border-peri-400/30 bg-peri-400/10 px-2 py-1 text-[10px] font-medium text-peri-200"
              >
                <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-peri-300" />
                Updating for new selection…
              </span>
            )}
            <span className="inline-flex items-center gap-1.5 rounded-md border border-[var(--color-line-soft)] bg-navy-900/60 px-2.5 py-1 text-[11px] font-medium text-ink-200">
              <CalendarDays size={13} className="text-peri-300" />
              Reporting Date · {reporting}
            </span>
          </span>
          <span className="text-[10px] text-ink-500">
            {prior ? `vs prior run · ${prior.reporting_date ?? prior.run_id}` : "No prior reporting date available"}
          </span>
        </div>
      </div>

      <div className="mt-4 grid grid-cols-2 gap-3 lg:grid-cols-3 xl:grid-cols-4">
        {kpis.map((kpi) => (
          <KpiTile key={kpi.id} kpi={kpi} />
        ))}
      </div>

      {(snapshot.stratifications?.length ?? 0) > 0 && (
        <div className="mt-4">
          <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
            <div className="text-[11px] font-semibold uppercase tracking-wider text-ink-400">
              Stratifications · {BAR_MEASURE_LABEL[measure].toLowerCase()} by dimension
            </div>
            {/* Presentation-only view switch on the shared bar-list seam.
                Balance, share and count are all already in the stratification
                payload — nothing is recomputed in the browser. */}
            <MeasureToggle
              measures={["balance", "share", "count"]}
              active={measure}
              onChange={setMeasure}
              label="Stratification measure"
              testIdPrefix="strat-measure"
            />
          </div>
          <div className="grid grid-cols-1 gap-3 md:grid-cols-2 xl:grid-cols-3">
            {snapshot.stratifications!.map((s) => {
              // Natural bucket order (LTV %, age, vintage year, rate… else
              // alphabetical; Unknown last) + label tidy-ups ("2008.0" → "2008").
              const data: BarDatum[] = sortStratBars(s.bars).map((b) => ({
                label: cleanBucketLabel(b.label),
                // The selected measure, read straight from the payload.
                value: measure === "balance" ? b.balance
                  : measure === "share" ? b.sharePct
                  : b.count,
                count: b.count,
              }));
              // The backend decides whether a dimension is available, entirely
              // null, not supplied for these portfolios, or only partially
              // covered. Each state renders distinctly — a chart area is never
              // simply left blank.
              const state = s.availability ?? (data.length ? "available" : "not_supplied");
              const drawable = data.length > 0;
              return (
                <div key={s.key}
                  data-testid={`strat-${s.key}`}
                  data-availability={state}
                  className="rounded-lg border border-navy-600/60 bg-navy-800/50 p-3 shadow-sm">
                  <div className="mb-2 flex items-baseline justify-between gap-2">
                    <span className="text-[11px] font-medium text-ink-300">{s.label}</span>
                    {state === "partially_available" && (
                      <span className="rounded-full border border-amber-400/30 bg-amber-400/10 px-1.5 py-0.5 text-[9px] font-medium uppercase tracking-wider text-amber-200/90">
                        Partial
                      </span>
                    )}
                  </div>
                  {drawable ? (
                    <BarList
                      data={data}
                      format={BAR_MEASURE_FORMAT[measure]}
                      onSelect={onDrill && ((label) => onDrill(s.label, label))}
                      selectTitle={(label) => `Ask the MI engine about ${label}`}
                    />
                  ) : (
                    <p className="py-3 text-[11px] leading-relaxed text-ink-500">
                      {s.reason ?? "Not available for the selected portfolios."}
                    </p>
                  )}
                  {drawable && s.reason && (
                    <p className="mt-2 text-[10px] leading-relaxed text-ink-500">{s.reason}</p>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      )}

      {snapshot.warnings.length > 0 && (
        <div className="mt-3 rounded-lg border border-amber-400/20 bg-amber-400/5 px-3 py-2 text-[11px] text-amber-300/90">
          {snapshot.warnings.map((w, i) => (
            <div key={i}>⚠ {w}</div>
          ))}
        </div>
      )}

      {snapshot.diagnostics.length > 0 && (
        <div className="mt-3">
          <button
            type="button"
            onClick={() => setShowDiagnostics((s) => !s)}
            className="inline-flex items-center gap-1.5 text-[11px] font-medium text-ink-500 hover:text-ink-300"
          >
            <ChevronDown size={13} className={cn("transition-transform", !showDiagnostics && "-rotate-90")} />
            Technical details ({snapshot.diagnostics.length})
          </button>
          {showDiagnostics && (
            <ul className="mt-1.5 list-disc space-y-0.5 rounded-lg border border-[var(--color-line-soft)] bg-navy-900/60 px-5 py-2 text-[11px] text-ink-400">
              {snapshot.diagnostics.map((d, i) => (
                <li key={i}>{d}</li>
              ))}
            </ul>
          )}
        </div>
      )}
    </section>
  );
}
