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
    <div className={cn("rounded-lg border border-[var(--color-line-soft)] bg-navy-850/60 p-3.5", dim && "opacity-60")}>
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
}: {
  snapshot: FundedSnapshot | null;
  loading?: boolean;
}) {
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
          <span className="inline-flex items-center gap-1.5 rounded-md border border-[var(--color-line-soft)] bg-navy-900/60 px-2.5 py-1 text-[11px] font-medium text-ink-200">
            <CalendarDays size={13} className="text-peri-300" />
            Reporting Date · {reporting}
          </span>
          <span className="text-[10px] text-ink-500">
            {prior ? `vs prior run · ${prior.reporting_date ?? prior.run_id}` : "No prior reporting date available"}
          </span>
        </div>
      </div>

      <div className="mt-4 grid grid-cols-2 gap-3 lg:grid-cols-3 xl:grid-cols-4">
        {snapshot.kpis.map((kpi) => (
          <KpiTile key={kpi.id} kpi={kpi} />
        ))}
      </div>

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
