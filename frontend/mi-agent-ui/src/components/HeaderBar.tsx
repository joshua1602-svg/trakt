import { Bell, CalendarDays, Settings, ShieldCheck } from "lucide-react";
import { PortfolioSelector } from "@/components/PortfolioSelector";
import type { SnapshotPortfolio, SnapshotRun } from "@/domain";

function formatDate(iso: string): string {
  return new Date(iso).toLocaleDateString("en-GB", {
    day: "2-digit",
    month: "short",
    year: "numeric",
  });
}

function runLabel(run: SnapshotRun): string {
  return run.reporting_date ? formatDate(run.reporting_date) : run.run_id;
}

export function HeaderBar({
  portfolios,
  runs,
  selectedClientId,
  selectedRunId,
  onPortfolioChange,
  onRunChange,
  mock,
}: {
  portfolios: SnapshotPortfolio[];
  runs: SnapshotRun[];
  selectedClientId: string | null;
  selectedRunId: string | null;
  onPortfolioChange: (clientId: string) => void;
  onRunChange: (runId: string) => void;
  mock: boolean;
}) {
  return (
    <header className="flex h-14 shrink-0 items-center gap-4 border-b border-[var(--color-line)] bg-navy-900/70 px-5 backdrop-blur">
      {/* Brand */}
      <div className="flex items-center gap-2.5">
        <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-navy-700">
          <span className="font-mono text-sm font-bold text-peri-300">T</span>
        </div>
        <div className="leading-tight">
          <div className="text-sm font-semibold tracking-tight text-ink-100">
            Trakt <span className="font-normal text-ink-400">· MI Agent</span>
          </div>
          <div className="text-[10px] uppercase tracking-wider text-ink-500">
            Funded Portfolio Intelligence
          </div>
        </div>
      </div>

      <div className="mx-1 h-7 w-px bg-[var(--color-line)]" />

      <PortfolioSelector
        portfolios={portfolios}
        value={selectedClientId}
        onChange={onPortfolioChange}
      />

      {/* Reporting date — only runs that actually exist for this portfolio. */}
      <label className="flex items-center gap-2 rounded-lg border border-[var(--color-line)] bg-navy-900/60 px-3 py-1.5">
        <CalendarDays size={15} className="text-peri-300" />
        <span className="text-[10px] uppercase tracking-wider text-ink-500">Reporting Date</span>
        <select
          value={selectedRunId ?? ""}
          onChange={(e) => onRunChange(e.target.value)}
          disabled={runs.length === 0}
          className="cursor-pointer bg-transparent text-[13px] font-medium text-ink-100 focus:outline-none disabled:opacity-50"
        >
          {runs.length === 0 && <option value="">No reporting runs</option>}
          {runs.map((r) => (
            <option key={r.run_id} value={r.run_id} className="bg-navy-900 text-ink-100">
              {runLabel(r)}
            </option>
          ))}
        </select>
      </label>

      <div className="ml-auto flex items-center gap-3">
        <span
          className={
            mock
              ? "inline-flex items-center gap-1.5 rounded-full border border-amber-400/30 bg-amber-400/10 px-2.5 py-1 text-[11px] font-medium text-amber-400"
              : "inline-flex items-center gap-1.5 rounded-full border border-mint-400/30 bg-mint-400/10 px-2.5 py-1 text-[11px] font-medium text-mint-400"
          }
        >
          <ShieldCheck size={13} />
          {mock ? "Staging · Mock Data" : "Production"}
        </span>
        <div className="flex items-center gap-0.5">
          <button
            type="button"
            aria-label="Notifications"
            className="relative inline-flex h-8 w-8 items-center justify-center rounded-md text-ink-400 hover:bg-navy-800 hover:text-ink-100"
          >
            <Bell size={16} />
            <span className="absolute right-1.5 top-1.5 h-1.5 w-1.5 rounded-full bg-rose-400" />
          </button>
          <button
            type="button"
            aria-label="Settings"
            className="inline-flex h-8 w-8 items-center justify-center rounded-md text-ink-400 hover:bg-navy-800 hover:text-ink-100"
          >
            <Settings size={16} />
          </button>
        </div>
        <div className="flex items-center gap-2 rounded-lg border border-[var(--color-line)] bg-navy-900/60 py-1 pl-1 pr-3">
          <div className="flex h-7 w-7 items-center justify-center rounded-md bg-gradient-to-br from-peri-400 to-navy-600 text-[11px] font-bold text-navy-950">
            JA
          </div>
          <div className="leading-tight">
            <div className="text-[12px] font-medium text-ink-100">J. Analyst</div>
            <div className="text-[10px] text-ink-500">Risk &amp; MI</div>
          </div>
        </div>
      </div>
    </header>
  );
}
