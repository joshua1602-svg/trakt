import { Bell, CalendarDays, RefreshCw, Settings, ShieldCheck } from "lucide-react";
import { PortfolioSelector } from "@/components/PortfolioSelector";
import { DeckDownloadMenu } from "@/components/DeckDownloadMenu";
import type { AgentClient } from "@/api";
import type { SnapshotPortfolio, SnapshotRun } from "@/domain";
import {
  type UserIdentity,
  canSeeAdminControls,
  displayInitials,
  formatDisplayName,
  roleLabel,
} from "@/lib/identity";

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
  client,
  portfolioId,
  reportingPeriod,
  identity,
  onRefresh,
  refreshing,
}: {
  portfolios: SnapshotPortfolio[];
  runs: SnapshotRun[];
  selectedClientId: string | null;
  selectedRunId: string | null;
  onPortfolioChange: (clientId: string) => void;
  onRunChange: (runId: string) => void;
  mock: boolean;
  client: AgentClient;
  portfolioId: string;
  reportingPeriod?: string | null;
  identity: UserIdentity | null;
  onRefresh?: () => void;
  refreshing?: boolean;
}) {
  // Role-based visibility: operators/admins may see settings + notifications;
  // client users get a clean, read-only MI surface (fail-closed when unknown).
  const showAdminControls = canSeeAdminControls(identity);
  const displayName = formatDisplayName(identity?.user);
  const role = roleLabel(identity);

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

      <div className="ml-auto flex items-center gap-2.5">
        {/* Environment badge: the mock/staging warning stays prominent (it means
            the data is not real); Production is a quiet, low-key marker. */}
        {mock ? (
          <span className="inline-flex items-center gap-1.5 rounded-full border border-amber-400/30 bg-amber-400/10 px-2.5 py-1 text-[11px] font-medium text-amber-400">
            <ShieldCheck size={13} /> Staging · Mock Data
          </span>
        ) : (
          <span
            title="Production environment"
            className="inline-flex items-center gap-1 text-[10px] font-medium uppercase tracking-wider text-ink-500"
          >
            <span className="h-1.5 w-1.5 rounded-full bg-mint-400" /> Prod
          </span>
        )}

        {/* Investor deck download (top-right actions). */}
        <DeckDownloadMenu client={client} portfolioId={portfolioId} reportingPeriod={reportingPeriod} />

        {/* Manual refresh — clears the client cache + reloads the active data. */}
        {onRefresh && (
          <button
            type="button"
            onClick={onRefresh}
            aria-label="Refresh data"
            title="Refresh MI data"
            data-testid="refresh-button"
            className="inline-flex h-8 w-8 items-center justify-center rounded-md text-ink-400 hover:bg-navy-800 hover:text-ink-100"
          >
            <RefreshCw size={15} className={refreshing ? "animate-spin" : undefined} />
          </button>
        )}

        {/* Settings / notifications are operator/admin-only (hidden for clients). */}
        {showAdminControls && (
          <div className="flex items-center gap-0.5">
            <button
              type="button"
              aria-label="Notifications"
              className="relative inline-flex h-8 w-8 items-center justify-center rounded-md text-ink-400 hover:bg-navy-800 hover:text-ink-100"
            >
              <Bell size={16} />
            </button>
            <button
              type="button"
              aria-label="Settings"
              className="inline-flex h-8 w-8 items-center justify-center rounded-md text-ink-400 hover:bg-navy-800 hover:text-ink-100"
            >
              <Settings size={16} />
            </button>
          </div>
        )}

        {/* Signed-in identity (Entra-derived). Shown only when authenticated —
            no hardcoded fallback name. */}
        {displayName && (
          <div className="flex items-center gap-2 rounded-lg border border-[var(--color-line)] bg-navy-900/60 py-1 pl-1 pr-3" data-testid="user-identity">
            <div className="flex h-7 w-7 items-center justify-center rounded-md bg-gradient-to-br from-peri-400 to-navy-600 text-[11px] font-bold text-navy-950">
              {displayInitials(identity?.user)}
            </div>
            <div className="leading-tight">
              <div className="text-[12px] font-medium text-ink-100">{displayName}</div>
              {role && <div className="text-[10px] text-ink-500">{role}</div>}
            </div>
          </div>
        )}
      </div>
    </header>
  );
}
