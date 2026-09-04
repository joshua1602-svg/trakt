/**
 * Phase 3A — the Weekly Portfolio Brief panel.
 *
 * A restrained list above the existing dashboard. It replaces no chart, changes
 * no layout when disabled, and renders nothing at all unless the flag is on AND
 * the API returned insights — so an environment with the UI flag on and the API
 * flag off looks exactly like today.
 *
 * Every number here came from the API. Nothing is computed in this component:
 * an insight already carries its headline, its summary, its severity and its
 * dates, which is what makes the same object reusable by another channel later.
 */

import { useEffect, useMemo, useState } from "react";
import { AlertTriangle, Check, ChevronDown, Info, TriangleAlert } from "lucide-react";
import type { AgentClient } from "@/api";
import type { Insight, InsightSeverity, WeeklyBrief } from "@/domain";
import { useWeeklyBrief } from "@/hooks/useWeeklyBrief";
import { weeklyBriefEnabled } from "@/lib/featureFlags";
import { cn } from "@/lib/utils";

const SEVERITY_STYLE: Record<InsightSeverity, { chip: string; icon: JSX.Element }> = {
  concern: {
    chip: "bg-[#eb6f6f]/15 text-[#eb6f6f]",
    icon: <TriangleAlert size={12} />,
  },
  attention: {
    chip: "bg-[#e0a458]/15 text-[#e0a458]",
    icon: <AlertTriangle size={12} />,
  },
  info: {
    chip: "bg-[#7c9cf0]/15 text-[#7c9cf0]",
    icon: <Info size={12} />,
  },
};

const SEVERITY_LABEL: Record<InsightSeverity, string> = {
  concern: "Concern", attention: "Attention", info: "Observation",
};

export function InsightCard({ insight, onNavigate }: {
  insight: Insight;
  onNavigate?: (deepLink: string) => void;
}) {
  const style = SEVERITY_STYLE[insight.severity] ?? SEVERITY_STYLE.info;
  const clickable = Boolean(insight.deep_link && onNavigate);
  const go = () => {
    if (insight.deep_link && onNavigate) onNavigate(insight.deep_link);
  };
  return (
    <li
      data-testid="insight-card"
      data-insight-type={insight.insight_type}
      data-severity={insight.severity}
      className={cn(
        "rounded-lg border border-[var(--color-line-soft)] bg-navy-900/40 px-3 py-2",
        clickable && "cursor-pointer hover:border-[var(--color-line)]")}
      {...(clickable
        ? {
          role: "button", tabIndex: 0, onClick: go,
          "aria-label": `${insight.headline} — open the related dashboard section`,
          onKeyDown: (e: React.KeyboardEvent) => {
            if (e.key === "Enter" || e.key === " ") { e.preventDefault(); go(); }
          },
        }
        : {})}
    >
      <div className="flex items-start justify-between gap-3">
        <span className="text-[12px] font-semibold text-ink-100">
          {insight.headline}
        </span>
        <span className={cn(
          "inline-flex shrink-0 items-center gap-1 rounded px-1.5 py-0.5 text-[9px] uppercase tracking-wide",
          style.chip)}>
          {style.icon}{SEVERITY_LABEL[insight.severity] ?? insight.severity}
        </span>
      </div>
      <p className="mt-1 text-[11px] leading-snug text-ink-400">{insight.summary}</p>
      {(insight.as_of_date || insight.comparison_date) && (
        <p className="mt-1 text-[9px] text-ink-500">
          {insight.as_of_date}
          {insight.comparison_date && ` vs ${insight.comparison_date}`}
        </p>
      )}
    </li>
  );
}

/** The omissions, collapsed. Silence has to be explainable on demand. */
function Omissions({ brief }: { brief: WeeklyBrief }) {
  const [open, setOpen] = useState(false);
  const rows = brief.omitted ?? [];
  if (!rows.length) return null;
  return (
    <div className="mt-2" data-testid="weekly-brief-omissions">
      <button type="button" onClick={() => setOpen((o) => !o)} aria-expanded={open}
        className="flex items-center gap-1 text-[10px] text-ink-500 hover:text-ink-300">
        <ChevronDown size={11} className={cn("transition-transform", !open && "-rotate-90")} />
        {rows.length} item{rows.length === 1 ? "" : "s"} not shown
      </button>
      {open && (
        <ul className="mt-1 space-y-0.5 pl-4">
          {rows.map((o) => (
            <li key={`${o.insight_type}-${o.category}`} className="text-[10px] text-ink-500">
              <span className="text-ink-400">{o.insight_type}</span> — {o.reason}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

const DISMISSED_KEY_PREFIX = "mi.weeklyBrief.dismissedFor.";

/** What makes this week's brief THIS week's brief. A new data upload changes
 *  at least one of these dates, so it is never mistaken for an already-read
 *  one — the reader sees every new brief exactly once. */
function briefIdentity(brief: WeeklyBrief): string {
  return [brief.as_of_date, brief.comparison_date, brief.source_dates?.funded_as_of]
    .filter(Boolean).join("|");
}

function readDismissed(portfolioId: string): string | null {
  try {
    return typeof localStorage === "undefined"
      ? null : localStorage.getItem(DISMISSED_KEY_PREFIX + portfolioId);
  } catch {
    return null; // private browsing / storage blocked — never break the panel
  }
}

function writeDismissed(portfolioId: string, identity: string): void {
  try {
    if (typeof localStorage !== "undefined") {
      localStorage.setItem(DISMISSED_KEY_PREFIX + portfolioId, identity);
    }
  } catch {
    /* per-viewer convenience only — losing it just re-shows the brief */
  }
}

export function WeeklyBriefPanel({
  client, portfolioId, portfolioContext, asOf, onNavigate, enabled, ready = true,
}: {
  client: AgentClient;
  portfolioId: string;
  portfolioContext?: string;
  asOf?: string;
  /** Scrolls or switches to the existing dashboard section. No routing. */
  onNavigate?: (deepLink: string) => void;
  /** Injectable so a test does not need the bundler's env. */
  enabled?: boolean;
  /**
   * Whether the dashboard's primary data has resolved. The brief summarises
   * that same data, so fetching before the dashboard has it means paying for
   * the same cold work twice, concurrently. Defaults to true.
   */
  ready?: boolean;
}) {
  const on = useMemo(
    () => (enabled === undefined ? weeklyBriefEnabled() : enabled), [enabled]);
  const { brief, loading, unavailable } = useWeeklyBrief({
    client, portfolioId, portfolioContext, asOf, enabled: on, ready,
  });

  // Read once this brief's identity is known, and again whenever a NEW brief
  // (a new data upload) arrives with a different one — so a stale "dismissed"
  // flag from last week's brief never hides this week's.
  const identity = brief ? briefIdentity(brief) : null;
  const [dismissedId, setDismissedId] = useState<string | null>(null);
  useEffect(() => {
    if (identity) setDismissedId(readDismissed(portfolioId));
  }, [portfolioId, identity]);
  const dismissed = Boolean(identity) && identity === dismissedId;

  // Nothing to say and nothing to explain -> render nothing, so a disabled or
  // unsupported environment is visually identical to today.
  if (!on) return null;
  if (unavailable && !loading) return null;
  if (dismissed) return null;

  const insights = brief?.insights ?? [];
  if (!loading && !insights.length && !(brief?.omitted ?? []).length) return null;

  return (
    <section data-testid="weekly-brief"
      className="mx-6 mt-5 rounded-2xl border border-[var(--color-line)] bg-[var(--surface-brief)] p-4">
      <header className="mb-2 flex items-baseline justify-between gap-3">
        <h3 className="text-[13px] font-semibold text-ink-100">Weekly Portfolio Brief</h3>
        <div className="flex items-center gap-3">
          {brief?.as_of_date && (
            <span className="text-[10px] text-ink-500">
              Pipeline {brief.as_of_date}
              {brief.comparison_date && ` vs ${brief.comparison_date}`}
              {brief.source_dates?.funded_as_of
                && ` · funded actuals ${brief.source_dates.funded_as_of}`}
            </span>
          )}
          {!loading && identity && (
            <button
              type="button"
              data-testid="weekly-brief-mark-read"
              aria-label="Mark this week's brief as read"
              title="Mark as read — reappears when a new upload has new insights"
              onClick={() => {
                writeDismissed(portfolioId, identity);
                setDismissedId(identity);
              }}
              className="inline-flex shrink-0 items-center gap-1 rounded-md border border-[var(--color-line)] px-1.5 py-0.5 text-[10px] text-ink-400 hover:border-mint-400/40 hover:text-mint-300"
            >
              <Check size={11} /> Read
            </button>
          )}
        </div>
      </header>

      {loading && (
        <p className="text-[11px] text-ink-500" data-testid="weekly-brief-loading">
          Preparing this week's brief…
        </p>
      )}

      {!loading && brief?.status === "partial" && (
        <p className="mb-2 rounded-md bg-[#e0a458]/10 px-2 py-1 text-[10px] text-[#e0a458]"
          data-testid="weekly-brief-partial">
          {brief.reason ?? "Some insights could not be produced this week."}
        </p>
      )}

      {!loading && insights.length === 0 && (
        <p className="text-[11px] text-ink-500" data-testid="weekly-brief-empty">
          No material changes this week.
        </p>
      )}

      {!loading && insights.length > 0 && (
        <ul className="space-y-2" data-testid="weekly-brief-list">
          {insights.map((i) => (
            <InsightCard key={i.insight_id} insight={i} onNavigate={onNavigate} />
          ))}
        </ul>
      )}

      {!loading && brief && <Omissions brief={brief} />}
    </section>
  );
}
