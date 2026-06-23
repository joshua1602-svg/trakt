import { useState } from "react";
import { ChevronDown, ShieldAlert } from "lucide-react";
import type { WatchlistItem } from "@/domain";
import { Badge } from "@/components/ui";
import { severityTone } from "@/components/pipeline/bits";
import { cn } from "@/lib/utils";

const SEVERITY_RANK: Record<WatchlistItem["severity"], number> = { blocker: 0, warning: 1, info: 2 };

/**
 * Deterministic pipeline early-warning / watchlist tiles. Concise business-facing
 * titles are always shown; the underlying technical detail is expandable. All
 * items are backend-derived (forecast_bridge.build_pipeline_watchlist).
 */
export function PipelineWatchlist({ items }: { items: WatchlistItem[] }) {
  const [expanded, setExpanded] = useState<Record<number, boolean>>({});

  if (!items || items.length === 0) {
    return (
      <section className="rounded-xl border border-[var(--color-line)] bg-navy-900/40 p-5">
        <Header count={0} />
        <p className="mt-2 text-[12px] text-ink-400">No early warnings for this run.</p>
      </section>
    );
  }

  const sorted = [...items].sort((a, b) => SEVERITY_RANK[a.severity] - SEVERITY_RANK[b.severity]);

  return (
    <section className="rounded-xl border border-[var(--color-line)] bg-navy-900/40 p-5">
      <Header count={items.length} />
      <ul className="mt-3 space-y-2">
        {sorted.map((item, i) => {
          const open = !!expanded[i];
          return (
            <li
              key={`${item.category}-${i}`}
              className="rounded-lg border border-[var(--color-line-soft)] bg-navy-850/60 px-3.5 py-2.5"
            >
              <div className="flex items-start justify-between gap-3">
                <div className="flex items-start gap-2">
                  <Badge tone={severityTone(item.severity)}>{item.severity}</Badge>
                  <span className="text-[12px] font-medium text-ink-200">{item.title}</span>
                </div>
                {item.detail && (
                  <button
                    type="button"
                    aria-label="Toggle technical detail"
                    onClick={() => setExpanded((e) => ({ ...e, [i]: !e[i] }))}
                    className="text-ink-500 hover:text-ink-300"
                  >
                    <ChevronDown size={14} className={cn("transition-transform", !open && "-rotate-90")} />
                  </button>
                )}
              </div>
              {open && item.detail && (
                <p className="mt-1.5 border-t border-[var(--color-line-soft)] pt-1.5 text-[11px] text-ink-400">
                  {item.detail}
                </p>
              )}
            </li>
          );
        })}
      </ul>
    </section>
  );
}

function Header({ count }: { count: number }) {
  return (
    <div className="flex items-center justify-between gap-2">
      <div className="flex items-center gap-2.5">
        <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-navy-700/70 text-amber-400">
          <ShieldAlert size={17} />
        </div>
        <h2 className="text-sm font-semibold text-ink-100">Pipeline Watchlist</h2>
      </div>
      {count > 0 && <Badge tone="amber">{count}</Badge>}
    </div>
  );
}
