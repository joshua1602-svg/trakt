/**
 * Phase 2A — a ranked list of contributors to a CHANGE.
 *
 * These are deltas, not current balances, so the heading has to say so: the
 * top entry is whoever moved the number most, in either direction, and a
 * negative entry is normal. Presenting them without that framing would invite
 * exactly the misreading the brief warns about.
 */

import type { MovementContributor } from "@/domain";
import { cn, formatGBP } from "@/lib/utils";

export function ContributorList({
  title, rows, emptyLabel = "No contributors for this week.", max = 3,
}: {
  title: string;
  rows: MovementContributor[] | undefined;
  emptyLabel?: string;
  max?: number;
}) {
  const items = (rows ?? []).slice(0, max);
  return (
    <div className="mt-2">
      <div className="text-[10px] uppercase tracking-wide text-ink-500">{title}</div>
      {items.length === 0 ? (
        <div className="mt-1 text-[11px] text-ink-500">{emptyLabel}</div>
      ) : (
        <ul className="mt-1 space-y-0.5">
          {items.map((c) => (
            <li key={c.name} className="flex items-baseline justify-between gap-3">
              {/* Long broker names are common; truncate rather than reflow the
                  tooltip, and keep the full value in the title attribute. */}
              <span className="min-w-0 flex-1 truncate text-[11px] text-ink-300"
                title={c.name}>
                {c.name}
              </span>
              <span className={cn("shrink-0 text-[11px] tabular-nums",
                c.amount < 0 ? "text-[#eb6f6f]" : "text-[#5ec6b8]")}>
                {c.amount >= 0 ? "+" : "−"}{formatGBP(Math.abs(c.amount), { compact: true })}
              </span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
