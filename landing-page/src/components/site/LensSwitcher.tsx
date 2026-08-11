"use client";

import { useState } from "react";

import { cx } from "@/components/ui";
import type { DemoScopeInfo } from "@/types/demo";

/**
 * The operating-model lens switcher: three lenses over the same underlying
 * book rows, recomputing the total on select. No dataset changes hands — the
 * rows are the demo pack's existing figures, excluded rows dim rather than
 * disappear, and the total is summed from the very rows on screen. That is
 * the section's claim ("individually reportable and consolidated, without
 * separate datasets to reconcile") demonstrated instead of asserted.
 */

const LENSES = [
  { id: "origination", label: "Origination", bookIds: ["alp_origination"] },
  { id: "acquired", label: "Acquired", bookIds: ["alp_acquired"] },
  {
    id: "sponsor",
    label: "Sponsor total",
    bookIds: ["alp_origination", "alp_acquired", "spv1_sponsored"],
  },
] as const;

type LensId = (typeof LENSES)[number]["id"];

/** Parse a display figure like "£15,432,544" back to its integer pounds. */
const parseBalance = (display: string): number =>
  Number(display.replace(/[^0-9]/g, "")) || 0;

const formatBalance = (value: number): string => `£${value.toLocaleString("en-GB")}`;

export function LensSwitcher({ scope }: { scope: DemoScopeInfo }) {
  const [lensId, setLensId] = useState<LensId>("sponsor");
  const lens = LENSES.find((l) => l.id === lensId) ?? LENSES[2];

  const included = new Set<string>(lens.bookIds);
  const total = scope.books
    .filter((book) => included.has(book.id))
    .reduce((sum, book) => sum + parseBalance(book.balanceDisplay), 0);

  return (
    <div className="rounded-2xl border border-line bg-navy-900/80 p-4 shadow-[0_24px_60px_-30px_rgba(0,0,0,0.9)] sm:p-5">
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-line pb-3">
        <p className="text-xs font-medium text-ink-300">Portfolio scope</p>
        <div role="group" aria-label="Portfolio lens" className="flex gap-1">
          {LENSES.map((option) => (
            <button
              key={option.id}
              type="button"
              aria-pressed={option.id === lensId}
              onClick={() => setLensId(option.id)}
              className={cx(
                "rounded-full border px-2.5 py-1 text-[11px] font-medium transition-colors",
                option.id === lensId
                  ? "border-peri-500/60 bg-navy-850 text-ink-100"
                  : "border-line bg-navy-900/60 text-ink-400 hover:border-peri-500/40 hover:text-ink-200",
              )}
            >
              {option.label}
            </button>
          ))}
        </div>
      </div>

      {/* The same rows under every lens: excluded books dim, they never
          disappear — one dataset, different scopes. */}
      <ul className="space-y-1.5 pt-4">
        {scope.books.map((book) => {
          const active = included.has(book.id);
          return (
            <li
              key={book.id}
              className={cx(
                "flex items-baseline justify-between gap-3 rounded-lg border px-3 py-2 transition-opacity",
                active
                  ? "border-line-soft bg-navy-900/60 opacity-100"
                  : "border-line-soft bg-navy-900/40 opacity-40",
              )}
            >
              <p className="min-w-0 text-[12px] text-ink-300">
                {book.label}
                {book.balanceSheetStatus === "sold" ? (
                  <span className="ml-1.5 text-[10px] text-amber-400">sold</span>
                ) : null}
              </p>
              <p className="shrink-0 text-[13px] font-semibold tabular-nums text-ink-200">
                {book.balanceDisplay}
              </p>
            </li>
          );
        })}
        <li className="flex items-baseline justify-between gap-3 rounded-lg border border-peri-500/60 bg-navy-850 px-3 py-2">
          <p className="text-[12px] font-semibold text-ink-100">{lens.label}</p>
          <p
            aria-live="polite"
            className="shrink-0 text-[13px] font-semibold tabular-nums text-ink-100"
          >
            {formatBalance(total)}
          </p>
        </li>
      </ul>
    </div>
  );
}
