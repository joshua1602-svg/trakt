"use client";

import { buttonStyles } from "@/components/ui";
import { track } from "@/lib/analytics";
import type { DemoScopeInfo } from "@/types/demo";

/**
 * Value proposition.
 *
 * Leads on the governed single-source claim rather than on channel breadth —
 * channel is a delivery choice and belongs in the delivery model section, where
 * it is harder for an incumbent to copy.
 *
 * The three proof points are chips, not cards: a sceptical reader scanning for
 * "can I trust the numbers" gets all three in one row without reading prose.
 * Green marks them, and marks the reconciliation claim, because in the product
 * green means proven — see the accent note in `app/globals.css`.
 */

const PROOF_POINTS = [
  "One governed portfolio model",
  "Deterministic, traceable calculations",
  "Reporting, controls and AI from the same data",
] as const;

export function Hero({ scope }: { scope: DemoScopeInfo }) {
  return (
    <div className="grid items-center gap-10 lg:grid-cols-12 lg:gap-x-8">
      <div className="lg:col-span-6">
        {/* The strapline, in the eyebrow's existing type treatment — measured
            at 555px against a 656px column at 1440 and 770px at 834, so it
            holds one line on both. At 390 the column is 350px and it must
            wrap; the second sentence is a block below `sm` so the break lands
            on the full stop by construction rather than by luck, and stays
            there if the type or the column ever changes. */}
        <p className="mb-4 text-xs font-semibold uppercase tracking-[0.18em] text-peri-400">
          <span>Agentic portfolio intelligence.</span>{" "}
          <span className="block sm:inline">Deterministic by design.</span>
        </p>
        <h1 className="text-balance text-4xl font-semibold leading-[1.08] tracking-tight text-ink-100 sm:text-5xl">
          One governed view of your lending portfolios.
        </h1>
        <p className="mt-5 max-w-[72ch] text-lg leading-relaxed text-ink-200">
          {/* "for specialist lenders" carries the audience naming the eyebrow
              used to do. Without it the page names no audience above the fold:
              the headline says "lending portfolios", not specialist lending,
              and metadata is not visible. */}
          Connect loan data, documents and funding requirements once. Trakt turns them
          into live portfolio monitoring, forecasting, controls and reporting for
          specialist lenders.
        </p>

        {/* The reconciliation line now leads the governance section — too
            abstract for a first screen, exactly right as the proof beneath
            the page's claims. */}

        {/* Chips, not cards: on a phone these stack as three tight lines, so the
            reader takes all three in without reading prose. Borders appear only
            once they sit in a row, where they separate the chips; unbordered on
            mobile they stay a list rather than becoming a green panel. */}
        <ul className="mt-6 flex flex-col gap-2 sm:flex-row sm:flex-wrap sm:gap-2.5">
          {PROOF_POINTS.map((point) => (
            <li
              key={point}
              className="flex items-start gap-2 text-[13px] leading-snug text-ink-200 sm:max-w-[15rem] sm:flex-1 sm:rounded-lg sm:border sm:border-mint-400/30 sm:px-3 sm:py-2"
            >
              <CheckMark />
              {point}
            </li>
          ))}
        </ul>

        <div className="mt-8 flex flex-col gap-3 sm:flex-row">
          <a
            href="#query-demo"
            onClick={() => track("hero_demo_click")}
            className={buttonStyles.primary}
          >
            Explore the live demo
          </a>
          {/* Word for word the nav button, and the same anchor. Three names for
              one form — "Book a demo", "Book a portfolio walkthrough", "Book a
              tailored demonstration" — made a visitor work out that they were
              the same destination. Two controls, one offer, one label; the
              closing CTA keeps its own wording because it sits above the form
              itself and competes with nothing. */}
          <a
            href="#book-a-demo"
            onClick={() => track("book_demo_click", { source: "hero" })}
            className={buttonStyles.secondary}
          >
            Demo on your portfolio
          </a>
        </div>
      </div>

      <InterfacePreview scope={scope} />
    </div>
  );
}

function CheckMark() {
  return (
    <svg
      width="14"
      height="14"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2.4"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      focusable="false"
      className="mt-0.5 shrink-0 text-mint-400"
    >
      <path d="M20 6L9 17l-5-5" />
    </svg>
  );
}

/**
 * A static recreation of the Trakt Copilot exchange, using this portfolio's real
 * figures. Static markup rather than an animation: it costs nothing to render,
 * cannot shift layout, and needs no motion exemption.
 */
function InterfacePreview({ scope }: { scope: DemoScopeInfo }) {
  return (
    <div
      className="rounded-2xl border border-line bg-navy-900/80 p-4 shadow-[0_24px_60px_-30px_rgba(0,0,0,0.9)] sm:p-5 lg:col-span-6"
      aria-label="Preview of a governed Trakt answer"
      role="img"
    >
      <p className="border-b border-line pb-3 text-xs font-medium text-ink-300">
        Microsoft 365 Copilot
      </p>

      <div className="space-y-3 pt-4">
        <div className="flex justify-end">
          <p className="rounded-xl rounded-br-sm bg-navy-700 px-3.5 py-2 text-[13px] text-ink-100">
            Show the funded balance by book.
          </p>
        </div>

        <div className="rounded-xl border border-line bg-navy-950/70 p-3.5">
          <p className="text-[13px] leading-relaxed text-ink-100">Three governed books.</p>

          {/* The differentiator, first thing a visitor sees. A flat total here
              would quietly contradict the multi-book claim above it — and the
              two totals are the point: one book is sold and derecognised, and
              the sponsor still reports on it. */}
          <dl className="mt-3 space-y-1.5">
            {scope.books.map((book) => (
              <div
                key={book.id}
                className="flex items-baseline justify-between gap-3 rounded-lg border border-line-soft bg-navy-900/60 px-3 py-1.5"
              >
                <dt className="min-w-0 text-[11px] text-ink-300">
                  {book.shortLabel}
                  {book.balanceSheetStatus === "sold" ? (
                    <span className="ml-1.5 text-[10px] text-amber-400">sold</span>
                  ) : null}
                </dt>
                <dd className="shrink-0 text-[13px] font-semibold tabular-nums text-ink-100">
                  {book.balanceDisplay}
                </dd>
              </div>
            ))}
            <div className="flex items-baseline justify-between gap-3 border-t border-line-soft px-3 pt-2">
              <dt className="text-[11px] text-ink-400">Platform total</dt>
              <dd className="text-[13px] font-semibold tabular-nums text-ink-200">
                {scope.platformBalanceDisplay}
              </dd>
            </div>
            <div className="flex items-baseline justify-between gap-3 px-3">
              <dt className="text-[11px] text-ink-300">Sponsor total</dt>
              <dd className="text-base font-semibold tabular-nums text-ink-100">
                {scope.totalBalanceDisplay}
              </dd>
            </div>
          </dl>

          {/* One confirmatory provenance signal; the as-at date and the
              synthetic disclosure live once, in the demo's portfolio header. */}
          <p className="mt-3 border-t border-line-soft pt-2 text-[10px] font-medium text-mint-400">
            Deterministic
          </p>
        </div>
      </div>
    </div>
  );
}
