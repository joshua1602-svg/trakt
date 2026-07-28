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
  "Deterministic engine — same question, same number, every channel",
  "Traceable lineage — every published figure ties back to source",
  "Client-isolated environments and controlled data handling",
] as const;

export function Hero({ scope }: { scope: DemoScopeInfo }) {
  return (
    <div className="grid items-center gap-10 lg:grid-cols-[1.05fr_1fr] lg:gap-14">
      <div>
        <p className="mb-4 text-xs font-semibold uppercase tracking-[0.18em] text-peri-400">
          Trakt for specialist lending
        </p>
        <h1 className="text-balance text-4xl font-semibold leading-[1.08] tracking-tight text-ink-100 sm:text-5xl">
          One governed portfolio dataset. Every book, every report.
        </h1>
        <p className="mt-5 max-w-xl text-lg leading-relaxed text-ink-200">
          Trakt normalises loan tapes, servicing extracts, valuations and funding data
          into one governed model that drives management, investor and regulatory
          reporting.
        </p>

        <p className="mt-5 max-w-xl text-[15px] font-medium leading-relaxed text-mint-400">
          Every figure is reconciled by construction rather than by comparison.
        </p>

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
            href="#example"
            onClick={() => track("hero_demo_click")}
            className={buttonStyles.primary}
          >
            Explore the live demo
          </a>
          <a
            href="#book-a-demo"
            onClick={() => track("book_demo_click", { source: "hero" })}
            className={buttonStyles.secondary}
          >
            Book a portfolio walkthrough
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
      className="rounded-2xl border border-line bg-navy-900/80 p-4 shadow-[0_24px_60px_-30px_rgba(0,0,0,0.9)] sm:p-5"
      aria-label="Preview of a governed Trakt answer"
      role="img"
    >
      <p className="border-b border-line pb-3 text-xs font-medium text-ink-300">
        Microsoft 365 Copilot
      </p>

      <div className="space-y-3 pt-4">
        <div className="flex justify-end">
          <p className="rounded-xl rounded-br-sm bg-navy-700 px-3.5 py-2 text-[13px] text-ink-100">
            What is the current funded portfolio balance?
          </p>
        </div>

        <div className="rounded-xl border border-line bg-navy-950/70 p-3.5">
          <p className="text-[13px] leading-relaxed text-ink-100">
            The funded book stands at {scope.totalBalanceDisplay} across{" "}
            {scope.loanCount} exposures as at {scope.asOfDisplay}.
          </p>

          <dl className="mt-3 grid grid-cols-2 gap-2">
            <div className="rounded-lg border border-line-soft bg-navy-900/60 px-3 py-2">
              <dt className="text-[10px] uppercase tracking-wider text-ink-500">
                Balance
              </dt>
              <dd className="mt-0.5 text-base font-semibold tabular-nums text-ink-100">
                {scope.totalBalanceDisplay}
              </dd>
            </div>
            <div className="rounded-lg border border-line-soft bg-navy-900/60 px-3 py-2">
              <dt className="text-[10px] uppercase tracking-wider text-ink-500">
                Exposures
              </dt>
              <dd className="mt-0.5 text-base font-semibold tabular-nums text-ink-100">
                {scope.loanCount}
              </dd>
            </div>
          </dl>

          <div className="mt-3 space-y-1.5" aria-hidden="true">
            {[
              { label: "East Midlands", width: "100%" },
              { label: "East of England", width: "91%" },
              { label: "Yorkshire & Humber", width: "90%" },
              { label: "London", width: "86%" },
            ].map((bar) => (
              <div key={bar.label} className="flex items-center gap-2">
                <span className="w-28 shrink-0 text-[10px] text-ink-400">{bar.label}</span>
                <span className="h-1.5 flex-1 overflow-hidden rounded-full bg-navy-800">
                  <span
                    className="block h-full rounded-full bg-peri-400"
                    style={{ width: bar.width }}
                  />
                </span>
              </div>
            ))}
          </div>

          {/* Provenance travels with the figures. "Deterministic" is a
              confirmatory signal, so it carries the accent; the synthetic label
              is a one-word provenance tag, not the page's disclaimer — that
              lives once, in the example section. */}
          <p className="mt-3 flex flex-wrap gap-x-2 border-t border-line-soft pt-2 text-[10px] text-ink-500">
            <span className="font-medium text-mint-400">Deterministic</span>
            <span>· As at {scope.asOfDisplay}</span>
            <span>· Synthetic portfolio</span>
          </p>
        </div>
      </div>
    </div>
  );
}
