"use client";

import { buttonStyles, cx } from "@/components/ui";
import { track } from "@/lib/analytics";
import type { DemoScopeInfo } from "@/types/demo";

/**
 * Hero, with an interface preview built from the same synthetic portfolio the
 * live demo answers from — not stock photography, and not an invented screen.
 */
export function Hero({ scope }: { scope: DemoScopeInfo }) {
  return (
    <div className="grid items-center gap-10 lg:grid-cols-[1.05fr_1fr] lg:gap-14">
      <div>
        <p className="mb-4 text-xs font-semibold uppercase tracking-[0.18em] text-peri-400">
          Trakt for specialist lending
        </p>
        <h1 className="text-balance text-4xl font-semibold leading-[1.08] tracking-tight text-ink-100 sm:text-5xl">
          Portfolio intelligence. Wherever you work.
        </h1>
        <p className="mt-5 max-w-xl text-lg leading-relaxed text-ink-200">
          Trakt turns fragmented portfolio data into trusted answers, automated
          reporting and governed workflows.
        </p>
        <p className="mt-3 max-w-xl text-[15px] leading-relaxed text-ink-400">
          Use Trakt through Microsoft 365 Copilot, Teams, the Trakt workspace or
          automated reporting processes.
        </p>

        <div className="mt-8 flex flex-col gap-3 sm:flex-row">
          <a
            href="#live-demo"
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

        <p className="mt-6 text-xs text-ink-500">
          The public demonstration uses a wholly synthetic portfolio. No client or
          consumer information is displayed, and no upload is accepted.
        </p>
      </div>

      <InterfacePreview scope={scope} />
    </div>
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
      <div className="flex items-center justify-between border-b border-line pb-3">
        <p className="text-xs font-medium text-ink-300">Microsoft 365 Copilot</p>
        <p className="text-[10px] uppercase tracking-wider text-amber-400">Synthetic</p>
      </div>

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

          <p className="mt-3 border-t border-line-soft pt-2 text-[10px] text-ink-500">
            Deterministic · As at {scope.asOfDisplay} · {scope.client}
          </p>
        </div>
      </div>

      <p className={cx("mt-4 text-center text-[10px] text-ink-500")}>
        The same governed answer reaches Teams, the Trakt workspace and scheduled
        reports.
      </p>
    </div>
  );
}
