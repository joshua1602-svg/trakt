"use client";

import { useState } from "react";

import { CopilotDemo } from "@/components/demo/CopilotDemo";
import { Badge, buttonStyles } from "@/components/ui";
import type { DemoMetaResponse } from "@/types/demo";

/**
 * Demo 1 — the portfolio query demo, user-started.
 *
 * A static poster of the demo surface with a play affordance; nothing runs
 * until the visitor presses start. Starting mounts the live interactive
 * demo and opens with one scripted question — "Show the funded balance by
 * book." — so the visitor sees the query → governed answer flow from frame
 * zero, then can ask their own questions. Restart is the demo's own Reset.
 */

const OPENING_QUESTION_ID = "balance_by_book";

export function QueryDemo({ meta }: { meta: DemoMetaResponse }) {
  const [started, setStarted] = useState(false);

  const opening =
    meta.suggestedQuestions.find((q) => q.id === OPENING_QUESTION_ID) ??
    meta.suggestedQuestions[0];

  if (started) {
    return <CopilotDemo meta={meta} initialQuestion={opening} />;
  }

  return (
    <figure className="m-0">
      {/* Poster: a non-interactive replica of the demo surface, carrying the
          real synthetic scope so the frame is honest before it is live. Same
          5:4 frame as the controls demo — at its natural height the faded
          replica read as a broken render rather than a still. */}
      <div className="relative aspect-[5/4] overflow-hidden rounded-2xl border border-line bg-navy-900/70">
        {/* A faithful still of the demo's pre-question state — header, ask
            row, every suggestion, the refusal heading — so the frame fills
            and reads as the product rather than a fragment. */}
        <div
          aria-hidden="true"
          className="pointer-events-none flex h-full select-none flex-col gap-4 p-4 opacity-70 sm:p-5"
        >
          <div className="flex flex-wrap items-start justify-between gap-3 rounded-xl border border-line bg-navy-850/80 px-4 py-3">
            <div>
              <p className="text-[15px] font-semibold text-ink-100">{meta.scope.client}</p>
              <p className="mt-1 text-[12px] text-ink-300">
                {meta.scope.portfolioName} · {meta.scope.loanCount} exposures ·{" "}
                {meta.scope.totalBalanceDisplay} as at {meta.scope.asOfDisplay}
              </p>
            </div>
            <Badge tone="synthetic">Synthetic data</Badge>
          </div>

          <div className="flex gap-2">
            <div className="flex-1 rounded-xl border border-line bg-navy-950/60 px-4 py-3 text-sm text-ink-500">
              Ask about balance, concentration, LTV, reporting…
            </div>
            <div className="rounded-lg bg-peri-400/60 px-5 py-3 text-sm font-semibold text-navy-950">
              Ask Trakt
            </div>
          </div>

          <div className="flex-1">
            <p className="mb-2 text-[11px] font-semibold uppercase tracking-wider text-ink-400">
              Try one of these
            </p>
            <div className="flex flex-wrap gap-2">
              {meta.suggestedQuestions.map((question) => (
                <span
                  key={question.id}
                  className="rounded-full border border-line bg-navy-850 px-3.5 py-1.5 text-xs text-ink-200"
                >
                  {question.label}
                </span>
              ))}
              {meta.reportActions.map((action) => (
                <span
                  key={action.id}
                  className="rounded-full border border-peri-500/50 bg-peri-400/10 px-3.5 py-1.5 text-xs font-medium text-peri-200"
                >
                  {action.label}
                </span>
              ))}
            </div>
          </div>

          <div className="rounded-xl border border-mint-400/25 bg-navy-950/40 p-4">
            <p className="text-lg font-semibold tracking-tight text-mint-400">
              Trakt declines what it cannot derive.
            </p>
            <div className="mt-3 flex flex-wrap gap-2">
              {meta.exampleUnsupported.map((example) => (
                <span
                  key={example.id}
                  className="rounded-full border border-line bg-navy-850 px-3.5 py-1.5 text-xs text-ink-200"
                >
                  {example.label}
                </span>
              ))}
            </div>
          </div>
        </div>

        {/* A real scrim, not a tint: the button has to stay legible over
            whatever sits beneath it at any width. */}
        <div className="absolute inset-0 flex items-center justify-center bg-navy-950/80 backdrop-blur-[2px]">
          {/* demo_open fires from the demo itself when the scripted opening
              question runs, so starting is counted exactly once. */}
          <button type="button" onClick={() => setStarted(true)} className={buttonStyles.primary}>
            <svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
              <path d="M7 4l14 8-14 8z" />
            </svg>{" "}
            Watch query demo
          </button>
        </div>
      </div>

      {/* Below the frame, never over it — it used to cross the input row and
          the Ask Trakt button at phone width. */}
      <figcaption className="mt-3 text-[11px] leading-relaxed text-ink-500">
        Interactive · opens with “{opening?.label ?? "a portfolio question"}”
      </figcaption>
    </figure>
  );
}
