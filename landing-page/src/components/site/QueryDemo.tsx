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
    <div className="relative">
      {/* Poster: a non-interactive replica of the demo surface, carrying the
          real synthetic scope so the frame is honest before it is live. */}
      <div
        aria-hidden="true"
        className="pointer-events-none select-none rounded-2xl border border-line bg-navy-900/70 p-4 opacity-70 sm:p-5"
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
        <div className="mt-4 flex gap-2">
          <div className="flex-1 rounded-xl border border-line bg-navy-950/60 px-4 py-3 text-sm text-ink-500">
            Ask about balance, concentration, LTV, reporting…
          </div>
          <div className="rounded-lg bg-peri-400/60 px-5 py-3 text-sm font-semibold text-navy-950">
            Ask Trakt
          </div>
        </div>
        <div className="mt-3 flex flex-wrap gap-2">
          {meta.suggestedQuestions.slice(0, 2).map((question) => (
            <span
              key={question.id}
              className="rounded-full border border-line bg-navy-850 px-3 py-1.5 text-[12px] text-ink-300"
            >
              {question.label}
            </span>
          ))}
        </div>
      </div>

      <div className="absolute inset-0 flex flex-col items-center justify-center gap-2 rounded-2xl bg-navy-950/45">
        {/* demo_open fires from the demo itself when the scripted opening
            question runs, so starting is counted exactly once. */}
        <button type="button" onClick={() => setStarted(true)} className={buttonStyles.primary}>
          <svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
            <path d="M7 4l14 8-14 8z" />
          </svg>{" "}
          Watch query demo
        </button>
        <span className="max-w-xs px-6 text-center text-[12px] font-medium leading-relaxed text-ink-300">
          Interactive · opens with “{opening?.label ?? "a portfolio question"}”
        </span>
      </div>
    </div>
  );
}
