"use client";

import { useCallback, useEffect, useId, useRef, useState } from "react";

import { ArtifactList } from "@/components/demo/Artifacts";
import { ReportPreview } from "@/components/demo/ReportPreview";
import { Badge, Card, buttonStyles, cx } from "@/components/ui";
import { track } from "@/lib/analytics";
import type {
  DemoAnswerResponse,
  DemoMetaResponse,
  DemoReportResponse,
} from "@/types/demo";

/**
 * The interactive synthetic demonstration.
 *
 * Mirrors the Trakt Copilot action layer — ask a governed portfolio question,
 * get an evidence-backed answer, request a report action — rather than the
 * wider MI Agent workspace. Every answer comes from `/api/demo/*`; this
 * component performs no portfolio calculation and holds no portfolio data
 * beyond what the current answer returned.
 */

type Turn =
  | { kind: "question"; id: string; text: string }
  | { kind: "answer"; id: string; payload: DemoAnswerResponse }
  | { kind: "report"; id: string; payload: DemoReportResponse }
  | { kind: "pending"; id: string };

let turnCounter = 0;
const nextTurnId = () => `turn-${(turnCounter += 1)}`;

async function postJson<T>(url: string, body: unknown): Promise<T> {
  const response = await fetch(url, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = (await response.json()) as T & { error?: string };
  if (!response.ok && data.error) throw new Error(data.error);
  if (!response.ok) throw new Error("Something went wrong. Please try again.");
  return data;
}

export function CopilotDemo({ meta }: { meta: DemoMetaResponse }) {
  const [turns, setTurns] = useState<Turn[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [remaining, setRemaining] = useState(meta.limits.questionsPerSession);

  const inputId = useId();
  const logRef = useRef<HTMLDivElement>(null);
  const started = turns.length > 0;

  // Keep the newest turn in view without yanking the whole page around.
  useEffect(() => {
    if (!started) return;
    const node = logRef.current?.lastElementChild;
    node?.scrollIntoView({ block: "nearest", behavior: "smooth" });
  }, [turns, started]);

  const ask = useCallback(
    async (payload: { question?: string; questionId?: string }, display: string) => {
      setBusy(true);
      setError(null);
      const pendingId = nextTurnId();
      setTurns((current) => [
        ...current,
        { kind: "question", id: nextTurnId(), text: display },
        { kind: "pending", id: pendingId },
      ]);

      try {
        const answer = await postJson<DemoAnswerResponse>("/api/demo/query", payload);
        setRemaining(answer.usage.questionsRemaining);
        setTurns((current) => [
          ...current.filter((t) => t.id !== pendingId),
          { kind: "answer", id: nextTurnId(), payload: answer },
        ]);

        if (answer.status === "answered") {
          track("answer_supported", { intentId: answer.intentId ?? undefined });
        } else if (answer.status === "limit_reached") {
          track("session_limit_reached", { source: "question" });
        } else {
          track("answer_unsupported", { intentId: answer.intentId ?? "unmatched" });
        }
      } catch (caught) {
        setTurns((current) => current.filter((t) => t.id !== pendingId));
        setError(caught instanceof Error ? caught.message : "Something went wrong.");
      } finally {
        setBusy(false);
      }
    },
    [],
  );

  const requestReport = useCallback(async (reportId: string, label: string) => {
    setBusy(true);
    setError(null);
    track("report_preview_request", { reportId });
    const pendingId = nextTurnId();
    setTurns((current) => [
      ...current,
      { kind: "question", id: nextTurnId(), text: label },
      { kind: "pending", id: pendingId },
    ]);

    try {
      const report = await postJson<DemoReportResponse>("/api/demo/report", { reportId });
      setTurns((current) => [
        ...current.filter((t) => t.id !== pendingId),
        { kind: "report", id: nextTurnId(), payload: report },
      ]);
      if (report.status === "limit_reached") {
        track("session_limit_reached", { source: "report" });
      }
    } catch (caught) {
      setTurns((current) => current.filter((t) => t.id !== pendingId));
      setError(caught instanceof Error ? caught.message : "Something went wrong.");
    } finally {
      setBusy(false);
    }
  }, []);

  const onSuggested = (id: string, label: string) => {
    track("suggested_question_click", { intentId: id });
    void ask({ questionId: id }, label);
  };

  const onSubmit = (event: React.FormEvent) => {
    event.preventDefault();
    const question = input.trim();
    if (!question || busy) return;
    track("typed_question_submit");
    setInput("");
    void ask({ question }, question);
  };

  const reset = () => {
    track("demo_reset");
    setTurns([]);
    setInput("");
    setError(null);
  };

  const outOfQuestions = remaining <= 0;

  return (
    <Card className="overflow-hidden p-0">
      {/* Scope header — always visible, so no answer is ever seen without it. */}
      <div className="flex flex-wrap items-start justify-between gap-3 border-b border-line bg-navy-850/60 px-5 py-4">
        <div>
          <p className="text-sm font-semibold text-ink-100">{meta.scope.client}</p>
          <p className="mt-0.5 text-xs text-ink-400">
            {meta.scope.portfolioName} · {meta.scope.assetClass} ·{" "}
            {meta.scope.loanCount.toLocaleString("en-GB")} exposures ·{" "}
            {meta.scope.totalBalanceDisplay} as at {meta.scope.asOfDisplay}
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Badge tone="synthetic">Synthetic data</Badge>
          {started ? (
            <button type="button" onClick={reset} className={cx(buttonStyles.ghost, "px-3 py-1.5 text-xs")}>
              Reset
            </button>
          ) : null}
        </div>
      </div>

      <div className="px-5 py-5 sm:px-6">
        {!started ? (
          <p className="mb-4 max-w-2xl text-sm leading-relaxed text-ink-300">
            {meta.scope.clientDescription} Ask a question below, or choose one of the
            suggestions — every answer is computed by Trakt&apos;s deterministic engine
            from this portfolio&apos;s governed dataset.
          </p>
        ) : null}

        {started ? (
          <div ref={logRef} className="mb-5 space-y-4" aria-live="polite" aria-atomic="false">
            {turns.map((turn) => (
              <TurnView
                key={turn.id}
                turn={turn}
                onFollowUp={onSuggested}
                onReport={requestReport}
                reportIds={meta.reportActions.map((r) => r.id)}
              />
            ))}
          </div>
        ) : null}

        {error ? (
          <p role="alert" className="mb-4 rounded-lg border border-rose-400/35 bg-rose-400/10 px-3 py-2 text-sm text-rose-400">
            {error}
          </p>
        ) : null}

        <form onSubmit={onSubmit} className="flex flex-col gap-2 sm:flex-row">
          <label htmlFor={inputId} className="sr-only">
            Ask a question about the synthetic portfolio
          </label>
          <input
            id={inputId}
            type="text"
            value={input}
            onChange={(event) => setInput(event.target.value)}
            maxLength={meta.limits.maxQuestionLength}
            disabled={busy || outOfQuestions}
            placeholder={
              outOfQuestions
                ? "Session question limit reached"
                : "Ask about balance, concentration, LTV, reporting…"
            }
            className="min-w-0 flex-1 rounded-lg border border-line bg-navy-950 px-4 py-3 text-sm text-ink-100 placeholder:text-ink-500 disabled:opacity-55"
          />
          <button
            type="submit"
            disabled={busy || outOfQuestions || input.trim().length === 0}
            className={buttonStyles.primary}
          >
            {busy ? "Working…" : "Ask Trakt"}
          </button>
        </form>

        <p className="mt-2 text-[11px] text-ink-500">
          {remaining} of {meta.limits.questionsPerSession} questions remaining in this
          demonstration session.
        </p>

        <div className="mt-5">
          <p className="mb-2 text-[11px] font-semibold uppercase tracking-wider text-ink-400">
            Suggested questions
          </p>
          <ul className="flex flex-wrap gap-2">
            {meta.suggestedQuestions.map((question) => (
              <li key={question.id}>
                <button
                  type="button"
                  disabled={busy || outOfQuestions}
                  onClick={() => onSuggested(question.id, question.label)}
                  className="rounded-full border border-line bg-navy-850 px-3.5 py-1.5 text-xs text-ink-200 transition-colors hover:border-peri-500 hover:text-ink-100 disabled:opacity-50"
                >
                  {question.label}
                </button>
              </li>
            ))}
          </ul>

          <p className="mb-2 mt-4 text-[11px] font-semibold uppercase tracking-wider text-ink-400">
            Report actions
          </p>
          <ul className="flex flex-wrap gap-2">
            {meta.reportActions.map((action) => (
              <li key={action.id}>
                <button
                  type="button"
                  disabled={busy}
                  onClick={() => requestReport(action.id, action.label)}
                  className="rounded-full border border-peri-500/50 bg-peri-400/10 px-3.5 py-1.5 text-xs font-medium text-peri-200 transition-colors hover:bg-peri-400/20 disabled:opacity-50"
                >
                  {action.label}
                </button>
              </li>
            ))}
          </ul>

          {meta.exampleUnsupported.length > 0 ? (
            <p className="mt-4 text-[11px] leading-relaxed text-ink-500">
              Trakt declines what it cannot derive. Try{" "}
              {meta.exampleUnsupported.map((example, index) => (
                <span key={example.id}>
                  {index > 0 ? " or " : ""}
                  <button
                    type="button"
                    disabled={busy || outOfQuestions}
                    onClick={() => onSuggested(example.id, example.label)}
                    className="underline decoration-dotted underline-offset-2 hover:text-ink-300 disabled:opacity-50"
                  >
                    “{example.label}”
                  </button>
                </span>
              ))}{" "}
              to see how.
            </p>
          ) : null}
        </div>
      </div>
    </Card>
  );
}

/* -------------------------------------------------------------------------- */

function TurnView({
  turn,
  onFollowUp,
  onReport,
  reportIds,
}: {
  turn: Turn;
  onFollowUp: (id: string, label: string) => void;
  onReport: (id: string, label: string) => void;
  reportIds: string[];
}) {
  if (turn.kind === "question") {
    return (
      <div className="flex justify-end">
        <p className="max-w-[85%] rounded-xl rounded-br-sm bg-navy-700 px-4 py-2.5 text-sm text-ink-100">
          {turn.text}
        </p>
      </div>
    );
  }

  if (turn.kind === "pending") {
    return (
      <div className="flex items-center gap-1.5 px-1 text-ink-400" role="status">
        <span className="sr-only">Working on your question</span>
        <span className="thinking-dot h-1.5 w-1.5 rounded-full bg-peri-400" />
        <span className="thinking-dot h-1.5 w-1.5 rounded-full bg-peri-400" />
        <span className="thinking-dot h-1.5 w-1.5 rounded-full bg-peri-400" />
      </div>
    );
  }

  if (turn.kind === "report") {
    return <ReportPreview payload={turn.payload} onFollowUp={onFollowUp} onReport={onReport} reportIds={reportIds} />;
  }

  return <AnswerView payload={turn.payload} onFollowUp={onFollowUp} onReport={onReport} reportIds={reportIds} />;
}

function AnswerView({
  payload,
  onFollowUp,
  onReport,
  reportIds,
}: {
  payload: DemoAnswerResponse;
  onFollowUp: (id: string, label: string) => void;
  onReport: (id: string, label: string) => void;
  reportIds: string[];
}) {
  const unsupported = payload.status === "unsupported";
  const limited = payload.status === "limit_reached";

  return (
    <article
      className={cx(
        "animate-rise rounded-xl border p-4 sm:p-5",
        unsupported
          ? "border-amber-400/30 bg-amber-400/[0.04]"
          : limited
            ? "border-peri-500/40 bg-peri-400/[0.06]"
            : "border-line bg-navy-950/60",
      )}
    >
      {unsupported ? (
        <p className="mb-2 text-[11px] font-semibold uppercase tracking-wider text-amber-400">
          Not supported in this demonstration
        </p>
      ) : null}

      <p className="text-sm leading-relaxed text-ink-100">{payload.answer}</p>

      {payload.productionNote ? (
        <p className="mt-3 border-l-2 border-line pl-3 text-xs leading-relaxed text-ink-400">
          {payload.productionNote}
        </p>
      ) : null}

      {payload.artifacts?.length ? <ArtifactList artifacts={payload.artifacts} /> : null}

      {payload.status !== "limit_reached" ? (
        <footer className="mt-4 border-t border-line-soft pt-3">
          <p className="flex flex-wrap gap-x-3 gap-y-1 text-[11px] text-ink-500">
            {payload.interpreted ? <span>{payload.interpreted}</span> : null}
            <span>As at {payload.asOfDisplay}</span>
            <span>{payload.portfolioScope}</span>
            <span className="text-amber-400">Synthetic portfolio</span>
          </p>

          {payload.followUps?.length ? (
            <ul className="mt-3 flex flex-wrap gap-2">
              {payload.followUps.map((followUp) => (
                <li key={followUp.id}>
                  <button
                    type="button"
                    onClick={() =>
                      reportIds.includes(followUp.id)
                        ? onReport(followUp.id, followUp.label)
                        : onFollowUp(followUp.id, followUp.label)
                    }
                    className="rounded-full border border-line px-3 py-1 text-xs text-ink-300 transition-colors hover:border-peri-500 hover:text-ink-100"
                  >
                    {followUp.label}
                  </button>
                </li>
              ))}
            </ul>
          ) : null}
        </footer>
      ) : (
        <a href="#book-a-demo" className={cx(buttonStyles.primary, "mt-4")}>
          Book a portfolio walkthrough
        </a>
      )}
    </article>
  );
}
