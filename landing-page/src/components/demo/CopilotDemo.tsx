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

/** Questions remaining at which the session counter becomes visible. */
const COUNTER_VISIBLE_FROM = 3;

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

export function CopilotDemo({
  meta,
  initialQuestion,
}: {
  meta: DemoMetaResponse;
  /**
   * Optional scripted opening: asked once when the component mounts, so the
   * query-demo section starts from the user's question rather than an empty
   * surface. Reset does not replay it.
   */
  initialQuestion?: { id: string; label: string };
}) {
  const [turns, setTurns] = useState<Turn[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [remaining, setRemaining] = useState(meta.limits.questionsPerSession);
  /** Announced to screen readers on every state change; visually hidden. */
  const [announcement, setAnnouncement] = useState("");

  const inputId = useId();
  const logRef = useRef<HTMLDivElement>(null);
  const openedRef = useRef(false);
  const started = turns.length > 0;

  /** Fires once per visit, the first time the visitor engages with the demo. */
  const markOpened = useCallback(() => {
    if (openedRef.current) return;
    openedRef.current = true;
    track("demo_open");
  }, []);

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
      setAnnouncement("Working on your question.");
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
          setAnnouncement("Answer ready.");
          track("demo_answer_returned", { intentId: answer.intentId ?? undefined });
        } else if (answer.status === "limit_reached") {
          setAnnouncement("Session question limit reached.");
        } else {
          setAnnouncement("Trakt declined that question. An explanation follows.");
          track("demo_refusal_returned", { intentId: answer.intentId ?? "unmatched" });
        }
      } catch (caught) {
        setTurns((current) => current.filter((t) => t.id !== pendingId));
        const message = caught instanceof Error ? caught.message : "Something went wrong.";
        setError(message);
        setAnnouncement(message);
      } finally {
        setBusy(false);
      }
    },
    [],
  );

  /**
   * The scripted opening. Deferred a tick so the effect body sets no state
   * synchronously; the ref keeps it one-shot, so Reset never replays it and
   * it consumes exactly one session question.
   */
  const initialAskedRef = useRef(false);
  useEffect(() => {
    if (!initialQuestion || initialAskedRef.current) return;
    const id = window.setTimeout(() => {
      if (initialAskedRef.current) return;
      initialAskedRef.current = true;
      markOpened();
      void ask({ questionId: initialQuestion.id }, initialQuestion.label);
    }, 0);
    return () => window.clearTimeout(id);
  }, [initialQuestion, ask, markOpened]);

  const requestReport = useCallback(async (reportId: string, label: string) => {
    setBusy(true);
    setError(null);
    setAnnouncement("Preparing the report preview.");
    markOpened();
    track("report_preview_opened", { reportId });
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
      setAnnouncement(
        report.status === "limit_reached"
          ? "Session report limit reached."
          : "Report preview ready.",
      );
    } catch (caught) {
      setTurns((current) => current.filter((t) => t.id !== pendingId));
      const message = caught instanceof Error ? caught.message : "Something went wrong.";
      setError(message);
      setAnnouncement(message);
    } finally {
      setBusy(false);
    }
  }, [markOpened]);

  const onSuggested = (id: string, label: string) => {
    markOpened();
    track("suggested_question_click", { intentId: id });
    void ask({ questionId: id }, label);
  };

  const onSubmit = (event: React.FormEvent) => {
    event.preventDefault();
    const question = input.trim();
    if (!question || busy) return;
    markOpened();
    track("typed_question_submit");
    setInput("");
    void ask({ question }, question);
  };

  const reset = () => {
    setTurns([]);
    setInput("");
    setError(null);
    setAnnouncement("Conversation reset.");
  };

  const outOfQuestions = remaining <= 0;

  return (
    <Card className="overflow-hidden p-0">
      {/* Scope header — always visible, so no answer is ever seen without it. */}
      <div className="flex flex-wrap items-start justify-between gap-3 border-b border-line bg-navy-850/60 px-5 py-4">
        <div>
          <p className="text-body font-semibold text-ink-100">{meta.scope.client}</p>
          <p className="mt-0.5 text-small text-ink-400">
            {meta.scope.portfolioName} · {meta.scope.assetClass} ·{" "}
            {meta.scope.loanCount.toLocaleString("en-GB")} exposures ·{" "}
            {meta.scope.totalBalanceDisplay} as at {meta.scope.asOfDisplay}
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Badge tone="synthetic">Synthetic data</Badge>
          {started ? (
            <button type="button" onClick={reset} className={cx(buttonStyles.ghost, "px-3 py-1.5 text-small")}>
              Reset
            </button>
          ) : null}
        </div>
      </div>

      <div className="px-5 py-5 sm:px-6">
        {/* Status announcements. Short and polite, so a screen reader is told
            what happened without having the whole answer read at it. */}
        <p role="status" aria-live="polite" className="sr-only">
          {announcement}
        </p>

        {!started && meta.scope.clientDescription ? (
          <p className="mb-4 max-w-2xl text-body leading-relaxed text-ink-300">
            {meta.scope.clientDescription}
          </p>
        ) : null}

        {started ? (
          <div ref={logRef} className="mb-5 space-y-4" role="log">
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
          <p role="alert" className="mb-4 rounded-lg border border-rose-400/35 bg-rose-400/10 px-3 py-2 text-body text-rose-400">
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
            className="min-w-0 flex-1 rounded-lg border border-line bg-navy-950 px-4 py-3 text-body text-ink-100 placeholder:text-ink-500 disabled:opacity-55"
          />
          <button
            type="submit"
            disabled={busy || outOfQuestions || input.trim().length === 0}
            className={buttonStyles.primary}
          >
            {busy ? "Working…" : "Ask Trakt"}
          </button>
        </form>

        {/* The session cap is real and server-enforced, but a counter running
            from the first question reads as metering. It stays silent until the
            visitor is nearly at the limit, which is the only point at which the
            number is of any use to them. */}
        {remaining <= COUNTER_VISIBLE_FROM ? (
          <p className="mt-2 text-small text-ink-500">
            {remaining} of {meta.limits.questionsPerSession} questions remaining in this
            demonstration session.
          </p>
        ) : null}

        <div className="mt-5">
          <p className="mb-2 text-small font-semibold uppercase tracking-wider text-ink-400">
            Try one of these
          </p>
          <ul className="flex flex-wrap gap-2">
            {meta.suggestedQuestions.map((question) => (
              <li key={question.id}>
                <button
                  type="button"
                  disabled={busy || outOfQuestions}
                  onClick={() => onSuggested(question.id, question.label)}
                  className="rounded-full border border-line bg-navy-850 px-3.5 py-1.5 text-left text-small text-ink-200 transition-colors hover:border-peri-500 hover:text-ink-100 disabled:opacity-50"
                >
                  {question.label}
                </button>
              </li>
            ))}
            {meta.reportActions.map((action) => (
              <li key={action.id}>
                <button
                  type="button"
                  disabled={busy}
                  onClick={() => requestReport(action.id, action.label)}
                  className="rounded-full border border-peri-500/50 bg-peri-400/10 px-3.5 py-1.5 text-left text-small font-medium text-peri-200 transition-colors hover:bg-peri-400/20 disabled:opacity-50"
                >
                  {action.label}
                </button>
              </li>
            ))}
          </ul>

          {/* The refusal claim itself now has its own section on the page —
              inside this card it sat behind a play control, visible only as
              text in a still. The prompts remain here as suggestions, which
              is what they are once the demo is running. */}
          {meta.exampleUnsupported.length > 0 ? (
            <ul className="mt-4 flex flex-wrap gap-2">
              {meta.exampleUnsupported.map((example) => (
                <li key={example.id}>
                  <button
                    type="button"
                    disabled={busy || outOfQuestions}
                    onClick={() => onSuggested(example.id, example.label)}
                    className="rounded-full border border-line bg-navy-850 px-3.5 py-1.5 text-left text-small text-ink-200 transition-colors hover:border-peri-500 hover:text-ink-100 disabled:opacity-50"
                  >
                    {example.label}
                  </button>
                </li>
              ))}
            </ul>
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
        <p className="max-w-[85%] rounded-xl rounded-br-sm bg-navy-700 px-4 py-2.5 text-body text-ink-100">
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
        <p className="mb-2 text-small font-semibold uppercase tracking-wider text-amber-400">
          Not supported in this demonstration
        </p>
      ) : null}

      <p className="text-body leading-relaxed text-ink-100">{payload.answer}</p>

      {payload.productionNote ? (
        <p className="mt-3 border-l-2 border-line pl-3 text-small leading-relaxed text-ink-400">
          {payload.productionNote}
        </p>
      ) : null}

      {payload.artifacts?.length ? <ArtifactList artifacts={payload.artifacts} /> : null}

      {payload.status !== "limit_reached" ? (
        <footer className="mt-4 border-t border-line-soft pt-3">
          {/* Provenance stays lean: the as-at date and the synthetic pill live
              once, in the portfolio header above every answer. */}
          <p className="flex flex-wrap gap-x-3 gap-y-1 break-words text-small text-ink-500">
            {payload.interpreted ? <span>{payload.interpreted}</span> : null}
            <span>{payload.portfolioScope}</span>
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
                    className="rounded-full border border-line px-3 py-1 text-small text-ink-300 transition-colors hover:border-peri-500 hover:text-ink-100"
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
