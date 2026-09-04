import { useState } from "react";
import { AlertTriangle, ChevronDown, FileBarChart, RefreshCw, Sparkles, User } from "lucide-react";
import type { ChatMessage as ChatMessageType } from "@/domain";
import { ChatResult } from "@/components/ChatResult";
import { cn, formatTime } from "@/lib/utils";

/** Secondary provenance (assumptions) behind a one-line disclosure so a busy
 *  answer doesn't stack meta boxes; warnings + coverage stay always-visible. */
function AssumptionsDisclosure({ assumptions }: { assumptions: string[] }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="mt-2">
      <button
        type="button"
        onClick={() => setOpen((s) => !s)}
        aria-expanded={open}
        className="inline-flex items-center gap-1 text-[10px] font-semibold uppercase tracking-wider text-ink-500 hover:text-ink-300"
      >
        <ChevronDown size={12} className={cn("transition-transform", !open && "-rotate-90")} />
        Assumptions ({assumptions.length})
      </button>
      {open && (
        <div className="mt-1 rounded-lg border border-[var(--color-line-soft)] bg-navy-900/50 px-3 py-2">
          <ul className="list-disc space-y-0.5 pl-4 text-[11px] leading-relaxed text-ink-400">
            {assumptions.map((a, i) => (
              <li key={i}>{a}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

export function ChatMessage({
  message,
  onOpenArtifact,
  onRetry,
  onTogglePin,
  workspaceArtifactIds,
}: {
  message: ChatMessageType;
  onOpenArtifact?: (id: string) => void;
  onRetry?: () => void;
  onTogglePin?: (id: string) => void;
  /** Ids still present in the artifact workspace. When provided, links to
   *  artifacts that were since cleared render stale instead of no-opping. */
  workspaceArtifactIds?: Set<string>;
}) {
  const isUser = message.role === "user";
  const hasInlineResult = !isUser && !!message.artifacts && message.artifacts.length > 0;
  // Minimal client-facing provenance: only WHICH BOOK answered (funded /
  // pipeline / forecast), since that materially changes what a number means.
  // The full interpretation / parse mode stays backend-side (response metadata).
  const showDatasetBadge =
    !isUser && !message.pending && !message.error && !!message.datasetContext;

  return (
    <div className="animate-fade-in flex gap-2.5" data-role={message.role}>
      <div
        className={cn(
          "mt-0.5 flex h-7 w-7 shrink-0 items-center justify-center rounded-lg",
          isUser
            ? "bg-slate-600/40 text-slate-200"
            : message.error
              ? "bg-rose-400/15 text-rose-400"
              : "bg-gradient-to-br from-teal-500 to-emerald-600 text-white shadow-sm shadow-teal-900/40",
        )}
      >
        {isUser ? <User size={14} /> : message.error ? <AlertTriangle size={14} /> : <Sparkles size={14} />}
      </div>

      <div className="min-w-0 flex-1">
        <div className="flex items-center gap-2">
          <span className={cn("text-xs font-semibold", isUser ? "text-slate-300" : "text-teal-200")}>
            {isUser ? "You" : "MI Agent"}
          </span>
          <span className="text-[10px] text-ink-500">{formatTime(message.createdAt)}</span>
        </div>

        {message.pending ? (
          <div className="mt-1.5 inline-flex items-center gap-1 rounded-lg border border-teal-700/30 bg-teal-900/20 px-3 py-2">
            <span className="dot-1 h-1.5 w-1.5 rounded-full bg-teal-300" />
            <span className="dot-2 h-1.5 w-1.5 rounded-full bg-teal-300" />
            <span className="dot-3 h-1.5 w-1.5 rounded-full bg-teal-300" />
          </div>
        ) : (
          <div
            data-testid={isUser ? "user-bubble" : "assistant-bubble"}
            className={cn(
              "mt-1 whitespace-pre-wrap rounded-2xl rounded-tl-sm border px-3.5 py-2.5 text-[13px] leading-relaxed",
              isUser
                ? "border-slate-600/30 bg-slate-700/20 text-slate-100"
                : message.error
                  ? "border-rose-400/25 bg-rose-400/5 text-rose-200"
                  : "border-teal-700/30 bg-teal-900/15 text-ink-100",
            )}
          >
            {message.content}
          </div>
        )}

        {message.error && onRetry && (
          <button
            type="button"
            onClick={onRetry}
            className="mt-2 inline-flex items-center gap-1.5 rounded-md border border-[var(--color-line)] bg-navy-800 px-2.5 py-1 text-[11px] font-medium text-ink-200 transition-colors hover:border-teal-400/40 hover:text-ink-100"
          >
            <RefreshCw size={12} />
            Retry
          </button>
        )}

        {showDatasetBadge && (
          <div className="mt-2 flex flex-wrap items-center gap-1.5 text-[11px]">
            <span
              title="The book this answer was computed from"
              className="inline-flex items-center rounded-full border border-slate-500/30 bg-slate-700/30 px-2 py-0.5 font-medium uppercase tracking-wider text-slate-300"
            >
              {message.datasetContext}
            </span>
          </div>
        )}

        {/* Governed portfolio coverage. Every word — including the list of
            portfolios and the "not fully consolidated" statement — is authored
            by the backend. The chat renders it; it never works out
            consolidation status for itself, which is what stops a partial
            answer being read as a Total.
 
            The WEIGHT it is given is the browser's to decide, and the two cases
            do not deserve the same. Full consolidation is the unremarkable
            case: it gets one quiet line, not a titled bordered box competing
            with the answer above it. Partial consolidation changes what every
            number in the answer means, so it keeps the amber panel and its
            heading. The sentence itself is verbatim in both. */}
        {message.portfolioCoverage?.disclosure && (
          message.portfolioCoverage.is_fully_consolidated ? (
            <div
              data-testid="chat-portfolio-coverage"
              data-fully-consolidated="true"
              className="mt-1.5 text-[11px] leading-relaxed text-ink-500"
            >
              {message.portfolioCoverage.disclosure}
            </div>
          ) : (
            <div
              data-testid="chat-portfolio-coverage"
              data-fully-consolidated="false"
              className="mt-2 rounded-lg border border-amber-400/25 bg-amber-400/5 px-3 py-2 text-[11px] leading-relaxed text-amber-200/90"
            >
              <div className="text-[10px] font-semibold uppercase tracking-wider text-amber-300/80">
                Portfolio coverage
              </div>
              <div className="mt-1">{message.portfolioCoverage.disclosure}</div>
            </div>
          )
        )}

        {message.assumptions && message.assumptions.length > 0 && (
          <AssumptionsDisclosure assumptions={message.assumptions} />
        )}

        {/* The amber warnings box is deliberately never rendered here. On a
            refusal it restated the red bubble's own explanation in different
            words (confusing, and duplicative — see the backend fix that
            already stops the refusal text itself being appended twice); on a
            successful answer it competed with the coverage/assumptions
            disclosures above for the same "something to flag" attention.
            `message.warnings` is still carried on the message for the query
            audit panel and other non-chat consumers. */}

        {/* Says nothing at all unless the result has been cleared from the
            workspace. The chart, its figures and its controls live on the
            artifact — see ChatResult. */}
        {hasInlineResult && onTogglePin && (
          <ChatResult
            artifacts={message.artifacts!}
            onTogglePin={onTogglePin}
            onOpenArtifact={onOpenArtifact}
            workspaceArtifactIds={workspaceArtifactIds}
          />
        )}

        {/* Fallback navigation links only when the result isn't embedded inline. */}
        {!hasInlineResult && message.artifactRefs && message.artifactRefs.length > 0 && (
          <div className="mt-2 flex flex-col gap-1">
            {message.artifactRefs.map((ref) => {
              const stale = workspaceArtifactIds ? !workspaceArtifactIds.has(ref.id) : false;
              return (
                <button
                  key={ref.id}
                  type="button"
                  disabled={stale}
                  title={stale
                    ? "No longer in the workspace — ask the question again to regenerate it"
                    : undefined}
                  onClick={() => !stale && onOpenArtifact?.(ref.id)}
                  className={cn(
                    "group inline-flex items-center gap-2 rounded-md border border-[var(--color-line)] bg-navy-800/50 px-2.5 py-1.5 text-left text-[11px] text-ink-300 transition-colors",
                    stale
                      ? "cursor-not-allowed opacity-50"
                      : "hover:border-teal-400/40 hover:text-ink-100",
                  )}
                >
                  <FileBarChart size={13} className="text-teal-300" />
                  <span className="truncate">{ref.title}</span>
                  <span className="ml-auto text-[10px] uppercase tracking-wider text-ink-500 group-hover:text-teal-300">
                    {stale ? "cleared" : `${ref.type} →`}
                  </span>
                </button>
              );
            })}
          </div>
        )}

        {/* Follow-up suggestions are NOT rendered here. They used to be, which
            put two independently-built suggestion sets on screen at once — this
            row from `buildSuggestedActions`, and "Suggested investigations" on
            the artifact card from `buildInvestigations` — in two visual
            languages, for one chart. They now share the card's surface, beside
            the result they refer to rather than in a rail that has scrolled.
            `message.suggestions` is still carried on the message for callers
            that read the transcript. */}
      </div>
    </div>
  );
}

