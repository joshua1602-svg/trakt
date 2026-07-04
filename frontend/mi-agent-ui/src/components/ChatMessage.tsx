import { AlertTriangle, FileBarChart, RefreshCw, Sparkles, User } from "lucide-react";
import type { ChatMessage as ChatMessageType } from "@/domain";
import { ChatResult } from "@/components/ChatResult";
import { cn, formatTime } from "@/lib/utils";

/** Trim engineer-only segments (Parser / Validation / Chart type) from the
 * backend interpretation so the chat shows the analytical reading — "what did
 * it think I asked" — without the routing internals. */
function summariseInterpretation(interpreted?: string): string | undefined {
  if (!interpreted) return undefined;
  const kept = interpreted
    .split(" · ")
    .filter((seg) => !/^\s*(parser|validation|chart)\s*:/i.test(seg))
    .join(" · ")
    .trim();
  return kept.length > 0 ? kept : undefined;
}

export function ChatMessage({
  message,
  onOpenArtifact,
  onRetry,
  onAsk,
  onTogglePin,
}: {
  message: ChatMessageType;
  onOpenArtifact?: (id: string) => void;
  onRetry?: () => void;
  /** Dispatch a suggested follow-up question (routes through context). */
  onAsk?: (question: string) => void;
  onTogglePin?: (id: string) => void;
}) {
  const isUser = message.role === "user";
  const hasInlineResult = !isUser && !!message.artifacts && message.artifacts.length > 0;
  const interpretation = summariseInterpretation(message.interpreted);
  const showProvenance =
    !isUser && !message.pending && !message.error && (!!interpretation || !!message.datasetContext);

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

        {showProvenance && (
          <div className="mt-2 flex flex-wrap items-center gap-1.5 text-[11px]">
            {message.datasetContext && (
              <span
                title="The dataset this answer was computed from"
                className="inline-flex items-center rounded-full border border-slate-500/30 bg-slate-700/30 px-2 py-0.5 font-medium uppercase tracking-wider text-slate-300"
              >
                {message.datasetContext}
              </span>
            )}
            {message.parserMode && (
              <span
                title="How the question was parsed"
                className="inline-flex items-center rounded-full border border-slate-500/20 bg-slate-800/40 px-2 py-0.5 text-[10px] uppercase tracking-wider text-ink-500"
              >
                {message.parserMode === "llm" ? "AI parse" : "rules parse"}
              </span>
            )}
            {interpretation && (
              <span className="text-ink-400">
                <span className="text-ink-500">Interpreted as:</span> {interpretation}
              </span>
            )}
          </div>
        )}

        {message.assumptions && message.assumptions.length > 0 && (
          <div className="mt-2 rounded-lg border border-[var(--color-line-soft)] bg-navy-900/50 px-3 py-2">
            <div className="text-[10px] font-semibold uppercase tracking-wider text-ink-500">Assumptions</div>
            <ul className="mt-1 list-disc space-y-0.5 pl-4 text-[11px] leading-relaxed text-ink-400">
              {message.assumptions.map((a, i) => (
                <li key={i}>{a}</li>
              ))}
            </ul>
          </div>
        )}

        {message.warnings && message.warnings.length > 0 && (
          <div className="mt-2 rounded-lg border border-amber-400/20 bg-amber-400/5 px-3 py-2 text-[11px] text-amber-300/90">
            {message.warnings.map((w, i) => (
              <div key={i}>⚠ {w}</div>
            ))}
          </div>
        )}

        {/* A compact result summary + links; the full chart/table/artifact lives
            in the Artifact Workspace (never duplicated inline in the chat). */}
        {hasInlineResult && onTogglePin && (
          <ChatResult
            artifacts={message.artifacts!}
            onTogglePin={onTogglePin}
            onOpenArtifact={onOpenArtifact}
          />
        )}

        {/* Fallback navigation links only when the result isn't embedded inline. */}
        {!hasInlineResult && message.artifactRefs && message.artifactRefs.length > 0 && (
          <div className="mt-2 flex flex-col gap-1">
            {message.artifactRefs.map((ref) => (
              <button
                key={ref.id}
                type="button"
                onClick={() => onOpenArtifact?.(ref.id)}
                className="group inline-flex items-center gap-2 rounded-md border border-[var(--color-line)] bg-navy-800/50 px-2.5 py-1.5 text-left text-[11px] text-ink-300 transition-colors hover:border-teal-400/40 hover:text-ink-100"
              >
                <FileBarChart size={13} className="text-teal-300" />
                <span className="truncate">{ref.title}</span>
                <span className="ml-auto text-[10px] uppercase tracking-wider text-ink-500 group-hover:text-teal-300">
                  {ref.type} →
                </span>
              </button>
            ))}
          </div>
        )}

        {!isUser && !message.pending && !message.error && message.suggestions && message.suggestions.length > 0 && (
          <div className="mt-2 flex flex-wrap gap-1.5">
            {message.suggestions.map((s) => (
              <button
                key={`${s.kind}:${s.question}`}
                type="button"
                onClick={() => onAsk?.(s.question)}
                title={s.question}
                className="inline-flex items-center rounded-full border border-teal-700/30 bg-teal-900/20 px-2.5 py-1 text-[11px] text-teal-100 transition-colors hover:border-teal-400/50 hover:bg-teal-800/30"
              >
                {s.label}
              </button>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

