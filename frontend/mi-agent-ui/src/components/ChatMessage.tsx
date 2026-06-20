import { AlertTriangle, FileBarChart, RefreshCw, Sparkles, User } from "lucide-react";
import type { ChatMessage as ChatMessageType } from "@/domain";
import { cn, formatTime } from "@/lib/utils";

export function ChatMessage({
  message,
  onOpenArtifact,
  onRetry,
}: {
  message: ChatMessageType;
  onOpenArtifact?: (id: string) => void;
  onRetry?: () => void;
}) {
  const isUser = message.role === "user";

  return (
    <div className="animate-fade-in flex gap-2.5">
      <div
        className={cn(
          "mt-0.5 flex h-7 w-7 shrink-0 items-center justify-center rounded-md",
          isUser
            ? "bg-navy-700 text-ink-300"
            : message.error
              ? "bg-rose-400/15 text-rose-400"
              : "bg-gradient-to-br from-peri-400/30 to-navy-700 text-peri-200",
        )}
      >
        {isUser ? <User size={14} /> : message.error ? <AlertTriangle size={14} /> : <Sparkles size={14} />}
      </div>

      <div className="min-w-0 flex-1">
        <div className="flex items-center gap-2">
          <span className="text-xs font-semibold text-ink-200">{isUser ? "You" : "MI Agent"}</span>
          <span className="text-[10px] text-ink-500">{formatTime(message.createdAt)}</span>
        </div>

        {message.pending ? (
          <div className="mt-1.5 inline-flex items-center gap-1 rounded-lg bg-navy-800/60 px-3 py-2">
            <span className="dot-1 h-1.5 w-1.5 rounded-full bg-peri-300" />
            <span className="dot-2 h-1.5 w-1.5 rounded-full bg-peri-300" />
            <span className="dot-3 h-1.5 w-1.5 rounded-full bg-peri-300" />
          </div>
        ) : (
          <p
            className={cn(
              "mt-1 whitespace-pre-wrap text-[13px] leading-relaxed",
              isUser ? "text-ink-200" : message.error ? "text-rose-300" : "text-ink-300",
            )}
          >
            {message.content}
          </p>
        )}

        {message.error && onRetry && (
          <button
            type="button"
            onClick={onRetry}
            className="mt-2 inline-flex items-center gap-1.5 rounded-md border border-[var(--color-line)] bg-navy-800 px-2.5 py-1 text-[11px] font-medium text-ink-200 transition-colors hover:border-peri-400/40 hover:text-ink-100"
          >
            <RefreshCw size={12} />
            Retry
          </button>
        )}

        {message.interpreted && (
          <div className="mt-2 inline-flex items-center gap-1.5 rounded-md border border-[var(--color-line-soft)] bg-navy-900/60 px-2 py-1 font-mono text-[10px] text-ink-400">
            <span className="text-ink-500">interpreted as</span>
            {message.interpreted}
          </div>
        )}

        {message.assumptions && message.assumptions.length > 0 && (
          <div className="mt-2 rounded-lg border border-[var(--color-line-soft)] bg-navy-900/60 px-3 py-2">
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

        {message.artifactRefs && message.artifactRefs.length > 0 && (
          <div className="mt-2 flex flex-col gap-1">
            {message.artifactRefs.map((ref) => (
              <button
                key={ref.id}
                type="button"
                onClick={() => onOpenArtifact?.(ref.id)}
                className="group inline-flex items-center gap-2 rounded-md border border-[var(--color-line)] bg-navy-800/50 px-2.5 py-1.5 text-left text-[11px] text-ink-300 transition-colors hover:border-peri-400/40 hover:text-ink-100"
              >
                <FileBarChart size={13} className="text-peri-300" />
                <span className="truncate">{ref.title}</span>
                <span className="ml-auto text-[10px] uppercase tracking-wider text-ink-500 group-hover:text-peri-300">
                  {ref.type} →
                </span>
              </button>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
