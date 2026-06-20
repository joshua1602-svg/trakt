import { FileBarChart, Sparkles, User } from "lucide-react";
import type { ChatMessage as ChatMessageType } from "@/types";
import { cn, formatTime } from "@/lib/utils";

export function ChatMessage({
  message,
  onOpenArtifact,
}: {
  message: ChatMessageType;
  onOpenArtifact?: (id: string) => void;
}) {
  const isUser = message.role === "user";

  return (
    <div className="animate-fade-in flex gap-2.5">
      <div
        className={cn(
          "mt-0.5 flex h-7 w-7 shrink-0 items-center justify-center rounded-md",
          isUser
            ? "bg-navy-700 text-ink-300"
            : "bg-gradient-to-br from-peri-400/30 to-navy-700 text-peri-200",
        )}
      >
        {isUser ? <User size={14} /> : <Sparkles size={14} />}
      </div>

      <div className="min-w-0 flex-1">
        <div className="flex items-center gap-2">
          <span className="text-xs font-semibold text-ink-200">
            {isUser ? "You" : "MI Agent"}
          </span>
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
              isUser ? "text-ink-200" : "text-ink-300",
            )}
          >
            {message.content}
          </p>
        )}

        {message.assumptions && message.assumptions.length > 0 && (
          <div className="mt-2 rounded-lg border border-[var(--color-line-soft)] bg-navy-900/60 px-3 py-2">
            <div className="text-[10px] font-semibold uppercase tracking-wider text-ink-500">
              Assumptions
            </div>
            <ul className="mt-1 list-disc space-y-0.5 pl-4 text-[11px] leading-relaxed text-ink-400">
              {message.assumptions.map((a, i) => (
                <li key={i}>{a}</li>
              ))}
            </ul>
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
                <span className="ml-auto text-[10px] text-ink-500 group-hover:text-peri-300">
                  view →
                </span>
              </button>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
