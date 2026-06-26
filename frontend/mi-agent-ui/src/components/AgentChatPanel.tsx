import { useEffect, useRef, useState } from "react";
import { CornerDownLeft, History, Sparkles, X } from "lucide-react";
import type { AnalysisContext } from "@/lib/analysisContext";
import { contextSummary } from "@/lib/analysisContext";
import type { Artifact, ChatMessage as ChatMessageType } from "@/domain";
import { ChatMessage } from "@/components/ChatMessage";
import { PromptSuggestions } from "@/components/PromptSuggestions";
import { cn } from "@/lib/utils";

export function AgentChatPanel({
  messages,
  isWorking,
  mock,
  onSubmit,
  onOpenArtifact,
  onRetry,
  context,
  onClearContext,
  onTogglePin,
  onDrill,
}: {
  messages: ChatMessageType[];
  isWorking: boolean;
  mock: boolean;
  onSubmit: (text: string) => void;
  onOpenArtifact: (id: string) => void;
  onRetry: () => void;
  context?: AnalysisContext | null;
  onClearContext?: () => void;
  onTogglePin?: (id: string) => void;
  onDrill?: (artifact: Artifact, filters: Record<string, unknown>) => void;
}) {
  const [input, setInput] = useState("");
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    scrollRef.current?.scrollTo({
      top: scrollRef.current.scrollHeight,
      behavior: "smooth",
    });
  }, [messages]);

  const submit = () => {
    const text = input.trim();
    if (!text || isWorking) return;
    onSubmit(text);
    setInput("");
  };

  return (
    <aside
      data-surface="ai-chat"
      className="flex h-full w-[460px] shrink-0 flex-col border-r border-teal-800/30 bg-gradient-to-b from-teal-950/40 to-navy-950/60"
    >
      <header className="flex items-center gap-2.5 border-b border-teal-800/30 bg-teal-950/30 px-5 py-3.5">
        <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-gradient-to-br from-teal-500 to-emerald-600 text-white shadow-sm shadow-teal-900/40">
          <Sparkles size={16} />
        </div>
        <div>
          <h1 className="text-sm font-semibold text-teal-50">MI Agent</h1>
          <p className="text-[11px] text-teal-200/70">Your portfolio analyst</p>
        </div>
        <span
          className={cn(
            "ml-auto inline-flex items-center gap-1.5 rounded-full border px-2 py-0.5 text-[10px] font-medium",
            mock
              ? "border-amber-400/30 bg-amber-400/10 text-amber-400"
              : "border-emerald-400/30 bg-emerald-400/10 text-emerald-300",
          )}
        >
          <span className={cn("h-1.5 w-1.5 rounded-full", mock ? "bg-amber-400" : "bg-emerald-400")} />
          {mock ? "Demo data" : "Online"}
        </span>
      </header>

      <div ref={scrollRef} className="min-h-0 flex-1 space-y-5 overflow-y-auto px-5 py-4">
        {messages.map((m) => (
          <ChatMessage
            key={m.id}
            message={m}
            onOpenArtifact={onOpenArtifact}
            onRetry={onRetry}
            onAsk={onSubmit}
            onTogglePin={onTogglePin}
            onDrill={onDrill}
          />
        ))}

        {messages.length <= 1 && (
          <div className="pt-1">
            <PromptSuggestions onPick={onSubmit} />
          </div>
        )}
      </div>

      {contextSummary(context) && (
        <div className="flex items-center gap-1.5 border-t border-teal-800/30 bg-teal-950/30 px-3 py-1.5">
          <History size={12} className="shrink-0 text-teal-300" />
          <span className="truncate text-[11px] text-teal-100/80">
            <span className="text-teal-300/70">Context:</span> {contextSummary(context)}
          </span>
          {onClearContext && (
            <button
              type="button"
              onClick={onClearContext}
              aria-label="Clear context"
              className="ml-auto inline-flex items-center gap-0.5 rounded px-1 py-0.5 text-[10px] text-teal-200/70 transition-colors hover:text-teal-100"
            >
              <X size={11} /> Clear
            </button>
          )}
        </div>
      )}

      <div className="border-t border-teal-800/30 bg-teal-950/20 p-3">
        <div className="rounded-xl border border-teal-800/40 bg-navy-950/60 focus-within:border-teal-400/60">
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                submit();
              }
            }}
            rows={2}
            placeholder="Ask about balances, concentration, pipeline, static pools or validation…"
            className="w-full resize-none bg-transparent px-3.5 py-2.5 text-[13px] text-ink-100 placeholder:text-ink-500 focus:outline-none"
          />
          <div className="flex items-center justify-between px-3 pb-2.5">
            <span className="text-[10px] text-ink-500">
              <kbd className="rounded bg-navy-800 px-1 py-0.5 font-sans">Enter</kbd> to send ·{" "}
              <kbd className="rounded bg-navy-800 px-1 py-0.5 font-sans">Shift+Enter</kbd> newline
            </span>
            <button
              type="button"
              onClick={submit}
              disabled={!input.trim() || isWorking}
              className="inline-flex items-center gap-1.5 rounded-lg bg-gradient-to-br from-teal-500 to-emerald-600 px-3 py-1.5 text-xs font-semibold text-white shadow-sm shadow-teal-900/30 transition-opacity hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-40"
            >
              Send
              <CornerDownLeft size={13} />
            </button>
          </div>
        </div>
      </div>
    </aside>
  );
}
