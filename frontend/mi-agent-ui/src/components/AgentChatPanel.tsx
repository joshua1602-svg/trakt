import { useEffect, useRef, useState } from "react";
import { CornerDownLeft, Sparkles } from "lucide-react";
import type { ChatMessage as ChatMessageType } from "@/domain";
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
}: {
  messages: ChatMessageType[];
  isWorking: boolean;
  mock: boolean;
  onSubmit: (text: string) => void;
  onOpenArtifact: (id: string) => void;
  onRetry: () => void;
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
    <aside className="flex h-full w-[400px] shrink-0 flex-col border-r border-[var(--color-line)] bg-navy-900/40">
      <header className="flex items-center gap-2.5 border-b border-[var(--color-line)] px-5 py-3.5">
        <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-gradient-to-br from-peri-400/40 to-navy-700 text-peri-100">
          <Sparkles size={16} />
        </div>
        <div>
          <h1 className="text-sm font-semibold text-ink-100">MI Agent</h1>
          <p className="text-[11px] text-ink-400">Portfolio intelligence assistant</p>
        </div>
        <span
          className={cn(
            "ml-auto inline-flex items-center gap-1.5 rounded-full border px-2 py-0.5 text-[10px] font-medium",
            mock
              ? "border-amber-400/30 bg-amber-400/10 text-amber-400"
              : "border-mint-400/30 bg-mint-400/10 text-mint-400",
          )}
        >
          <span className={cn("h-1.5 w-1.5 rounded-full", mock ? "bg-amber-400" : "bg-mint-400")} />
          {mock ? "Mock Agent" : "Online"}
        </span>
      </header>

      <div ref={scrollRef} className="min-h-0 flex-1 space-y-5 overflow-y-auto px-5 py-4">
        {messages.map((m) => (
          <ChatMessage key={m.id} message={m} onOpenArtifact={onOpenArtifact} onRetry={onRetry} />
        ))}

        {messages.length <= 1 && (
          <div className="pt-1">
            <PromptSuggestions onPick={onSubmit} />
          </div>
        )}
      </div>

      <div className="border-t border-[var(--color-line)] p-3">
        <div className="rounded-xl border border-[var(--color-line)] bg-navy-950/60 focus-within:border-peri-400/50">
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
              className="inline-flex items-center gap-1.5 rounded-lg bg-peri-400 px-3 py-1.5 text-xs font-semibold text-navy-950 transition-colors hover:bg-peri-300 disabled:cursor-not-allowed disabled:opacity-40"
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
