import { useEffect, useRef, useState } from "react";
import {
  CornerDownLeft,
  Eraser,
  History,
  Lightbulb,
  PanelLeftClose,
  Sparkles,
  Square,
  X,
} from "lucide-react";
import type { AnalysisContext } from "@/lib/analysisContext";
import { contextSummary, looksLikeFollowUp } from "@/lib/analysisContext";
import type { ChatMessage as ChatMessageType } from "@/domain";
import { ChatMessage } from "@/components/ChatMessage";
import { PromptSuggestions } from "@/components/PromptSuggestions";
import { cn } from "@/lib/utils";

export function AgentChatPanel({
  messages,
  isWorking,
  mock,
  initialInput,
  onSubmit,
  onStop,
  onOpenArtifact,
  onRetry,
  context,
  onClearContext,
  onClearChat,
  onTogglePin,
  workspaceArtifactIds,
}: {
  messages: ChatMessageType[];
  isWorking: boolean;
  mock: boolean;
  /** Pre-filled composer text (e.g. a Copilot "Open in Workspace" deep-link
   *  question). Applied once on mount; the user edits or sends it. */
  initialInput?: string;
  onSubmit: (text: string) => void;
  /** Cancel the in-flight request (renders a Stop control while working). */
  onStop?: () => void;
  onOpenArtifact: (id: string) => void;
  onRetry: () => void;
  context?: AnalysisContext | null;
  onClearContext?: () => void;
  /** Reset the conversation to the greeting (loaded MI data is untouched). */
  onClearChat?: () => void;
  onTogglePin?: (id: string) => void;
  /** Ids currently present in the artifact workspace, so links to artifacts
   *  that were since cleared render as stale rather than silently no-op. */
  workspaceArtifactIds?: Set<string>;
}) {
  const [input, setInput] = useState(initialInput ?? "");
  // Collapse to a slim rail — the only region that couldn't be collapsed.
  const [collapsed, setCollapsed] = useState<boolean>(
    () => (typeof localStorage !== "undefined"
      && localStorage.getItem("mi.chat.collapsed") === "1"));
  useEffect(() => {
    if (typeof localStorage !== "undefined") {
      localStorage.setItem("mi.chat.collapsed", collapsed ? "1" : "0");
    }
  }, [collapsed]);
  // Re-open the suggested questions after the conversation has started.
  const [showSuggestions, setShowSuggestions] = useState(false);
  // The context bar answers one question — "what will a follow-up attach to?"
  // — so it appears when that question is live: the composer has focus, or what
  // has been typed already reads as a follow-up. Permanently on, it was a third
  // printing of the measure and dimensions the answer's execution receipt and
  // the artifact's own title already carry.
  const [composerFocused, setComposerFocused] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);
  // Follow the conversation only while the reader is at (or near) the bottom;
  // never yank the viewport away from someone re-reading history.
  const nearBottom = useRef(true);

  useEffect(() => {
    if (!nearBottom.current) return;
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
    setShowSuggestions(false);
    nearBottom.current = true; // sending returns the reader to the live tail
  };

  if (collapsed) {
    return (
      <aside
        data-surface="ai-chat"
        className="flex h-full w-12 shrink-0 flex-col items-center border-r border-teal-800/30 bg-[var(--surface-chat)] py-3"
      >
        <button
          type="button"
          onClick={() => setCollapsed(false)}
          aria-label="Expand chat"
          aria-expanded={false}
          title="Expand the MI Agent chat"
          className="flex h-9 w-9 items-center justify-center rounded-lg bg-gradient-to-br from-teal-500 to-emerald-600 text-white shadow-sm shadow-teal-900/40 transition-opacity hover:opacity-90"
        >
          <Sparkles size={18} />
        </button>
        <span
          className="mt-3 text-[10px] font-semibold uppercase tracking-wider text-teal-200/70"
          style={{ writingMode: "vertical-rl" }}
        >
          MI Agent
        </span>
        {isWorking && (
          <span className="mt-3 h-1.5 w-1.5 animate-pulse rounded-full bg-teal-300" title="Working…" />
        )}
      </aside>
    );
  }

  return (
    <aside
      data-surface="ai-chat"
      className="flex h-full w-[380px] shrink-0 flex-col border-r border-teal-800/30 bg-[var(--surface-chat)] xl:w-[460px]"
    >
      {/* One flat fill for the whole window (the base colour requested),
          matching how Core Dashboard and Artifact Workspace hold a single
          surface colour with a border-only header — not a separate header
          tint on top of a gradient. */}
      <header className="flex items-center gap-2.5 border-b border-teal-800/30 px-5 py-3.5">
        <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-gradient-to-br from-teal-500 to-emerald-600 text-white shadow-sm shadow-teal-900/40">
          <Sparkles size={18} />
        </div>
        <h1 className="text-base font-semibold text-teal-50">MI Agent</h1>
        <span
          className="inline-flex items-center rounded-full border border-teal-700/40 bg-teal-800/20 px-1.5 py-0.5 text-[9px] font-semibold uppercase tracking-wider text-teal-200/80"
          title="In active development — behaviour and answers may still change"
        >
          Beta
        </span>
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
        {onClearChat && messages.length > 1 && (
          <button
            type="button"
            onClick={onClearChat}
            aria-label="Clear chat"
            title="Clear the conversation (loaded MI data is untouched)"
            className="inline-flex items-center gap-1 rounded-md px-2 py-1 text-[11px] font-medium text-teal-200/70 hover:text-rose-300"
          >
            <Eraser size={13} /> Clear
          </button>
        )}
        <button
          type="button"
          onClick={() => setCollapsed(true)}
          aria-label="Collapse chat"
          aria-expanded={true}
          title="Collapse the chat to a slim rail"
          className="inline-flex items-center rounded-md px-1.5 py-1 text-teal-200/70 hover:text-teal-100"
        >
          <PanelLeftClose size={15} />
        </button>
      </header>

      <div
        ref={scrollRef}
        onScroll={(e) => {
          const el = e.currentTarget;
          nearBottom.current = el.scrollHeight - el.scrollTop - el.clientHeight < 120;
        }}
        className="min-h-0 flex-1 space-y-5 overflow-y-auto px-5 py-4"
      >
        {messages.map((m) => (
          <ChatMessage
            key={m.id}
            message={m}
            onOpenArtifact={onOpenArtifact}
            onRetry={onRetry}
            onTogglePin={onTogglePin}
            workspaceArtifactIds={workspaceArtifactIds}
          />
        ))}

        {messages.length <= 1 && (
          <div className="pt-1">
            <PromptSuggestions onPick={onSubmit} />
          </div>
        )}
      </div>

      {contextSummary(context) && (composerFocused || looksLikeFollowUp(input, context)) && (
        <div
          data-testid="chat-context-bar"
          className="flex items-center gap-1.5 border-t border-teal-800/30 bg-teal-950/30 px-3 py-1.5"
        >
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

      {/* Suggested questions stay reachable mid-conversation, not only on the
          empty state. */}
      {showSuggestions && messages.length > 1 && (
        <div className="max-h-56 overflow-y-auto border-t border-teal-800/30 bg-teal-950/20 px-3 py-2.5">
          <PromptSuggestions
            onPick={(q) => {
              setShowSuggestions(false);
              onSubmit(q);
            }}
          />
        </div>
      )}

      <div className="border-t border-teal-800/30 bg-teal-950/20 p-3">
        <div className="rounded-xl border border-teal-800/40 bg-navy-950/60 focus-within:border-teal-400/60">
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onFocus={() => setComposerFocused(true)}
            // Keep the bar up while the Clear-context button is being clicked;
            // it sits inside the panel, so a bare blur would remove the control
            // from under the pointer.
            onBlur={(e) => {
              const next = e.relatedTarget as Node | null;
              if (!next || !e.currentTarget.closest("aside")?.contains(next)) {
                setComposerFocused(false);
              }
            }}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                submit();
              }
            }}
            rows={2}
            placeholder="Ask me about your portfolio…"
            className="w-full resize-none bg-transparent px-3.5 py-2.5 text-[13px] text-ink-100 placeholder:text-ink-500 focus:outline-none"
          />
          <div className="flex items-center justify-between px-3 pb-2.5">
            <span className="flex items-center gap-2 text-[10px] text-ink-500">
              {messages.length > 1 && (
                <button
                  type="button"
                  onClick={() => setShowSuggestions((s) => !s)}
                  aria-label="Suggested questions"
                  aria-expanded={showSuggestions}
                  title="Show suggested questions"
                  className={cn(
                    "inline-flex items-center gap-1 rounded-md px-1.5 py-0.5 text-[10px] font-medium transition-colors",
                    showSuggestions
                      ? "text-teal-200"
                      : "text-ink-500 hover:text-teal-200",
                  )}
                >
                  <Lightbulb size={12} /> Suggestions
                </button>
              )}
              <span>
                <kbd className="rounded bg-navy-800 px-1 py-0.5 font-sans">Enter</kbd> to send ·{" "}
                <kbd className="rounded bg-navy-800 px-1 py-0.5 font-sans">Shift+Enter</kbd> newline
              </span>
            </span>
            {isWorking && onStop ? (
              <button
                type="button"
                onClick={onStop}
                aria-label="Stop"
                title="Cancel the running request"
                className="inline-flex items-center gap-1.5 rounded-lg border border-rose-400/40 bg-rose-400/10 px-3 py-1.5 text-xs font-semibold text-rose-200 transition-colors hover:bg-rose-400/20"
              >
                <Square size={11} className="fill-current" />
                Stop
              </button>
            ) : (
              <button
                type="button"
                onClick={submit}
                disabled={!input.trim() || isWorking}
                className="inline-flex items-center gap-1.5 rounded-lg bg-gradient-to-br from-teal-500 to-emerald-600 px-3 py-1.5 text-xs font-semibold text-white shadow-sm shadow-teal-900/30 transition-opacity hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-40"
              >
                Send
                <CornerDownLeft size={13} />
              </button>
            )}
          </div>
        </div>
      </div>
    </aside>
  );
}
