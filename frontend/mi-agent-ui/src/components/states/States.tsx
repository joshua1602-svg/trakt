/** Shared empty / loading / error states for the artifact canvas. */
import { AlertTriangle, Inbox, RefreshCw, Sparkles } from "lucide-react";
import type { ReactNode } from "react";

export function LoadingState() {
  return (
    <div className="mb-4 flex items-center gap-2 rounded-lg border border-peri-400/20 bg-navy-800/40 px-4 py-3 text-sm text-peri-200">
      <Sparkles size={15} className="text-peri-300" />
      MI Agent is composing artifacts
      <span className="ml-1 inline-flex gap-0.5">
        <span className="dot-1 h-1 w-1 rounded-full bg-peri-300" />
        <span className="dot-2 h-1 w-1 rounded-full bg-peri-300" />
        <span className="dot-3 h-1 w-1 rounded-full bg-peri-300" />
      </span>
    </div>
  );
}

export function EmptyState({
  title = "No artifacts yet",
  hint = "Ask the MI Agent a question to generate analysis.",
  icon,
}: {
  title?: string;
  hint?: string;
  icon?: ReactNode;
}) {
  return (
    <div className="flex h-72 flex-col items-center justify-center text-center">
      <div className="flex h-12 w-12 items-center justify-center rounded-xl border border-[var(--color-line)] bg-navy-850/60 text-ink-500">
        {icon ?? <Inbox size={22} />}
      </div>
      <div className="mt-3 text-sm font-medium text-ink-200">{title}</div>
      <p className="mt-1 max-w-xs text-xs text-ink-500">{hint}</p>
    </div>
  );
}

export function ErrorState({ message, onRetry }: { message: string; onRetry?: () => void }) {
  return (
    <div className="flex flex-col items-center justify-center rounded-xl border border-rose-400/20 bg-rose-400/5 px-6 py-8 text-center">
      <div className="flex h-11 w-11 items-center justify-center rounded-xl bg-rose-400/10 text-rose-400">
        <AlertTriangle size={20} />
      </div>
      <div className="mt-3 text-sm font-medium text-ink-100">Something went wrong</div>
      <p className="mt-1 max-w-sm text-xs text-ink-400">{message}</p>
      {onRetry && (
        <button
          type="button"
          onClick={onRetry}
          className="mt-3 inline-flex items-center gap-1.5 rounded-lg border border-[var(--color-line)] bg-navy-800 px-3 py-1.5 text-xs font-medium text-ink-200 transition-colors hover:border-peri-400/40 hover:text-ink-100"
        >
          <RefreshCw size={13} />
          Retry
        </button>
      )}
    </div>
  );
}
