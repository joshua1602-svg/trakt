import { copy } from "@/lib/copy";

/** Inline plain-English error with a retry action. */
export function ErrorNote({ message, onRetry }: { message: string; onRetry?: () => void }) {
  return (
    <div className="flex items-center justify-between gap-4 rounded-xl border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-800">
      <span>{message}</span>
      {onRetry && (
        <button
          type="button"
          onClick={onRetry}
          className="shrink-0 rounded-lg border border-rose-300 bg-white px-3 py-1 text-xs font-medium text-rose-700 hover:bg-rose-100"
        >
          {copy.errors.retry}
        </button>
      )}
    </div>
  );
}

export function Loading() {
  return <p className="py-8 text-center text-sm text-stone-400">{copy.common.loading}</p>;
}
