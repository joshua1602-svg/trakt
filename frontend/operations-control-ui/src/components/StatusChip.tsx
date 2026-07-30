import clsx from "clsx";
import { copy } from "@/lib/copy";
import { humanize } from "@/lib/format";

const STYLES: Record<string, string> = {
  // green tint
  completed: "bg-emerald-50 text-emerald-700 border-emerald-200",
  published: "bg-emerald-50 text-emerald-700 border-emerald-200",
  // green outline
  approved: "bg-white text-emerald-700 border-emerald-400",
  // amber
  needs_review: "bg-amber-50 text-amber-800 border-amber-200",
  open: "bg-amber-50 text-amber-800 border-amber-200",
  incomplete: "bg-amber-50 text-amber-800 border-amber-200",
  review_required: "bg-amber-50 text-amber-800 border-amber-200",
  configuration_required: "bg-amber-50 text-amber-800 border-amber-200",
  // red tint
  blocked: "bg-rose-50 text-rose-700 border-rose-200",
  failed: "bg-rose-50 text-rose-700 border-rose-200",
  rejected: "bg-rose-50 text-rose-700 border-rose-200",
  // blue pulse
  running: "bg-blue-50 text-blue-700 border-blue-200 animate-pulse",
  receiving: "bg-blue-50 text-blue-700 border-blue-200 animate-pulse",
  classifying: "bg-blue-50 text-blue-700 border-blue-200 animate-pulse",
  // violet
  ready: "bg-violet-50 text-violet-700 border-violet-200",
  awaiting_publication: "bg-violet-50 text-violet-700 border-violet-200",
  prepared: "bg-violet-50 text-violet-700 border-violet-200",
  // configuration package lifecycle
  active: "bg-emerald-50 text-emerald-700 border-emerald-200",
  valid: "bg-emerald-50 text-emerald-700 border-emerald-200",
  passed: "bg-emerald-50 text-emerald-700 border-emerald-200",
  draft: "bg-amber-50 text-amber-800 border-amber-200 border-dashed",
  not_checked: "bg-amber-50 text-amber-800 border-amber-200 border-dashed",
  failed_checks: "bg-rose-50 text-rose-700 border-rose-200",
  superseded: "bg-stone-100 text-stone-500 border-stone-200",
  // delivery workflow steps
  complete: "bg-emerald-50 text-emerald-700 border-emerald-200",
  current: "bg-blue-50 text-blue-700 border-blue-200",
  pending: "bg-stone-100 text-stone-500 border-stone-200",
  not_applicable: "bg-stone-50 text-stone-400 border-stone-200 border-dashed",
  // grey
  waiting: "bg-stone-100 text-stone-500 border-stone-200",
  received: "bg-stone-100 text-stone-600 border-stone-200",
  held: "bg-stone-100 text-stone-600 border-stone-200",
  cancelled: "bg-stone-100 text-stone-500 border-stone-200",
  withdrawn: "bg-stone-100 text-stone-600 border-stone-300 border-dashed",
  resolved: "bg-stone-100 text-stone-600 border-stone-200",
};

export function statusLabelFor(status: string): string {
  return copy.statusLabels[status] ?? humanize(status);
}

/**
 * Soft status chip. Prefers the server-provided label when given;
 * falls back to our own plain-English label for the status code.
 */
export function StatusChip({
  status,
  label,
  className,
}: {
  status: string;
  label?: string;
  className?: string;
}) {
  return (
    <span
      data-status={status}
      className={clsx(
        "inline-flex items-center whitespace-nowrap rounded-full border px-2.5 py-0.5 text-xs font-medium",
        STYLES[status] ?? "bg-stone-100 text-stone-600 border-stone-200",
        className,
      )}
    >
      {label || statusLabelFor(status)}
    </span>
  );
}
