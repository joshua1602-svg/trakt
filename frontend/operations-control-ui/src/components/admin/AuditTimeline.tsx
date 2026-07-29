import clsx from "clsx";
import type { AuditEntry } from "@/api/adminTypes";
import { copy } from "@/lib/copy";
import { formatDate } from "@/lib/format";
import { EmptyNote } from "./primitives";

const DOT: Record<string, string> = {
  config_activated: "bg-emerald-500",
  config_rolled_back: "bg-amber-500",
  config_draft_created: "bg-stone-400",
  config_validated: "bg-blue-500",
};

function timeOf(at: string): string {
  const date = new Date(at);
  if (Number.isNaN(date.getTime())) return "";
  return date.toLocaleTimeString("en-GB", { hour: "2-digit", minute: "2-digit" });
}

/** Draft, validation, activation and rollback events, newest first. */
export function AuditTimeline({ entries }: { entries: AuditEntry[] }) {
  if (entries.length === 0) {
    return <EmptyNote>{copy.admin.audit.empty}</EmptyNote>;
  }
  return (
    <ol className="space-y-4">
      {entries.map((entry) => (
        <li key={entry.seq} data-event={entry.event} className="flex gap-4">
          <div className="flex flex-col items-center pt-1.5">
            <span
              className={clsx(
                "h-2.5 w-2.5 shrink-0 rounded-full",
                entry.outcome === "failed" ? "bg-rose-500" : DOT[entry.event] ?? "bg-stone-300",
              )}
              aria-hidden
            />
            <span className="mt-1 w-px flex-1 bg-stone-200" aria-hidden />
          </div>
          <div className="min-w-0 flex-1 pb-2">
            <p className="text-xs text-stone-500">
              {formatDate(entry.at)}
              {timeOf(entry.at) ? `, ${timeOf(entry.at)}` : ""} · {entry.event_label}
            </p>
            <p className="mt-1 text-sm leading-relaxed text-stone-800">{entry.description}</p>
            {entry.notes && <p className="mt-1 text-sm text-stone-500">{entry.notes}</p>}
          </div>
        </li>
      ))}
    </ol>
  );
}
