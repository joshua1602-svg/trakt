import { useState } from "react";
import { ChevronDown } from "lucide-react";
import clsx from "clsx";
import type { Comparison, ComparisonFile } from "@/api/adminTypes";
import { copy } from "@/lib/copy";
import { EmptyNote } from "./primitives";

const CHANGE_STYLES: Record<string, string> = {
  added: "border-emerald-200 bg-emerald-50 text-emerald-800",
  changed: "border-amber-200 bg-amber-50 text-amber-800",
  removed: "border-rose-200 bg-rose-50 text-rose-800",
  unchanged: "border-stone-200 bg-stone-50 text-stone-500",
};

const CHANGE_LABELS: Record<string, string> = {
  added: copy.admin.compare.added,
  changed: copy.admin.compare.changed,
  removed: copy.admin.compare.removed,
  unchanged: copy.admin.compare.unchanged,
};

const LINE_STYLES: Record<string, string> = {
  added: "bg-emerald-50 text-emerald-900",
  removed: "bg-rose-50 text-rose-900",
  hunk: "bg-stone-100 text-stone-500",
  context: "text-stone-600",
};

function FileSection({ file }: { file: ComparisonFile }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="rounded-xl border border-stone-200 bg-white">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="flex w-full items-center justify-between gap-3 px-4 py-3 text-left"
      >
        <span className="flex min-w-0 items-center gap-3">
          <ChevronDown
            className={clsx("h-4 w-4 shrink-0 text-stone-400 transition-transform", open && "rotate-180")}
            aria-hidden
          />
          <span className="truncate text-sm font-medium text-stone-900">{file.label}</span>
        </span>
        <span
          className={clsx(
            "shrink-0 rounded-full border px-2.5 py-0.5 text-xs font-medium",
            CHANGE_STYLES[file.change],
          )}
        >
          {CHANGE_LABELS[file.change]}
        </span>
      </button>
      {open && (
        <div className="border-t border-stone-100 px-2 py-2">
          <pre className="overflow-x-auto whitespace-pre text-xs leading-relaxed">
            {file.lines.map((line, i) => (
              <div key={i} className={clsx("px-2 font-mono", LINE_STYLES[line.kind])}>
                {line.text}
              </div>
            ))}
          </pre>
          {file.truncated && (
            <p className="px-3 py-2 text-xs text-stone-400">{copy.admin.compare.truncated}</p>
          )}
        </div>
      )}
    </div>
  );
}

/**
 * A grouped comparison of two versions. Grouping and diffing are done by the
 * backend; the browser only renders what it is given.
 */
export function ConfigurationDiff({ comparison }: { comparison: Comparison }) {
  const changed = comparison.files.filter((f) => f.change !== "unchanged");
  const unchanged = comparison.files.filter((f) => f.change === "unchanged");

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        {(["added", "changed", "removed", "unchanged"] as const).map((key) => (
          <div
            key={key}
            data-summary={key}
            className="rounded-2xl border border-stone-200 bg-white px-4 py-3"
          >
            <p className="text-xl font-semibold text-stone-900">{comparison.summary[key]}</p>
            <p className="mt-0.5 text-xs text-stone-500">{CHANGE_LABELS[key]}</p>
          </div>
        ))}
      </div>

      <p className="text-sm text-stone-500">
        v{comparison.from.version} → v{comparison.to.version} · {comparison.dependencies.note}
      </p>

      {changed.length === 0 ? (
        <EmptyNote>{copy.admin.compare.identical}</EmptyNote>
      ) : (
        <div className="space-y-2">
          {changed.map((file) => (
            <FileSection key={file.path} file={file} />
          ))}
        </div>
      )}

      {unchanged.length > 0 && (
        <p className="text-xs text-stone-400">
          {unchanged.length} {copy.admin.compare.unchanged.toLowerCase()}:{" "}
          {unchanged.map((f) => f.label).join(", ")}
        </p>
      )}
    </div>
  );
}
