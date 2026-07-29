import { AlertTriangle, CheckCircle2, CircleDashed, OctagonAlert } from "lucide-react";
import clsx from "clsx";
import type { ValidationSummary } from "@/api/adminTypes";
import { copy } from "@/lib/copy";
import { formatDate } from "@/lib/format";
import { TechnicalDetails } from "./primitives";

const TONE = {
  passed: {
    box: "border-emerald-200 bg-emerald-50 text-emerald-900",
    Icon: CheckCircle2,
  },
  failed: { box: "border-rose-200 bg-rose-50 text-rose-900", Icon: OctagonAlert },
  not_checked: { box: "border-amber-200 bg-amber-50 text-amber-900", Icon: CircleDashed },
} as const;

/**
 * Pass or fail, the blocking problems in plain English, any warnings, what the
 * package depends on, and the raw text behind a technical-details toggle.
 */
export function ValidationResultPanel({ summary }: { summary: ValidationSummary }) {
  const tone = TONE[summary.state];
  const { Icon } = tone;
  return (
    <section
      data-validation-state={summary.state}
      className={clsx("rounded-2xl border px-5 py-4", tone.box)}
    >
      <div className="flex items-start gap-3">
        <Icon className="mt-0.5 h-5 w-5 shrink-0" aria-hidden />
        <div className="min-w-0 flex-1">
          <p className="text-base font-semibold">{summary.headline}</p>
          <p className="mt-1 text-sm">{summary.detail}</p>
          {summary.checked_at && (
            <p className="mt-1 text-xs opacity-70">{formatDate(summary.checked_at)}</p>
          )}

          {summary.problems.length > 0 && (
            <div className="mt-4">
              <h3 className="text-xs font-semibold uppercase tracking-wide opacity-70">
                {copy.admin.validation.problemsHeading}
              </h3>
              <ul className="mt-2 space-y-2">
                {summary.problems.map((problem) => (
                  <li key={problem.technical} className="rounded-xl bg-white/70 px-3 py-2 text-sm">
                    {problem.summary}
                  </li>
                ))}
              </ul>
              <TechnicalDetails>
                <ul className="space-y-1">
                  {summary.problems.map((problem) => (
                    <li key={problem.technical}>{problem.technical}</li>
                  ))}
                </ul>
              </TechnicalDetails>
            </div>
          )}

          {summary.warnings.length > 0 && (
            <div className="mt-4">
              <h3 className="text-xs font-semibold uppercase tracking-wide opacity-70">
                {copy.admin.validation.warningsHeading}
              </h3>
              <ul className="mt-2 space-y-2">
                {summary.warnings.map((warning) => (
                  <li
                    key={warning}
                    className="flex items-start gap-2 rounded-xl bg-white/70 px-3 py-2 text-sm"
                  >
                    <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" aria-hidden />
                    {warning}
                  </li>
                ))}
              </ul>
            </div>
          )}

          {summary.dependencies.length > 0 && (
            <div className="mt-4">
              <h3 className="text-xs font-semibold uppercase tracking-wide opacity-70">
                {copy.admin.validation.dependenciesHeading}
              </h3>
              <ul className="mt-2 space-y-1 text-sm">
                {summary.dependencies.map((dependency) => (
                  <li key={dependency.name}>
                    <span className="font-medium">{dependency.name}</span> — {dependency.relationship}
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      </div>
    </section>
  );
}
