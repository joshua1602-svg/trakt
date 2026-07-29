import type { ImpactAnalysis } from "@/api/adminTypes";
import { StatusChip } from "@/components/StatusChip";
import { copy } from "@/lib/copy";
import { formatPeriod } from "@/lib/format";
import { EmptyNote, SectionHeading } from "./primitives";

/**
 * Who is affected by a version. Deliberately explicit that activation changes
 * only what future workflows resolve — it never rewrites prior results.
 */
export function ImpactPanel({ impact }: { impact: ImpactAnalysis }) {
  return (
    <section>
      <SectionHeading>{copy.admin.impact.heading}</SectionHeading>
      <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
        {[
          { label: copy.admin.impact.clients, value: impact.clients },
          { label: copy.admin.impact.portfolios, value: impact.portfolios },
          { label: copy.admin.impact.running, value: impact.running_pinned_workflows },
          {
            label: copy.admin.impact.future,
            value: impact.future_workflows_affected ? copy.admin.impact.yes : "—",
          },
        ].map((tile) => (
          <div key={tile.label} className="rounded-2xl border border-stone-200 bg-white px-4 py-4">
            <p className="text-2xl font-semibold text-stone-900">{tile.value}</p>
            <p className="mt-1 text-xs text-stone-500">{tile.label}</p>
          </div>
        ))}
      </div>

      <ul className="mt-4 space-y-1 text-sm text-stone-600">
        {impact.explanation.map((line) => (
          <li key={line}>{line}</li>
        ))}
      </ul>

      <h3 className="mb-2 mt-6 text-sm font-semibold text-stone-700">
        {copy.admin.impact.workflowsHeading}
      </h3>
      {impact.workflows.length === 0 ? (
        <EmptyNote>{copy.admin.impact.none}</EmptyNote>
      ) : (
        <div className="space-y-2">
          {impact.workflows.map((workflow) => (
            <div
              key={workflow.workflow_id}
              className="flex flex-wrap items-center justify-between gap-3 rounded-xl border border-stone-200 bg-white px-4 py-3"
            >
              <div className="min-w-0">
                <p className="truncate text-sm font-medium text-stone-900">
                  {workflow.client_id} · {workflow.portfolio_id}
                </p>
                <p className="mt-0.5 text-xs text-stone-500">
                  {formatPeriod(workflow.reporting_period)} · {copy.common.version}{" "}
                  {workflow.pinned_version.replace(/^v/, "")}
                </p>
              </div>
              <div className="flex shrink-0 items-center gap-2">
                {workflow.still_running && (
                  <span className="rounded-full bg-blue-50 px-2.5 py-0.5 text-xs font-medium text-blue-700">
                    {copy.admin.activate.pinnedNote}
                  </span>
                )}
                <StatusChip status={workflow.status} />
              </div>
            </div>
          ))}
        </div>
      )}
    </section>
  );
}
