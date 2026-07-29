import { useState } from "react";
import { Link } from "react-router-dom";
import { ChevronRight } from "lucide-react";
import clsx from "clsx";
import { useOpsClient } from "@/api/context";
import { ErrorNote, Loading } from "@/components/ErrorNote";
import { Page } from "@/components/Page";
import { StatusChip } from "@/components/StatusChip";
import { copy, decisionsLabel } from "@/lib/copy";
import { formatPeriod } from "@/lib/format";
import { useLoad } from "@/lib/useLoad";

const FILTERS: { label: string; status: string }[] = [
  { label: copy.workflows.filterAll, status: "" },
  { label: copy.workflows.filterNeedsReview, status: "needs_review" },
  { label: copy.workflows.filterBlocked, status: "blocked" },
  { label: copy.workflows.filterReady, status: "awaiting_publication" },
  { label: copy.workflows.filterPublished, status: "published" },
];

export function WorkflowsScreen() {
  const client = useOpsClient();
  const [status, setStatus] = useState("");
  const { data, error, loading, reload } = useLoad(
    () => client.getWorkflows(status ? { status } : undefined),
    [status],
  );

  return (
    <Page title={copy.workflows.title} subtitle={copy.workflows.subtitle}>
      <div className="mb-6 flex flex-wrap gap-2">
        {FILTERS.map((filter) => (
          <button
            key={filter.status}
            type="button"
            onClick={() => setStatus(filter.status)}
            className={clsx(
              "rounded-full border px-3.5 py-1.5 text-sm font-medium transition-colors",
              status === filter.status
                ? "border-stone-900 bg-stone-900 text-white"
                : "border-stone-200 bg-white text-stone-600 hover:border-stone-300",
            )}
          >
            {filter.label}
          </button>
        ))}
      </div>

      {loading && <Loading />}
      {error && !loading && <ErrorNote message={error} onRetry={() => void reload()} />}
      {data && !loading && (
        <div className="space-y-2">
          {data.length === 0 && (
            <p className="rounded-xl border border-dashed border-stone-200 px-4 py-10 text-center text-sm text-stone-400">
              {copy.workflows.empty}
            </p>
          )}
          {data.map((row) => (
            <Link
              key={row.workflow_id}
              to={`/workflows/${row.workflow_id}`}
              className="flex items-center justify-between gap-4 rounded-xl border border-stone-200 bg-white px-4 py-3 transition-colors hover:border-stone-300 hover:bg-stone-50"
            >
              <div className="min-w-0">
                <p className="truncate text-sm font-medium text-stone-900">
                  {row.client_id} · {row.portfolio_id}
                </p>
                <p className="mt-0.5 text-xs text-stone-500">
                  {row.workflow_type_label} · {formatPeriod(row.reporting_period)}
                </p>
              </div>
              <div className="flex shrink-0 items-center gap-2">
                {row.open_decisions > 0 && (
                  <span className="rounded-full bg-amber-100 px-2.5 py-0.5 text-xs font-medium text-amber-800">
                    {decisionsLabel(row.open_decisions)}
                  </span>
                )}
                <StatusChip status={row.status} />
                <ChevronRight className="h-4 w-4 text-stone-300" aria-hidden />
              </div>
            </Link>
          ))}
        </div>
      )}
    </Page>
  );
}
