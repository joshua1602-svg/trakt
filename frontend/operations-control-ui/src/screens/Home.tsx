import { Link } from "react-router-dom";
import { ChevronRight } from "lucide-react";
import { useOpsClient } from "@/api/context";
import type { Publication, WorkflowRow } from "@/api/types";
import { ErrorNote, Loading } from "@/components/ErrorNote";
import { Page } from "@/components/Page";
import { StatusChip } from "@/components/StatusChip";
import { copy, decisionsLabel } from "@/lib/copy";
import { formatDate, formatPeriod } from "@/lib/format";
import { useLoad } from "@/lib/useLoad";

const TILE_ORDER = [
  "new_deliveries",
  "needs_attention",
  "blocked",
  "ready_to_publish",
  "recently_published",
] as const;

function AttentionRow({ row }: { row: WorkflowRow }) {
  return (
    <Link
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
  );
}

function PublishedRow({ publication }: { publication: Publication }) {
  return (
    <Link
      to={`/workflows/${publication.workflow_id}`}
      className="flex items-center justify-between gap-4 rounded-xl border border-stone-200 bg-white px-4 py-3 transition-colors hover:border-stone-300 hover:bg-stone-50"
    >
      <div className="min-w-0">
        <p className="truncate text-sm font-medium text-stone-900">{publication.client_id}</p>
        <p className="mt-0.5 text-xs text-stone-500">
          {publication.workflow_type_label} · {formatPeriod(publication.reporting_period)} ·{" "}
          {copy.common.version} {publication.version}
        </p>
      </div>
      <div className="flex shrink-0 items-center gap-2 text-xs text-stone-400">
        {formatDate(publication.published_at)}
        <StatusChip status={publication.status} />
      </div>
    </Link>
  );
}

export function HomeScreen() {
  const client = useOpsClient();
  const { data, error, loading, reload } = useLoad(() => client.getDashboard(), []);

  return (
    <Page title={copy.home.title} subtitle={copy.home.subtitle}>
      {loading && <Loading />}
      {error && !loading && <ErrorNote message={error} onRetry={() => void reload()} />}
      {data && !loading && (
        <div className="space-y-10">
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-5">
            {TILE_ORDER.map((key) => (
              <div key={key} className="rounded-2xl border border-stone-200 bg-white px-4 py-5">
                <p className="text-2xl font-semibold text-stone-900">{data.tiles[key]}</p>
                <p className="mt-1 text-xs text-stone-500">{copy.home.tiles[key]}</p>
              </div>
            ))}
          </div>

          <section>
            <h2 className="mb-3 text-sm font-semibold text-stone-700">{copy.home.needsAttention}</h2>
            {data.needs_attention.length === 0 ? (
              <p className="rounded-xl border border-dashed border-stone-200 px-4 py-8 text-center text-sm text-stone-400">
                {copy.home.emptyAttention}
              </p>
            ) : (
              <div className="space-y-2">
                {data.needs_attention.map((row) => (
                  <AttentionRow key={row.workflow_id} row={row} />
                ))}
              </div>
            )}
          </section>

          <section>
            <h2 className="mb-3 text-sm font-semibold text-stone-700">
              {copy.home.recentlyPublished}
            </h2>
            {data.recently_published.length === 0 ? (
              <p className="rounded-xl border border-dashed border-stone-200 px-4 py-8 text-center text-sm text-stone-400">
                {copy.home.emptyPublished}
              </p>
            ) : (
              <div className="space-y-2">
                {data.recently_published.map((publication) => (
                  <PublishedRow key={publication.publication_id} publication={publication} />
                ))}
              </div>
            )}
          </section>
        </div>
      )}
    </Page>
  );
}
