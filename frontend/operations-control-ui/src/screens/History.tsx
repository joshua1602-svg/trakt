import { useMemo } from "react";
import { useOpsClient } from "@/api/context";
import type { Publication } from "@/api/types";
import { ErrorNote, Loading } from "@/components/ErrorNote";
import { Page } from "@/components/Page";
import { StatusChip } from "@/components/StatusChip";
import { copy } from "@/lib/copy";
import { formatDate, formatPeriod } from "@/lib/format";
import { useLoad } from "@/lib/useLoad";

export function HistoryScreen() {
  const client = useOpsClient();
  const { data, error, loading, reload } = useLoad(() => client.getHistory(), []);

  const grouped = useMemo(() => {
    const map = new Map<string, Publication[]>();
    for (const publication of data ?? []) {
      const list = map.get(publication.client_id) ?? [];
      list.push(publication);
      map.set(publication.client_id, list);
    }
    return [...map.entries()].sort(([a], [b]) => a.localeCompare(b));
  }, [data]);

  return (
    <Page title={copy.history.title} subtitle={copy.history.subtitle}>
      {loading && <Loading />}
      {error && !loading && <ErrorNote message={error} onRetry={() => void reload()} />}
      {data && !loading && (
        <div className="space-y-10">
          {grouped.length === 0 && (
            <p className="rounded-xl border border-dashed border-stone-200 px-4 py-10 text-center text-sm text-stone-400">
              {copy.history.empty}
            </p>
          )}
          {grouped.map(([clientId, publications]) => (
            <section key={clientId}>
              <h2 className="mb-3 text-sm font-semibold text-stone-700">{clientId}</h2>
              <div className="space-y-2">
                {publications.map((publication) => (
                  <div
                    key={publication.publication_id}
                    className="rounded-xl border border-stone-200 bg-white px-4 py-3"
                  >
                    <div className="flex flex-wrap items-center justify-between gap-3">
                      <div className="min-w-0">
                        <p className="text-sm font-medium text-stone-900">
                          {formatPeriod(publication.reporting_period)} ·{" "}
                          {publication.workflow_type_label}
                        </p>
                        <p className="mt-0.5 text-xs text-stone-500">
                          {copy.common.version} {publication.version} ·{" "}
                          {copy.history.builtWithPrefix} {publication.rule_version_count}{" "}
                          {copy.history.builtWithSuffix}
                          {publication.approved_by && <> · {publication.approved_by}</>}
                        </p>
                      </div>
                      <div className="flex shrink-0 items-center gap-2 text-xs text-stone-400">
                        {formatDate(publication.published_at)}
                        <StatusChip status={publication.status} />
                      </div>
                    </div>
                    {publication.previous_publication_id && (
                      <p className="mt-2 text-xs text-stone-400">{copy.history.previousVersion}</p>
                    )}
                  </div>
                ))}
              </div>
            </section>
          ))}
        </div>
      )}
    </Page>
  );
}
