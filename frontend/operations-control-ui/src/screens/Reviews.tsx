import { Link, useSearchParams } from "react-router-dom";
import { ChevronRight } from "lucide-react";
import { useOpsClient } from "@/api/context";
import { ErrorNote, Loading } from "@/components/ErrorNote";
import { Page } from "@/components/Page";
import { copy } from "@/lib/copy";
import { useLoad } from "@/lib/useLoad";

export function ReviewsScreen() {
  const client = useOpsClient();
  const [searchParams] = useSearchParams();
  const workflowId = searchParams.get("workflow") ?? undefined;

  const { data, error, loading, reload } = useLoad(
    () => client.getReviews(workflowId ? { workflow_id: workflowId } : undefined),
    [workflowId],
  );

  const open = (data ?? []).filter((r) => !r.resolved_at && r.status !== "resolved");

  return (
    <Page title={copy.reviews.title} subtitle={copy.reviews.subtitle}>
      {loading && <Loading />}
      {error && !loading && <ErrorNote message={error} onRetry={() => void reload()} />}
      {data && !loading && (
        <div className="space-y-2">
          {open.length === 0 && (
            <p className="rounded-xl border border-dashed border-stone-200 px-4 py-10 text-center text-sm text-stone-400">
              {copy.reviews.empty}
            </p>
          )}
          {open.map((review) => (
            <Link
              key={review.decision_id}
              to={`/reviews/${review.decision_id}`}
              className="flex items-center justify-between gap-4 rounded-xl border border-stone-200 bg-white px-4 py-4 transition-colors hover:border-stone-300 hover:bg-stone-50"
            >
              <div className="min-w-0">
                <p className="truncate text-sm font-semibold text-stone-900">{review.title}</p>
                <p className="mt-0.5 text-xs text-stone-500">{review.client_id}</p>
              </div>
              <div className="flex shrink-0 items-center gap-2">
                <span
                  className={
                    review.blocking
                      ? "rounded-full bg-amber-100 px-2.5 py-0.5 text-xs font-medium text-amber-800"
                      : "rounded-full bg-stone-100 px-2.5 py-0.5 text-xs font-medium text-stone-500"
                  }
                >
                  {review.blocking ? copy.reviews.blockingChip : copy.reviews.optionalChip}
                </span>
                <ChevronRight className="h-4 w-4 text-stone-300" aria-hidden />
              </div>
            </Link>
          ))}
        </div>
      )}
    </Page>
  );
}
