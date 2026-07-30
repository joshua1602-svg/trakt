import { Navigate, useParams } from "react-router-dom";
import { DecisionPanel } from "@/components/DecisionPanel";
import { ErrorNote, Loading } from "@/components/ErrorNote";
import { Page } from "@/components/Page";
import { useOpsClient } from "@/api/context";
import { copy } from "@/lib/copy";
import { isWorkflowId } from "@/lib/ids";
import { useLoad } from "@/lib/useLoad";

/**
 * A question about a delivery belongs inside that delivery's case file, so this
 * route forwards there — approval links are shared and bookmarked, and
 * forwarding keeps them working rather than breaking them.
 *
 * A question raised BEFORE a delivery exists (a file that still has to be
 * identified, held against the input pack) has no case file to open yet, so it
 * is still answered here.
 */
export function ReviewDetailScreen() {
  const { id = "" } = useParams();
  const client = useOpsClient();
  const { data: review, error, loading, reload } = useLoad(() => client.getReview(id), [id]);

  if (review && isWorkflowId(review.workflow_id)) {
    return (
      <Navigate
        replace
        to={`/workflows/${encodeURIComponent(review.workflow_id)}?decision=${encodeURIComponent(
          review.decision_id,
        )}`}
      />
    );
  }

  return (
    <Page title={review?.title ?? copy.reviews.title} subtitle={review?.client_id}>
      {loading && <Loading />}
      {error && !loading && <ErrorNote message={error} onRetry={() => void reload()} />}
      {!loading && !error && !review && (
        <p className="text-sm text-stone-500">{copy.reviews.notFound}</p>
      )}
      {review && (
        <div className="max-w-2xl">
          <DecisionPanel decisionId={review.decision_id} />
        </div>
      )}
    </Page>
  );
}
