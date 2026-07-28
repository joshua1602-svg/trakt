import { useEffect, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { Lightbulb } from "lucide-react";
import clsx from "clsx";
import { useOpsClient } from "@/api/context";
import { ErrorNote, Loading } from "@/components/ErrorNote";
import { Page } from "@/components/Page";
import { useToast } from "@/components/Toast";
import { copy } from "@/lib/copy";
import { errorMessage, useLoad } from "@/lib/useLoad";

const OTHER = "__other__";

const SCOPE_LABELS: Record<string, string> = {
  file: copy.scopes.file,
  portfolio: copy.scopes.portfolio,
  client: copy.scopes.client,
  global: copy.scopes.global,
};

export function ReviewDetailScreen() {
  const { id = "" } = useParams();
  const client = useOpsClient();
  const navigate = useNavigate();
  const toast = useToast();

  const { data: review, error, loading, reload } = useLoad(() => client.getReview(id), [id]);

  const [choice, setChoice] = useState("");
  const [otherValue, setOtherValue] = useState("");
  const [scope, setScope] = useState("");
  const [rejectOpen, setRejectOpen] = useState(false);
  const [rejectReason, setRejectReason] = useState("");
  const [busy, setBusy] = useState(false);
  const [savedNote, setSavedNote] = useState("");

  useEffect(() => {
    if (review) {
      setChoice(review.recommendation?.value ?? "");
      setScope(review.default_scope);
    }
  }, [review]);

  const isMapping = review?.kind.includes("mapping") ?? false;
  const valueToSend = choice === OTHER ? otherValue.trim() : choice;
  const canConfirm = Boolean(valueToSend && scope);

  async function submit(action: "approve" | "reject" | "amend", value: string, reason: string) {
    if (!review) return;
    setBusy(true);
    try {
      const result = await client.submitDecision(review.decision_id, {
        action,
        value,
        scope,
        reason,
      });
      setSavedNote(result.rerun_scheduled ? copy.reviews.savedRerun : copy.reviews.saved);
      setTimeout(() => {
        navigate(`/workflows/${review.workflow_id}`);
      }, 1200);
    } catch (err) {
      toast.show(errorMessage(err), "error");
      setBusy(false);
    }
  }

  function handleConfirm() {
    if (!canConfirm) return;
    const action = choice === OTHER ? "amend" : "approve";
    void submit(action, valueToSend, "");
  }

  return (
    <Page title={review?.title ?? copy.reviews.title} subtitle={review?.client_id}>
      {loading && <Loading />}
      {error && !loading && <ErrorNote message={error} onRetry={() => void reload()} />}
      {review && !loading && (
        <div className="max-w-2xl space-y-6">
          <h2 className="text-xl font-semibold leading-relaxed text-stone-900">{review.question}</h2>

          {savedNote ? (
            <p
              role="status"
              className="rounded-xl border border-emerald-200 bg-emerald-50 px-4 py-3 text-sm text-emerald-800"
            >
              {savedNote}
            </p>
          ) : (
            <>
              {review.recommendation && (
                <div className="flex items-start gap-3 rounded-2xl border border-blue-100 bg-blue-50/70 p-4">
                  <Lightbulb className="mt-0.5 h-5 w-5 shrink-0 text-blue-600" aria-hidden />
                  <div>
                    <p className="text-xs font-semibold uppercase tracking-wide text-blue-700">
                      {copy.reviews.recommendedHeading}
                    </p>
                    <p className="mt-1 text-sm text-stone-800">{review.recommendation.label}</p>
                  </div>
                </div>
              )}

              {/* Options */}
              <div className="space-y-2">
                {review.options.map((option) => (
                  <label
                    key={option.value}
                    className={clsx(
                      "flex cursor-pointer items-center gap-3 rounded-xl border p-3.5 text-sm transition-colors",
                      choice === option.value
                        ? "border-blue-600 bg-blue-50/50"
                        : "border-stone-200 bg-white hover:border-stone-300",
                    )}
                  >
                    <input
                      type="radio"
                      name="option"
                      value={option.value}
                      checked={choice === option.value}
                      onChange={() => setChoice(option.value)}
                    />
                    <span className="font-medium text-stone-800">{option.label}</span>
                  </label>
                ))}
                {isMapping && (
                  <label
                    className={clsx(
                      "flex cursor-pointer flex-col gap-2 rounded-xl border p-3.5 text-sm transition-colors",
                      choice === OTHER
                        ? "border-blue-600 bg-blue-50/50"
                        : "border-stone-200 bg-white hover:border-stone-300",
                    )}
                  >
                    <span className="flex items-center gap-3">
                      <input
                        type="radio"
                        name="option"
                        value={OTHER}
                        checked={choice === OTHER}
                        onChange={() => setChoice(OTHER)}
                      />
                      <span className="font-medium text-stone-800">{copy.reviews.somethingElse}</span>
                    </span>
                    {choice === OTHER && (
                      <input
                        value={otherValue}
                        onChange={(event) => setOtherValue(event.target.value)}
                        placeholder={copy.reviews.somethingElsePlaceholder}
                        className="ml-7 rounded-lg border border-stone-300 px-3 py-2 text-sm outline-none focus:border-blue-500 focus:ring-2 focus:ring-blue-100"
                      />
                    )}
                  </label>
                )}
              </div>

              {/* Scope */}
              <div>
                <h3 className="mb-2 text-sm font-semibold text-stone-700">
                  {copy.reviews.scopeHeading}
                </h3>
                <div className="space-y-2">
                  {review.allowed_scopes.map((scopeValue) => (
                    <label
                      key={scopeValue}
                      className={clsx(
                        "flex cursor-pointer items-start gap-3 rounded-xl border p-3.5 transition-colors",
                        scope === scopeValue
                          ? "border-blue-600 bg-blue-50/50"
                          : "border-stone-200 bg-white hover:border-stone-300",
                      )}
                    >
                      <input
                        type="radio"
                        name="scope"
                        value={scopeValue}
                        checked={scope === scopeValue}
                        onChange={() => setScope(scopeValue)}
                        className="mt-0.5"
                      />
                      <span>
                        <span className="block text-sm font-medium text-stone-800">
                          {SCOPE_LABELS[scopeValue] ?? scopeValue}
                        </span>
                        {review.scope_explanations[scopeValue] && (
                          <span className="mt-0.5 block text-xs text-stone-500">
                            {review.scope_explanations[scopeValue]}
                          </span>
                        )}
                      </span>
                    </label>
                  ))}
                </div>
              </div>

              {/* Actions */}
              <div className="flex flex-wrap items-center gap-3 pt-2">
                <button
                  type="button"
                  disabled={busy || !canConfirm}
                  onClick={handleConfirm}
                  className="rounded-xl bg-blue-600 px-8 py-2.5 text-sm font-semibold text-white transition-colors hover:bg-blue-700 disabled:cursor-not-allowed disabled:opacity-40"
                >
                  {copy.reviews.confirm}
                </button>
                <button
                  type="button"
                  disabled={busy}
                  onClick={() => setRejectOpen((open) => !open)}
                  className="rounded-xl border border-stone-300 bg-white px-4 py-2.5 text-sm font-medium text-stone-600 transition-colors hover:bg-stone-50 disabled:opacity-40"
                >
                  {copy.reviews.reject}
                </button>
              </div>

              {rejectOpen && (
                <div className="rounded-xl border border-stone-200 bg-white p-4">
                  <label
                    className="mb-2 block text-sm font-medium text-stone-700"
                    htmlFor="reject-reason"
                  >
                    {copy.reviews.rejectReason}
                  </label>
                  <textarea
                    id="reject-reason"
                    rows={3}
                    value={rejectReason}
                    onChange={(event) => setRejectReason(event.target.value)}
                    className="mb-3 w-full rounded-xl border border-stone-300 px-3 py-2 text-sm outline-none focus:border-blue-500 focus:ring-2 focus:ring-blue-100"
                  />
                  <button
                    type="button"
                    disabled={busy || !rejectReason.trim()}
                    onClick={() => void submit("reject", valueToSend || choice, rejectReason.trim())}
                    className="rounded-xl bg-stone-900 px-4 py-2 text-sm font-semibold text-white transition-colors hover:bg-stone-700 disabled:opacity-40"
                  >
                    {copy.common.save}
                  </button>
                </div>
              )}
            </>
          )}
        </div>
      )}
    </Page>
  );
}
