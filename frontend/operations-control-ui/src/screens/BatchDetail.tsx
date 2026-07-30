import { useEffect, useState } from "react";
import { Link, useParams, useSearchParams } from "react-router-dom";
import { ArrowRight, FileText } from "lucide-react";
import clsx from "clsx";
import { useOpsClient } from "@/api/context";
import type { Batch, BatchInputRole } from "@/api/types";
import { ErrorNote, Loading } from "@/components/ErrorNote";
import { Page } from "@/components/Page";
import { StatusChip } from "@/components/StatusChip";
import { useToast } from "@/components/Toast";
import { copy, decisionsLabel } from "@/lib/copy";
import { formatPeriod } from "@/lib/format";
import { errorMessage, useLoad } from "@/lib/useLoad";

const POLLED_STATUSES = ["receiving", "classifying", "running"];

const OUTCOME_LABELS = {
  mi: copy.newWorkflow.outcomeMi,
  mi_annex2: copy.newWorkflow.outcomeAnnex,
};

function RoleState({ role }: { role: BatchInputRole }) {
  if (role.satisfied) {
    return <span className="text-sm font-medium text-emerald-700">{copy.batch.roleReceived}</span>;
  }
  if (role.required) {
    return <span className="text-sm font-medium text-amber-700">{copy.batch.roleWaiting}</span>;
  }
  return <span className="text-sm text-stone-400">{copy.batch.roleOptional}</span>;
}

export function BatchDetailScreen() {
  const { id = "" } = useParams();
  const [searchParams] = useSearchParams();
  const clientParam = searchParams.get("client") ?? undefined;
  const client = useOpsClient();
  const toast = useToast();

  const { data: batch, error, loading, reload } = useLoad(
    () => client.getBatch(id, clientParam),
    [id, clientParam],
  );

  const [chosen, setChosen] = useState<File[]>([]);
  const [busy, setBusy] = useState(false);

  // Poll while the server is still receiving, classifying or running.
  const status = batch?.status;
  useEffect(() => {
    if (!status || !POLLED_STATUSES.includes(status)) return;
    const timer = setInterval(() => void reload({ quiet: true }), 2500);
    return () => clearInterval(timer);
  }, [status, reload]);

  async function run(action: () => Promise<unknown>) {
    setBusy(true);
    try {
      await action();
      await reload({ quiet: true });
    } catch (err) {
      toast.show(errorMessage(err), "error");
    } finally {
      setBusy(false);
    }
  }

  async function handleAddFiles(current: Batch) {
    if (chosen.length === 0) return;
    await run(() => client.uploadBatchFiles(current.batch_id, chosen, clientParam));
    setChosen([]);
  }

  const canStart = batch?.status === "ready" && !batch.auto_start_when_ready;
  const showWorkflowLink =
    batch != null &&
    (batch.status === "running" || batch.status === "completed") &&
    Boolean(batch.workflow_id);

  return (
    <Page
      title={batch ? `${batch.client_id} · ${batch.portfolio_id}` : copy.batch.inputPack}
      subtitle={
        batch
          ? `${OUTCOME_LABELS[batch.workflow_type]} · ${formatPeriod(batch.reporting_date)}`
          : undefined
      }
    >
      {loading && !batch && <Loading />}
      {error && !batch && <ErrorNote message={error} onRetry={() => void reload()} />}
      {!loading && !batch && !error && (
        <p className="text-sm text-stone-500">{copy.batch.notFound}</p>
      )}
      {batch && (
        <div className="space-y-6">
          {/* Overall status */}
          <div className="rounded-2xl border border-stone-200 bg-white p-6">
            <div className="flex flex-wrap items-center gap-3">
              <h2 className="text-xl font-semibold text-stone-900">{batch.status_label}</h2>
              <StatusChip status={batch.status} label={batch.status_label} />
            </div>
            <p className="mt-2 text-base leading-relaxed text-stone-700">{batch.status_sentence}</p>
            {batch.auto_start_when_ready &&
              !["running", "completed", "failed"].includes(batch.status) && (
                <p className="mt-2 text-sm text-stone-500">{copy.batch.autoStartNote}</p>
              )}
          </div>

          {/* Input pack */}
          <section className="rounded-2xl border border-stone-200 bg-white p-6">
            <h2 className="mb-4 text-lg font-semibold text-stone-900">{copy.batch.inputPack}</h2>

            <ul className="divide-y divide-stone-100">
              {batch.input_roles.map((role) => (
                <li key={role.role} className="flex items-center justify-between gap-4 py-3">
                  <span className="text-sm font-medium text-stone-800">{role.label}</span>
                  <RoleState role={role} />
                </li>
              ))}
            </ul>

            <h3 className="mb-3 mt-6 text-sm font-semibold text-stone-700">
              {copy.batch.filesHeading}
            </h3>
            {batch.files.length === 0 ? (
              <p className="text-sm text-stone-400">{copy.batch.noFiles}</p>
            ) : (
              <ul className="space-y-2">
                {batch.files.map((file) => (
                  <li
                    key={file.source_file_id}
                    className={clsx(
                      "flex items-start gap-3 rounded-xl border px-4 py-3",
                      file.status === "ambiguous"
                        ? "border-amber-200 bg-amber-50"
                        : "border-stone-200 bg-white",
                    )}
                  >
                    <FileText
                      className={clsx(
                        "mt-0.5 h-4 w-4 shrink-0",
                        file.status === "ambiguous" ? "text-amber-600" : "text-stone-400",
                      )}
                      aria-hidden
                    />
                    <span className="min-w-0">
                      <span
                        className={clsx(
                          "block truncate text-sm font-medium",
                          file.status === "ambiguous" ? "text-amber-900" : "text-stone-900",
                        )}
                      >
                        {file.filename}
                      </span>
                      <span
                        className={clsx(
                          "block text-sm leading-relaxed",
                          file.status === "ambiguous" ? "text-amber-800" : "text-stone-500",
                        )}
                      >
                        {file.status_sentence}
                      </span>
                    </span>
                  </li>
                ))}
              </ul>
            )}

            {/* Configuration */}
            <div className="mt-6 flex items-center justify-between gap-4 border-t border-stone-100 pt-4">
              <span className="text-sm font-medium text-stone-800">{copy.batch.configuration}</span>
              {batch.configuration_ready ? (
                <span className="text-sm font-medium text-emerald-700">{copy.batch.configReady}</span>
              ) : (
                <span className="text-sm font-medium text-amber-700">{copy.batch.configNeeded}</span>
              )}
            </div>

            {/* Blocking decisions */}
            <div className="mt-2 flex items-center justify-between gap-4 border-t border-stone-100 pt-4">
              <span className="text-sm font-medium text-stone-800">
                {copy.batch.blockingDecisions}
              </span>
              {batch.blocking_decisions.length === 0 ? (
                <span className="text-sm text-stone-400">{copy.batch.none}</span>
              ) : (
                <Link
                  to={`/reviews?workflow=${batch.batch_id}`}
                  className="text-sm font-medium text-blue-700 hover:underline"
                >
                  {decisionsLabel(batch.blocking_decisions.length)}
                </Link>
              )}
            </div>
          </section>

          {/* Actions */}
          <section className="rounded-2xl border border-stone-200 bg-white p-6">
            <h2 className="mb-1 text-sm font-semibold text-stone-700">
              {copy.batch.addFilesHeading}
            </h2>
            <p className="mb-2 text-xs leading-relaxed text-stone-500">
              {copy.newWorkflow.uploadHelper}
            </p>
            <div className="flex flex-col gap-2 sm:flex-row">
              <input
                aria-label={copy.newWorkflow.uploadLabel}
                type="file"
                multiple
                accept=".csv,.xlsx,.xls,.xlsm"
                onChange={(event) => setChosen(Array.from(event.target.files ?? []))}
                className="block w-full rounded-xl border border-stone-300 px-3 py-2.5 text-sm file:mr-3 file:rounded-lg file:border-0 file:bg-stone-100 file:px-3 file:py-1.5 file:text-sm file:font-medium file:text-stone-700 hover:file:bg-stone-200"
              />
              <button
                type="button"
                disabled={busy || chosen.length === 0}
                onClick={() => void handleAddFiles(batch)}
                className="shrink-0 rounded-xl border border-stone-300 bg-white px-4 py-2.5 text-sm font-medium text-stone-700 transition-colors hover:bg-stone-50 disabled:cursor-not-allowed disabled:opacity-40"
              >
                {copy.batch.addFilesButton}
              </button>
            </div>

            <div className="mt-5 flex flex-wrap items-center gap-3">
              {canStart && (
                <button
                  type="button"
                  disabled={busy}
                  onClick={() => void run(() => client.startBatch(batch.batch_id, clientParam))}
                  className="rounded-xl bg-blue-600 px-6 py-2.5 text-sm font-semibold text-white transition-colors hover:bg-blue-700 disabled:cursor-not-allowed disabled:opacity-40"
                >
                  {copy.batch.startButton}
                </button>
              )}
              {showWorkflowLink && (
                <Link
                  to={`/workflows/${batch.workflow_id}`}
                  className="flex items-center gap-2 rounded-xl bg-stone-900 px-4 py-2.5 text-sm font-semibold text-white transition-colors hover:bg-stone-700"
                >
                  {copy.batch.viewWorkflow}
                  <ArrowRight className="h-4 w-4" aria-hidden />
                </Link>
              )}
            </div>
          </section>
        </div>
      )}
    </Page>
  );
}
