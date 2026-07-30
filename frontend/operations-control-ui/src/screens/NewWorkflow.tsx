import { useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { FileText, Info, X } from "lucide-react";
import clsx from "clsx";
import { useOpsClient } from "@/api/context";
import type { Batch, BatchDataset, WorkflowOutcome } from "@/api/types";
import { Page } from "@/components/Page";
import { StatusChip } from "@/components/StatusChip";
import { useToast } from "@/components/Toast";
import { copy } from "@/lib/copy";
import { errorMessage } from "@/lib/useLoad";

const NEW_CLIENT = "__new__";

/**
 * One numbered step of the guided initiation flow. The numbers are the
 * operator's map of the task, not a wizard: every step stays on screen and
 * stays readable once it has been answered.
 */
function SectionCard({
  step,
  heading,
  children,
}: {
  step: number;
  heading: string;
  children: React.ReactNode;
}) {
  return (
    <section className="rounded-2xl border border-stone-200 bg-white p-6">
      <div className="mb-4 flex items-baseline gap-3">
        <span className="shrink-0 text-xs font-semibold uppercase tracking-wide text-stone-400">
          {copy.newWorkflow.stepLabel} {step}
        </span>
        <h2 className="text-lg font-semibold text-stone-900">{heading}</h2>
      </div>
      {children}
    </section>
  );
}

export function NewWorkflowScreen() {
  const client = useOpsClient();
  const navigate = useNavigate();
  const toast = useToast();

  // Step 1 — what should Trakt prepare?
  const [outcome, setOutcome] = useState<WorkflowOutcome>("mi");

  // Step 2 — which book. Regime reporting is prepared from the funded book
  // only, so choosing Pipeline forces MI and disables the annex option. The
  // backend refuses the combination as well — this is convenience, not the
  // control.
  const [dataset, setDataset] = useState<BatchDataset>("funded");
  const regimeAvailable = dataset === "funded";

  // Steps 3 and 4 — who is it for, and which reporting period
  const [clients, setClients] = useState<string[]>([]);
  const [clientChoice, setClientChoice] = useState("");
  const [newClientName, setNewClientName] = useState("");
  const [portfolio, setPortfolio] = useState("");
  const [period, setPeriod] = useState("");
  const [creating, setCreating] = useState(false);

  // Steps 5 and 6 — upload the files, then confirm. The browser sends file
  // CONTENT only; the destination is derived by the server from the fields
  // above, so there is nowhere here to type a storage location.
  const [batch, setBatch] = useState<Batch | null>(null);
  const [chosen, setChosen] = useState<File[]>([]);
  const [uploading, setUploading] = useState(false);
  const fileInput = useRef<HTMLInputElement>(null);

  const clientId = clientChoice === NEW_CLIENT ? newClientName.trim() : clientChoice;

  function chooseDataset(next: BatchDataset) {
    setDataset(next);
    if (next !== "funded") setOutcome("mi");
  }

  useEffect(() => {
    client
      .getClients()
      .then(setClients)
      .catch(() => setClients([]));
  }, [client]);

  async function handleCreate() {
    if (!clientId || !portfolio.trim() || !period) return;
    setCreating(true);
    try {
      const created = await client.createBatch({
        client_id: clientId,
        portfolio_id: portfolio.trim(),
        reporting_date: period,
        workflow_type: outcome,
        dataset,
        // Sending the files IS the operator's confirmation, so the pack starts
        // as soon as the existing intake path judges it complete.
        auto_start_when_ready: true,
      });
      setBatch(created);
    } catch (err) {
      toast.show(errorMessage(err), "error");
    } finally {
      setCreating(false);
    }
  }

  function addFiles(list: FileList | null) {
    if (!list) return;
    const incoming = Array.from(list);
    setChosen((current) => [
      ...current,
      ...incoming.filter((f) => !current.some((c) => c.name === f.name && c.size === f.size)),
    ]);
    if (fileInput.current) fileInput.current.value = "";
  }

  async function handleUpload() {
    if (!batch || chosen.length === 0) return;
    setUploading(true);
    try {
      const updated = await client.uploadBatchFiles(batch.batch_id, chosen, batch.client_id);
      setBatch(updated);
      setChosen([]);
      // The intake path decides whether the pack is complete. When it is, a
      // workflow already exists — go straight to it.
      if (updated.workflow_id) {
        navigate(`/workflows/${updated.workflow_id}`);
      } else {
        navigate(`/batches/${updated.batch_id}?client=${encodeURIComponent(updated.client_id)}`);
      }
    } catch (err) {
      toast.show(errorMessage(err), "error");
      setUploading(false);
    }
  }

  return (
    <Page title={copy.newWorkflow.title}>
      <div className="space-y-6">
        <p className="flex items-start gap-2 rounded-xl border border-stone-200 bg-stone-50 px-4 py-3 text-sm leading-relaxed text-stone-600">
          <Info className="mt-0.5 h-4 w-4 shrink-0" aria-hidden />
          {copy.newWorkflow.intro}
        </p>

        {/* 1 — What should Trakt prepare? */}
        <SectionCard step={1} heading={copy.newWorkflow.outcomeHeading}>
          <div className="grid gap-3 sm:grid-cols-2">
            {(
              [
                { value: "mi", label: copy.newWorkflow.outcomeMi, help: copy.newWorkflow.outcomeMiHelp },
                {
                  value: "mi_annex2",
                  label: copy.newWorkflow.outcomeAnnex,
                  help: copy.newWorkflow.outcomeAnnexHelp,
                },
              ] as const
            ).map((option) => (
              <label
                key={option.value}
                className={clsx(
                  "flex flex-col gap-1 rounded-2xl border-2 p-5 transition-colors",
                  batch ? "cursor-default opacity-70" : "cursor-pointer",
                  outcome === option.value
                    ? "border-blue-600 bg-blue-50/50"
                    : "border-stone-200 bg-white hover:border-stone-300",
                  option.value === "mi_annex2" && !regimeAvailable && "opacity-50",
                )}
              >
                <input
                  type="radio"
                  name="outcome"
                  value={option.value}
                  checked={outcome === option.value}
                  disabled={Boolean(batch) || (option.value === "mi_annex2" && !regimeAvailable)}
                  onChange={() => setOutcome(option.value)}
                  className="sr-only"
                />
                <span className="text-base font-semibold text-stone-900">{option.label}</span>
                <span className="text-sm text-stone-500">{option.help}</span>
              </label>
            ))}
          </div>
          {!regimeAvailable && (
            <p className="mt-3 text-sm text-stone-500">{copy.newWorkflow.bookPipelineLocksMi}</p>
          )}
        </SectionCard>

        {/* 2 — Which book? */}
        <SectionCard step={2} heading={copy.newWorkflow.bookHeading}>
          <div className="grid gap-3 sm:grid-cols-2">
            {(
              [
                { value: "funded", label: copy.newWorkflow.bookFunded, help: copy.newWorkflow.bookFundedHelp },
                {
                  value: "pipeline",
                  label: copy.newWorkflow.bookPipeline,
                  help: copy.newWorkflow.bookPipelineHelp,
                },
              ] as const
            ).map((option) => (
              <label
                key={option.value}
                data-dataset={option.value}
                className={clsx(
                  "flex flex-col gap-1 rounded-2xl border-2 p-5 transition-colors",
                  batch ? "cursor-default opacity-70" : "cursor-pointer",
                  dataset === option.value
                    ? "border-blue-600 bg-blue-50/50"
                    : "border-stone-200 bg-white hover:border-stone-300",
                )}
              >
                <input
                  type="radio"
                  name="dataset"
                  value={option.value}
                  checked={dataset === option.value}
                  disabled={Boolean(batch)}
                  onChange={() => chooseDataset(option.value)}
                  className="sr-only"
                />
                <span className="text-base font-semibold text-stone-900">{option.label}</span>
                <span className="text-sm text-stone-500">{option.help}</span>
              </label>
            ))}
          </div>
        </SectionCard>

        {/* 3 and 4 — who is it for, and which reporting period */}
        <SectionCard step={3} heading={copy.newWorkflow.detailsHeading}>
          <div className="grid gap-4 sm:grid-cols-2">
            <div>
              <label className="mb-1 block text-sm font-medium text-stone-700" htmlFor="client">
                {copy.newWorkflow.clientLabel}
              </label>
              <select
                id="client"
                value={clientChoice}
                disabled={Boolean(batch)}
                onChange={(event) => setClientChoice(event.target.value)}
                className="w-full rounded-xl border border-stone-300 bg-white px-3 py-2.5 text-sm outline-none focus:border-blue-500 focus:ring-2 focus:ring-blue-100 disabled:opacity-70"
              >
                <option value="">{copy.newWorkflow.clientPlaceholder}</option>
                {clients.map((name) => (
                  <option key={name} value={name}>
                    {name}
                  </option>
                ))}
                <option value={NEW_CLIENT}>{copy.newWorkflow.newClientOption}</option>
              </select>
            </div>
            {clientChoice === NEW_CLIENT && (
              <div>
                <label className="mb-1 block text-sm font-medium text-stone-700" htmlFor="new-client">
                  {copy.newWorkflow.newClientLabel}
                </label>
                <input
                  id="new-client"
                  value={newClientName}
                  disabled={Boolean(batch)}
                  onChange={(event) => setNewClientName(event.target.value)}
                  className="w-full rounded-xl border border-stone-300 px-3 py-2.5 text-sm outline-none focus:border-blue-500 focus:ring-2 focus:ring-blue-100 disabled:opacity-70"
                />
              </div>
            )}
            <div>
              <label className="mb-1 block text-sm font-medium text-stone-700" htmlFor="portfolio">
                {copy.newWorkflow.portfolioLabel}
              </label>
              <input
                id="portfolio"
                value={portfolio}
                disabled={Boolean(batch)}
                onChange={(event) => setPortfolio(event.target.value)}
                className="w-full rounded-xl border border-stone-300 px-3 py-2.5 text-sm outline-none focus:border-blue-500 focus:ring-2 focus:ring-blue-100 disabled:opacity-70"
              />
            </div>
          </div>
        </SectionCard>

        <SectionCard step={4} heading={copy.newWorkflow.periodHeading}>
          <div className="grid gap-4 sm:grid-cols-2">
            <div>
              <label className="mb-1 block text-sm font-medium text-stone-700" htmlFor="period">
                {copy.newWorkflow.periodLabel}
              </label>
              <input
                id="period"
                type="month"
                value={period}
                disabled={Boolean(batch)}
                onChange={(event) => setPeriod(event.target.value)}
                className="w-full rounded-xl border border-stone-300 px-3 py-2.5 text-sm outline-none focus:border-blue-500 focus:ring-2 focus:ring-blue-100 disabled:opacity-70"
              />
            </div>
          </div>

          {!batch && (
            <button
              type="button"
              disabled={creating || !clientId || !portfolio.trim() || !period}
              onClick={() => void handleCreate()}
              className="mt-6 rounded-xl bg-blue-600 px-6 py-2.5 text-sm font-semibold text-white transition-colors hover:bg-blue-700 disabled:cursor-not-allowed disabled:opacity-40"
            >
              {copy.newWorkflow.createButton}
            </button>
          )}
        </SectionCard>

        {/* 5 and 6 — upload, then confirm and submit */}
        {batch && (
          <>
            <SectionCard step={5} heading={copy.newWorkflow.filesHeading}>
              <div className="mb-5 flex flex-wrap items-center gap-3">
                <StatusChip status={batch.status} label={batch.status_label} />
                <p className="text-sm leading-relaxed text-stone-600">{batch.status_sentence}</p>
              </div>

              <label className="mb-1 block text-sm font-medium text-stone-700" htmlFor="files">
                {copy.newWorkflow.uploadLabel}
              </label>
              <p className="mb-2 text-xs leading-relaxed text-stone-500">
                {copy.newWorkflow.uploadHelper}
              </p>
              <input
                id="files"
                ref={fileInput}
                type="file"
                multiple
                accept=".csv,.xlsx,.xls,.xlsm"
                onChange={(event) => addFiles(event.target.files)}
                className="block w-full rounded-xl border border-stone-300 px-3 py-2.5 text-sm file:mr-3 file:rounded-lg file:border-0 file:bg-stone-100 file:px-3 file:py-1.5 file:text-sm file:font-medium file:text-stone-700 hover:file:bg-stone-200"
              />

              <h3 className="mb-2 mt-5 text-sm font-semibold text-stone-700">
                {copy.newWorkflow.chosenFiles}
              </h3>
              {chosen.length === 0 ? (
                <p className="text-sm text-stone-400">{copy.newWorkflow.noFilesChosen}</p>
              ) : (
                <ul className="space-y-2">
                  {chosen.map((file) => (
                    <li
                      key={`${file.name}-${file.size}`}
                      className="flex items-center justify-between gap-3 rounded-xl border border-stone-200 bg-white px-4 py-2.5"
                    >
                      <span className="flex min-w-0 items-center gap-2">
                        <FileText className="h-4 w-4 shrink-0 text-stone-400" aria-hidden />
                        <span className="truncate text-sm text-stone-800">{file.name}</span>
                      </span>
                      <button
                        type="button"
                        aria-label={`${copy.newWorkflow.removeFile} ${file.name}`}
                        onClick={() => setChosen((c) => c.filter((f) => f !== file))}
                        className="shrink-0 rounded-lg p-1 text-stone-400 hover:bg-stone-100 hover:text-stone-700"
                      >
                        <X className="h-4 w-4" aria-hidden />
                      </button>
                    </li>
                  ))}
                </ul>
              )}
            </SectionCard>

            <SectionCard step={6} heading={copy.newWorkflow.confirmHeading}>
              <dl className="grid grid-cols-2 gap-4 sm:grid-cols-4">
                <div>
                  <dt className="text-xs text-stone-500">{copy.newWorkflow.clientLabel}</dt>
                  <dd className="mt-0.5 text-sm text-stone-900">{batch.client_id}</dd>
                </div>
                <div>
                  <dt className="text-xs text-stone-500">{copy.newWorkflow.portfolioLabel}</dt>
                  <dd className="mt-0.5 text-sm text-stone-900">{batch.portfolio_id}</dd>
                </div>
                <div>
                  <dt className="text-xs text-stone-500">{copy.newWorkflow.periodLabel}</dt>
                  <dd className="mt-0.5 text-sm text-stone-900">{batch.reporting_date}</dd>
                </div>
                <div>
                  <dt className="text-xs text-stone-500">{copy.newWorkflow.bookHeading}</dt>
                  <dd className="mt-0.5 text-sm text-stone-900">{batch.dataset_label}</dd>
                </div>
              </dl>
              <button
                type="button"
                disabled={uploading || chosen.length === 0}
                onClick={() => void handleUpload()}
                className="mt-6 rounded-xl bg-blue-600 px-6 py-2.5 text-sm font-semibold text-white transition-colors hover:bg-blue-700 disabled:cursor-not-allowed disabled:opacity-40"
              >
                {uploading ? copy.newWorkflow.uploading : copy.newWorkflow.uploadButton}
              </button>
            </SectionCard>
          </>
        )}
      </div>
    </Page>
  );
}
