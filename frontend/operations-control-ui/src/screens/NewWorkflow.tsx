import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import clsx from "clsx";
import { useOpsClient } from "@/api/context";
import type { Batch, BatchDataset, WorkflowOutcome } from "@/api/types";
import { Page } from "@/components/Page";
import { StatusChip } from "@/components/StatusChip";
import { useToast } from "@/components/Toast";
import { copy } from "@/lib/copy";
import { errorMessage } from "@/lib/useLoad";

const NEW_CLIENT = "__new__";

function SectionCard({ children }: { children: React.ReactNode }) {
  return <section className="rounded-2xl border border-stone-200 bg-white p-6">{children}</section>;
}

export function NewWorkflowScreen() {
  const client = useOpsClient();
  const navigate = useNavigate();
  const toast = useToast();

  // Step a — outcome
  const [outcome, setOutcome] = useState<WorkflowOutcome>("mi");

  // Step a2 — which book. Regime reporting is prepared from the funded book
  // only, so choosing Pipeline forces MI and disables the annex option. The
  // backend refuses the combination as well — this is convenience, not the
  // control.
  const [dataset, setDataset] = useState<BatchDataset>("funded");
  const regimeAvailable = dataset === "funded";

  // Step b — who is this for
  const [clients, setClients] = useState<string[]>([]);
  const [clientChoice, setClientChoice] = useState("");
  const [newClientName, setNewClientName] = useState("");
  const [portfolio, setPortfolio] = useState("");
  const [period, setPeriod] = useState("");
  const [creating, setCreating] = useState(false);

  // Step c — where are the files
  const [batch, setBatch] = useState<Batch | null>(null);
  const [folder, setFolder] = useState("");
  const [registering, setRegistering] = useState(false);

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
        auto_start_when_ready: false,
      });
      setBatch(created);
    } catch (err) {
      toast.show(errorMessage(err), "error");
    } finally {
      setCreating(false);
    }
  }

  async function handleAddFiles() {
    if (!batch || !folder.trim()) return;
    setRegistering(true);
    try {
      await client.registerBatchFile(batch.batch_id, folder.trim(), batch.client_id);
      navigate(`/batches/${batch.batch_id}?client=${encodeURIComponent(batch.client_id)}`);
    } catch (err) {
      toast.show(errorMessage(err), "error");
      setRegistering(false);
    }
  }

  return (
    <Page title={copy.newWorkflow.title}>
      <div className="space-y-6">
        {/* (a) Outcome */}
        <SectionCard>
          <h2 className="mb-4 text-lg font-semibold text-stone-900">
            {copy.newWorkflow.outcomeHeading}
          </h2>
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

        {/* (a2) Which book */}
        <SectionCard>
          <h2 className="mb-4 text-lg font-semibold text-stone-900">
            {copy.newWorkflow.bookHeading}
          </h2>
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

        {/* (b) Client, portfolio, period */}
        <SectionCard>
          <h2 className="mb-4 text-lg font-semibold text-stone-900">
            {copy.newWorkflow.detailsHeading}
          </h2>
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

        {/* (c) Where are the files? */}
        {batch && (
          <SectionCard>
            <h2 className="mb-2 text-lg font-semibold text-stone-900">
              {copy.newWorkflow.filesHeading}
            </h2>
            <div className="mb-5 flex flex-wrap items-center gap-3">
              <StatusChip status={batch.status} label={batch.status_label} />
              <p className="text-sm leading-relaxed text-stone-600">{batch.status_sentence}</p>
            </div>
            <label className="mb-1 block text-sm font-medium text-stone-700" htmlFor="folder">
              {copy.newWorkflow.folderLabel}
            </label>
            <p className="mb-2 text-xs text-stone-500">{copy.newWorkflow.folderHelper}</p>
            <div className="flex gap-2">
              <input
                id="folder"
                value={folder}
                onChange={(event) => setFolder(event.target.value)}
                className="w-full rounded-xl border border-stone-300 px-3 py-2.5 text-sm outline-none focus:border-blue-500 focus:ring-2 focus:ring-blue-100"
              />
              <button
                type="button"
                disabled={registering || !folder.trim()}
                onClick={() => void handleAddFiles()}
                className="shrink-0 rounded-xl bg-blue-600 px-4 py-2.5 text-sm font-semibold text-white transition-colors hover:bg-blue-700 disabled:cursor-not-allowed disabled:opacity-40"
              >
                {copy.newWorkflow.addFilesButton}
              </button>
            </div>
          </SectionCard>
        )}
      </div>
    </Page>
  );
}
