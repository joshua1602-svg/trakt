import { useEffect, useMemo, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { FileText, Info, X } from "lucide-react";
import clsx from "clsx";
import { useOpsClient } from "@/api/context";
import type { Batch, BatchDataset, PortfolioOption } from "@/api/types";
import { Page } from "@/components/Page";
import { StatusChip } from "@/components/StatusChip";
import { useToast } from "@/components/Toast";
import { copy } from "@/lib/copy";
import { errorMessage } from "@/lib/useLoad";

const NEW_CLIENT = "__new__";
const NEW_PORTFOLIO = "__new_portfolio__";

/**
 * One numbered step of the guided initiation flow.
 *
 * The order follows the real dependency model, not the shape of the form: the
 * operator supplies the governing facts (who, which book), and Trakt derives
 * what it will prepare from what that book is registered to report. Every step
 * stays on screen and stays readable once it has been answered.
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

function Fact({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div>
      <dt className="text-xs text-stone-500">{label}</dt>
      <dd className="mt-0.5 text-sm text-stone-900">{children}</dd>
    </div>
  );
}

export function NewWorkflowScreen() {
  const client = useOpsClient();
  const navigate = useNavigate();
  const toast = useToast();

  // Step 1 — who is it for
  const [clients, setClients] = useState<string[]>([]);
  const [clientChoice, setClientChoice] = useState("");
  const [newClientName, setNewClientName] = useState("");
  const [portfolios, setPortfolios] = useState<PortfolioOption[]>([]);
  const [portfolioChoice, setPortfolioChoice] = useState("");
  const [newPortfolioId, setNewPortfolioId] = useState("");

  // Step 2 — which book
  const [dataset, setDataset] = useState<BatchDataset>("funded");

  // Step 4 — reporting period
  const [period, setPeriod] = useState("");
  const [creating, setCreating] = useState(false);

  // Steps 5 and 6 — upload, then confirm. The browser sends file CONTENT only;
  // the destination is derived by the server, so there is nowhere here to type
  // a storage location.
  const [batch, setBatch] = useState<Batch | null>(null);
  const [chosen, setChosen] = useState<File[]>([]);
  const [uploading, setUploading] = useState(false);
  const fileInput = useRef<HTMLInputElement>(null);

  const clientId = clientChoice === NEW_CLIENT ? newClientName.trim() : clientChoice;
  const isNewPortfolio = portfolioChoice === NEW_PORTFOLIO;
  const portfolio = useMemo(
    () => portfolios.find((p) => p.portfolio_id === portfolioChoice) ?? null,
    [portfolios, portfolioChoice],
  );
  const portfolioId = isNewPortfolio ? newPortfolioId.trim() : portfolioChoice;

  // A book Trakt has never seen delivers a funded pack and reports management
  // information — the same default the automated route applies.
  const availableDatasets = isNewPortfolio
    ? ["funded"]
    : (portfolio?.datasets ?? ["funded", "pipeline"]);
  // What Trakt will prepare is DERIVED, never chosen: regime reporting belongs
  // to the funded book, and only when the book is registered for it.
  const outcome =
    dataset === "funded" && portfolio?.regime_required ? "mi_annex2" : "mi";
  const outcomeLabel =
    outcome === "mi_annex2" ? copy.newWorkflow.outcomeAnnex : copy.newWorkflow.outcomeMi;

  useEffect(() => {
    client
      .getClients()
      .then(setClients)
      .catch(() => setClients([]));
  }, [client]);

  // Portfolios are always fetched for the CHOSEN client, so a portfolio
  // belonging to somebody else is never even in the list. The API scopes the
  // same way — this is convenience, not the control.
  useEffect(() => {
    setPortfolioChoice("");
    setNewPortfolioId("");
    if (!clientId || clientChoice === NEW_CLIENT) {
      setPortfolios([]);
      return;
    }
    let live = true;
    client
      .getPortfolios(clientId)
      .then((list) => live && setPortfolios(list))
      .catch(() => live && setPortfolios([]));
    return () => {
      live = false;
    };
  }, [client, clientId, clientChoice]);

  // Never leave a book selected that this portfolio does not deliver.
  useEffect(() => {
    if (!availableDatasets.includes(dataset)) {
      setDataset(availableDatasets[0] as BatchDataset);
    }
  }, [availableDatasets, dataset]);

  async function handleCreate() {
    if (!clientId || !portfolioId || !period) return;
    setCreating(true);
    try {
      const created = await client.createBatch({
        client_id: clientId,
        portfolio_id: portfolioId,
        reporting_date: period,
        workflow_type: outcome,
        dataset,
        // Sending the files IS the operator's confirmation, so the pack starts
        // as soon as the existing intake path judges it complete.
        auto_start_when_ready: true,
        new_portfolio: isNewPortfolio,
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

  const locked = Boolean(batch);

  return (
    <Page title={copy.newWorkflow.title}>
      <div className="space-y-6">
        <p className="flex items-start gap-2 rounded-xl border border-stone-200 bg-stone-50 px-4 py-3 text-sm leading-relaxed text-stone-600">
          <Info className="mt-0.5 h-4 w-4 shrink-0" aria-hidden />
          {copy.newWorkflow.intro}
        </p>

        {/* 1 — Who is it for? Client first, then that client's portfolios. */}
        <SectionCard step={1} heading={copy.newWorkflow.detailsHeading}>
          <div className="grid gap-4 sm:grid-cols-2">
            <div>
              <label className="mb-1 block text-sm font-medium text-stone-700" htmlFor="client">
                {copy.newWorkflow.clientLabel}
              </label>
              <select
                id="client"
                value={clientChoice}
                disabled={locked}
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
                  disabled={locked}
                  onChange={(event) => setNewClientName(event.target.value)}
                  className="w-full rounded-xl border border-stone-300 px-3 py-2.5 text-sm outline-none focus:border-blue-500 focus:ring-2 focus:ring-blue-100 disabled:opacity-70"
                />
              </div>
            )}
            <div>
              <label className="mb-1 block text-sm font-medium text-stone-700" htmlFor="portfolio">
                {copy.newWorkflow.portfolioLabel}
              </label>
              <select
                id="portfolio"
                value={portfolioChoice}
                disabled={locked || !clientId}
                onChange={(event) => setPortfolioChoice(event.target.value)}
                className="w-full rounded-xl border border-stone-300 bg-white px-3 py-2.5 text-sm outline-none focus:border-blue-500 focus:ring-2 focus:ring-blue-100 disabled:opacity-70"
              >
                <option value="">{copy.newWorkflow.portfolioPlaceholder}</option>
                {portfolios.map((option) => (
                  <option key={option.portfolio_id} value={option.portfolio_id}>
                    {option.display_name} · {option.portfolio_id}
                  </option>
                ))}
                <option value={NEW_PORTFOLIO}>{copy.newWorkflow.newPortfolioOption}</option>
              </select>
              {clientId && portfolios.length === 0 && !isNewPortfolio && (
                <p className="mt-1 text-xs text-stone-500">{copy.newWorkflow.noPortfolios}</p>
              )}
            </div>
            {isNewPortfolio && (
              <div>
                <label
                  className="mb-1 block text-sm font-medium text-stone-700"
                  htmlFor="new-portfolio"
                >
                  {copy.newWorkflow.newPortfolioLabel}
                </label>
                <input
                  id="new-portfolio"
                  value={newPortfolioId}
                  disabled={locked}
                  onChange={(event) => setNewPortfolioId(event.target.value)}
                  className="w-full rounded-xl border border-stone-300 px-3 py-2.5 text-sm outline-none focus:border-blue-500 focus:ring-2 focus:ring-blue-100 disabled:opacity-70"
                />
                <p className="mt-1 text-xs leading-relaxed text-stone-500">
                  {copy.newWorkflow.newPortfolioHelp}
                </p>
              </div>
            )}
          </div>

          {/* Read-only context for the chosen book. The identifier stays
              visible, but it is not the headline. */}
          {portfolio && (
            <dl
              data-portfolio-context={portfolio.portfolio_id}
              className="mt-5 grid grid-cols-2 gap-4 border-t border-stone-100 pt-4 sm:grid-cols-4"
            >
              <Fact label={copy.newWorkflow.portfolioLabel}>
                <span className="block">{portfolio.display_name}</span>
                <span className="block font-mono text-xs text-stone-500">
                  {portfolio.portfolio_id}
                </span>
              </Fact>
              <Fact label={copy.newWorkflow.assetClass}>{portfolio.asset_label || "—"}</Fact>
              <Fact label={copy.newWorkflow.availableBooks}>
                {portfolio.dataset_labels.join(", ") || "—"}
              </Fact>
              <Fact label={copy.newWorkflow.reportingRequirement}>
                {portfolio.reporting_requirement}
              </Fact>
            </dl>
          )}
        </SectionCard>

        {/* 2 — Which book? Two separate governed deliveries, not one form. */}
        <SectionCard step={2} heading={copy.newWorkflow.bookHeading}>
          <p className="mb-4 text-sm leading-relaxed text-stone-500">
            {copy.newWorkflow.bookNote}
          </p>
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
            ).map((option) => {
              const available = availableDatasets.includes(option.value);
              return (
                <label
                  key={option.value}
                  data-dataset={option.value}
                  className={clsx(
                    "flex flex-col gap-1 rounded-2xl border-2 p-5 transition-colors",
                    locked || !available ? "cursor-default" : "cursor-pointer",
                    !available && "opacity-50",
                    locked && "opacity-70",
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
                    disabled={locked || !available}
                    onChange={() => setDataset(option.value)}
                    className="sr-only"
                  />
                  <span className="text-base font-semibold text-stone-900">{option.label}</span>
                  <span className="text-sm text-stone-500">
                    {available ? option.help : copy.newWorkflow.bookUnavailable}
                  </span>
                </label>
              );
            })}
          </div>
        </SectionCard>

        {/* 3 — What Trakt will prepare. Derived, read-only. */}
        <SectionCard step={3} heading={copy.newWorkflow.outcomeHeading}>
          <p
            data-derived-outcome={outcome}
            className="rounded-2xl border border-stone-200 bg-stone-50 px-4 py-3 text-base font-semibold text-stone-900"
          >
            {outcomeLabel}
          </p>
          <p className="mt-2 text-sm leading-relaxed text-stone-500">
            {copy.newWorkflow.outcomeDerived}
          </p>
        </SectionCard>

        {/* 4 — Which reporting period? */}
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
                disabled={locked}
                onChange={(event) => setPeriod(event.target.value)}
                className="w-full rounded-xl border border-stone-300 px-3 py-2.5 text-sm outline-none focus:border-blue-500 focus:ring-2 focus:ring-blue-100 disabled:opacity-70"
              />
            </div>
          </div>

          {!batch && (
            <button
              type="button"
              disabled={creating || !clientId || !portfolioId || !period}
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
                <Fact label={copy.newWorkflow.clientLabel}>{batch.client_id}</Fact>
                <Fact label={copy.newWorkflow.portfolioLabel}>
                  <span className="block">{portfolio?.display_name ?? batch.portfolio_id}</span>
                  <span className="block font-mono text-xs text-stone-500">
                    {batch.portfolio_id}
                  </span>
                </Fact>
                <Fact label={copy.newWorkflow.assetClass}>{portfolio?.asset_label || "—"}</Fact>
                <Fact label={copy.newWorkflow.bookHeading}>{batch.dataset_label}</Fact>
                <Fact label={copy.newWorkflow.periodLabel}>{batch.reporting_date}</Fact>
                <Fact label={copy.newWorkflow.outcomeHeading}>{outcomeLabel}</Fact>
                <Fact label={copy.newWorkflow.chosenFiles}>{chosen.length}</Fact>
              </dl>

              {batch.source_prefix && (
                <div className="mt-5 border-t border-stone-100 pt-4">
                  <p className="text-xs text-stone-500">
                    {copy.newWorkflow.destinationHeading}
                  </p>
                  <p className="mt-0.5 break-all font-mono text-xs text-stone-700">
                    {batch.source_prefix}
                  </p>
                  <p className="mt-1 text-xs text-stone-500">
                    {copy.newWorkflow.destinationHelp}
                  </p>
                </div>
              )}

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
