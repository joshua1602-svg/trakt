import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { FileText, Sparkles } from "lucide-react";
import clsx from "clsx";
import { useOpsClient } from "@/api/context";
import type { Delivery, WorkflowOutcome, WorkflowType } from "@/api/types";
import { Page } from "@/components/Page";
import { useToast } from "@/components/Toast";
import { copy } from "@/lib/copy";
import { formatPeriod } from "@/lib/format";
import { errorMessage } from "@/lib/useLoad";

const NEW_CLIENT = "__new__";

const TYPE_OPTIONS: { value: WorkflowType; label: string }[] = [
  { value: "new_client", label: copy.newWorkflow.typeNewClient },
  { value: "new_portfolio", label: copy.newWorkflow.typeNewPortfolio },
  { value: "recurring", label: copy.newWorkflow.typeRecurring },
  { value: "backfill", label: copy.newWorkflow.typeBackfill },
];

function SectionCard({ children }: { children: React.ReactNode }) {
  return <section className="rounded-2xl border border-stone-200 bg-white p-6">{children}</section>;
}

export function NewWorkflowScreen() {
  const client = useOpsClient();
  const navigate = useNavigate();
  const toast = useToast();

  // Step a — outcome
  const [outcome, setOutcome] = useState<WorkflowOutcome>("mi");

  // Step b — who and which files
  const [clients, setClients] = useState<string[]>([]);
  const [clientChoice, setClientChoice] = useState("");
  const [newClientName, setNewClientName] = useState("");
  const [portfolio, setPortfolio] = useState("");
  const [period, setPeriod] = useState("");
  const [deliveries, setDeliveries] = useState<Delivery[]>([]);
  const [delivery, setDelivery] = useState<Delivery | null>(null);
  const [folder, setFolder] = useState("");
  const [registering, setRegistering] = useState(false);

  // Step c — classification override
  const [changeOpen, setChangeOpen] = useState(false);
  const [overrideType, setOverrideType] = useState<WorkflowType | "">("");
  const [overrideReason, setOverrideReason] = useState("");
  const [starting, setStarting] = useState(false);

  const clientId = clientChoice === NEW_CLIENT ? newClientName.trim() : clientChoice;

  useEffect(() => {
    client
      .getClients()
      .then(setClients)
      .catch(() => setClients([]));
  }, [client]);

  useEffect(() => {
    setDelivery(null);
    setChangeOpen(false);
    setOverrideType("");
    if (clientChoice && clientChoice !== NEW_CLIENT) {
      client
        .getDeliveries(clientChoice)
        .then(setDeliveries)
        .catch(() => setDeliveries([]));
    } else {
      setDeliveries([]);
    }
  }, [client, clientChoice]);

  async function handleRegister() {
    if (!clientId || !portfolio.trim() || !period || !folder.trim()) return;
    setRegistering(true);
    try {
      const registered = await client.registerDelivery({
        client_id: clientId,
        portfolio_id: portfolio.trim(),
        input_path: folder.trim(),
        reporting_period: period,
      });
      setDelivery(registered);
      setChangeOpen(false);
      setOverrideType("");
    } catch (err) {
      toast.show(errorMessage(err), "error");
    } finally {
      setRegistering(false);
    }
  }

  async function handleStart() {
    if (!clientId || !delivery) return;
    setStarting(true);
    try {
      const workflow = await client.startWorkflow({
        client_id: clientId,
        delivery_id: delivery.delivery_id,
        outcome,
        ...(overrideType
          ? {
              workflow_type: overrideType,
              ...(overrideReason.trim() ? { override_reason: overrideReason.trim() } : {}),
            }
          : {}),
        start: true,
      });
      navigate(`/workflows/${workflow.workflow_id}`);
    } catch (err) {
      toast.show(errorMessage(err), "error");
      setStarting(false);
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
                  "flex cursor-pointer flex-col gap-1 rounded-2xl border-2 p-5 transition-colors",
                  outcome === option.value
                    ? "border-blue-600 bg-blue-50/50"
                    : "border-stone-200 bg-white hover:border-stone-300",
                )}
              >
                <input
                  type="radio"
                  name="outcome"
                  value={option.value}
                  checked={outcome === option.value}
                  onChange={() => setOutcome(option.value)}
                  className="sr-only"
                />
                <span className="text-base font-semibold text-stone-900">{option.label}</span>
                <span className="text-sm text-stone-500">{option.help}</span>
              </label>
            ))}
          </div>
        </SectionCard>

        {/* (b) Client, portfolio, period, files */}
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
                onChange={(event) => setClientChoice(event.target.value)}
                className="w-full rounded-xl border border-stone-300 bg-white px-3 py-2.5 text-sm outline-none focus:border-blue-500 focus:ring-2 focus:ring-blue-100"
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
                  onChange={(event) => setNewClientName(event.target.value)}
                  className="w-full rounded-xl border border-stone-300 px-3 py-2.5 text-sm outline-none focus:border-blue-500 focus:ring-2 focus:ring-blue-100"
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
                onChange={(event) => setPortfolio(event.target.value)}
                className="w-full rounded-xl border border-stone-300 px-3 py-2.5 text-sm outline-none focus:border-blue-500 focus:ring-2 focus:ring-blue-100"
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
                onChange={(event) => setPeriod(event.target.value)}
                className="w-full rounded-xl border border-stone-300 px-3 py-2.5 text-sm outline-none focus:border-blue-500 focus:ring-2 focus:ring-blue-100"
              />
            </div>
          </div>

          <h3 className="mb-3 mt-8 text-sm font-semibold text-stone-700">
            {copy.newWorkflow.filesHeading}
          </h3>
          {clientChoice && clientChoice !== NEW_CLIENT && (
            <div className="mb-5">
              <p className="mb-2 text-xs font-medium uppercase tracking-wide text-stone-400">
                {copy.newWorkflow.pickDelivery}
              </p>
              {deliveries.length === 0 ? (
                <p className="text-sm text-stone-400">{copy.newWorkflow.noDeliveries}</p>
              ) : (
                <div className="space-y-2">
                  {deliveries.map((option) => (
                    <label
                      key={option.delivery_id}
                      className={clsx(
                        "flex cursor-pointer items-center gap-3 rounded-xl border p-3 transition-colors",
                        delivery?.delivery_id === option.delivery_id
                          ? "border-blue-600 bg-blue-50/50"
                          : "border-stone-200 hover:border-stone-300",
                      )}
                    >
                      <input
                        type="radio"
                        name="delivery"
                        checked={delivery?.delivery_id === option.delivery_id}
                        onChange={() => {
                          setDelivery(option);
                          setChangeOpen(false);
                          setOverrideType("");
                        }}
                        className="sr-only"
                      />
                      <FileText className="h-4 w-4 shrink-0 text-stone-400" aria-hidden />
                      <span className="min-w-0">
                        <span className="block truncate text-sm font-medium text-stone-900">
                          {option.portfolio_id} · {formatPeriod(option.reporting_period)}
                        </span>
                        <span className="block text-xs text-stone-500">
                          {option.files.length} {copy.newWorkflow.filesReceived} · {option.dataset}
                        </span>
                      </span>
                    </label>
                  ))}
                </div>
              )}
            </div>
          )}

          <p className="mb-2 text-xs font-medium uppercase tracking-wide text-stone-400">
            {copy.newWorkflow.orRegister}
          </p>
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
              disabled={registering || !clientId || !portfolio.trim() || !period || !folder.trim()}
              onClick={() => void handleRegister()}
              className="shrink-0 rounded-xl border border-stone-300 bg-white px-4 py-2.5 text-sm font-medium text-stone-700 transition-colors hover:bg-stone-50 disabled:cursor-not-allowed disabled:opacity-40"
            >
              {copy.newWorkflow.registerButton}
            </button>
          </div>
        </SectionCard>

        {/* (c) Classification + start */}
        {delivery && (
          <SectionCard>
            <div className="rounded-2xl bg-blue-50/70 p-5">
              <div className="mb-2 flex items-center gap-2">
                <Sparkles className="h-4 w-4 text-blue-600" aria-hidden />
                <span className="text-xs font-semibold uppercase tracking-wide text-blue-700">
                  {overrideType
                    ? TYPE_OPTIONS.find((t) => t.value === overrideType)?.label
                    : delivery.classification_label}
                </span>
              </div>
              <p className="text-base leading-relaxed text-stone-800">
                {delivery.classification_sentence}
              </p>
              <button
                type="button"
                onClick={() => setChangeOpen((open) => !open)}
                className="mt-3 text-sm font-medium text-blue-700 hover:underline"
              >
                {copy.newWorkflow.changeLink}
              </button>
            </div>

            {changeOpen && (
              <div className="mt-4 space-y-2">
                {TYPE_OPTIONS.map((option) => (
                  <label
                    key={option.value}
                    className={clsx(
                      "flex cursor-pointer items-center gap-3 rounded-xl border p-3 text-sm transition-colors",
                      (overrideType || delivery.classification) === option.value
                        ? "border-blue-600 bg-blue-50/50"
                        : "border-stone-200 hover:border-stone-300",
                    )}
                  >
                    <input
                      type="radio"
                      name="workflow-type"
                      checked={(overrideType || delivery.classification) === option.value}
                      onChange={() =>
                        setOverrideType(option.value === delivery.classification ? "" : option.value)
                      }
                    />
                    <span className="font-medium text-stone-800">{option.label}</span>
                  </label>
                ))}
                <div>
                  <label
                    className="mb-1 mt-2 block text-sm font-medium text-stone-700"
                    htmlFor="override-reason"
                  >
                    {copy.common.optionalReason}
                  </label>
                  <input
                    id="override-reason"
                    value={overrideReason}
                    onChange={(event) => setOverrideReason(event.target.value)}
                    className="w-full rounded-xl border border-stone-300 px-3 py-2.5 text-sm outline-none focus:border-blue-500 focus:ring-2 focus:ring-blue-100"
                  />
                </div>
              </div>
            )}

            <button
              type="button"
              disabled={starting || !clientId}
              onClick={() => void handleStart()}
              className="mt-6 w-full rounded-xl bg-blue-600 px-4 py-3 text-base font-semibold text-white transition-colors hover:bg-blue-700 disabled:cursor-not-allowed disabled:opacity-40 sm:w-auto sm:px-10"
            >
              {copy.newWorkflow.startButton}
            </button>
          </SectionCard>
        )}
      </div>
    </Page>
  );
}
