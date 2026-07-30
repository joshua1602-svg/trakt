import { useEffect, useMemo, useState, type ReactNode } from "react";
import { useNavigate, useParams, useSearchParams } from "react-router-dom";
import {
  AlertTriangle,
  Check,
  ChevronDown,
  FileText,
  OctagonAlert,
  RefreshCw,
} from "lucide-react";
import clsx from "clsx";
import { useOpsClient } from "@/api/context";
import type { StepDecision, WithdrawalReason, WorkflowStep } from "@/api/types";
import { DecisionPanel } from "@/components/DecisionPanel";
import { ErrorNote, Loading } from "@/components/ErrorNote";
import { EvidenceView } from "@/components/EvidenceView";
import { Page } from "@/components/Page";
import { StatusChip } from "@/components/StatusChip";
import { useToast } from "@/components/Toast";
import { copy } from "@/lib/copy";
import { formatDate, formatPeriod } from "@/lib/format";
import { errorMessage, useLoad } from "@/lib/useLoad";

const APPROVAL_STEP = "publication_approval";
const ISSUES_STEP = "issues_reviewed";

function Modal({ children }: { children: ReactNode }) {
  return (
    <div className="fixed inset-0 z-40 flex items-center justify-center bg-stone-900/30 px-6">
      <div className="max-h-[85vh] w-full max-w-md overflow-y-auto rounded-2xl border border-stone-200 bg-white p-6 shadow-xl">
        {children}
      </div>
    </div>
  );
}

function Facts({ step }: { step: WorkflowStep }) {
  if (step.facts.length === 0) return null;
  return (
    <dl className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
      {step.facts.map((fact) => (
        <div key={fact.label}>
          <dt className="text-xs text-stone-500">{fact.label}</dt>
          <dd className="mt-0.5 text-sm leading-relaxed text-stone-900">{fact.value}</dd>
        </div>
      ))}
    </dl>
  );
}

function FileList({ step }: { step: WorkflowStep }) {
  if (!step.files || step.files.length === 0) return null;
  return (
    <div>
      <h4 className="mb-2 text-sm font-semibold text-stone-700">{copy.workflow.filesHeading}</h4>
      <ul className="space-y-2">
        {step.files.map((file) => (
          <li
            key={`${file.filename}-${file.checksum}`}
            className="flex items-start gap-3 rounded-xl border border-stone-200 bg-white px-4 py-3"
          >
            <FileText className="mt-0.5 h-4 w-4 shrink-0 text-stone-400" aria-hidden />
            <div className="min-w-0">
              <p className="truncate text-sm font-medium text-stone-900">{file.filename}</p>
              <p className="mt-0.5 text-xs text-stone-500">
                {[
                  file.role_label,
                  file.size && `${copy.workflow.fileSize} ${file.size}`,
                  file.received_at && `${copy.workflow.fileArrived} ${formatDate(file.received_at)}`,
                  file.checksum && `${copy.workflow.fileCheck} ${file.checksum}`,
                ]
                  .filter(Boolean)
                  .join(" · ")}
              </p>
            </div>
          </li>
        ))}
      </ul>
    </div>
  );
}

function DecisionSummary({ decisions, heading }: { decisions: StepDecision[]; heading: string }) {
  if (decisions.length === 0) return null;
  return (
    <div>
      <h4 className="mb-2 text-sm font-semibold text-stone-700">{heading}</h4>
      <ul className="space-y-2">
        {decisions.map((decision) => (
          <li
            key={decision.decision_id}
            className="rounded-xl border border-stone-200 bg-white px-4 py-3"
          >
            <p className="text-sm font-medium text-stone-900">{decision.title}</p>
            {decision.answer && (
              <p className="mt-0.5 text-sm text-stone-700">{decision.answer}</p>
            )}
            {(decision.actor || decision.at) && (
              <p className="mt-0.5 text-xs text-stone-500">
                {copy.workflow.answeredBy} {decision.actor || "—"}
                {decision.at ? ` · ${formatDate(decision.at)}` : ""}
              </p>
            )}
          </li>
        ))}
      </ul>
    </div>
  );
}

function Notes({ items, tone }: { items: string[]; tone: "warning" | "blocker" }) {
  if (items.length === 0) return null;
  const Icon = tone === "warning" ? AlertTriangle : OctagonAlert;
  return (
    <div className="space-y-2">
      {items.map((text, i) => (
        <p
          key={i}
          className={clsx(
            "flex items-start gap-2 rounded-xl px-3 py-2 text-sm",
            tone === "warning" ? "bg-amber-50 text-amber-900" : "bg-rose-50 text-rose-900",
          )}
        >
          <Icon className="mt-0.5 h-4 w-4 shrink-0" aria-hidden />
          {text}
        </p>
      ))}
    </div>
  );
}

/**
 * Taking a delivery out of active work.
 *
 * Deliberately not called "Cancel": cancelling reads as closing the dialog
 * without doing anything, and this does something. It is a tertiary action —
 * quiet next to "Approve and publish" until it is chosen, then destructive,
 * because withdrawing is the end of the road for this delivery.
 */
function WithdrawDialog({
  busy,
  reasons,
  onCancel,
  onConfirm,
}: {
  busy: boolean;
  reasons: WithdrawalReason[];
  onCancel: () => void;
  onConfirm: (reasonCode: string, note: string) => void;
}) {
  const [reason, setReason] = useState("");
  const [note, setNote] = useState("");
  const noteRequired = reason === "other";
  const ready = Boolean(reason) && (!noteRequired || note.trim().length > 0);

  return (
    <Modal>
      <h2 className="text-lg font-semibold text-stone-900">
        {copy.workflow.withdrawHeading}
      </h2>
      <p className="mt-2 text-sm leading-relaxed text-stone-600">
        {copy.workflow.withdrawBody}
      </p>

      <fieldset className="mt-5">
        <legend className="mb-2 text-sm font-semibold text-stone-700">
          {copy.workflow.withdrawReason}
        </legend>
        <div className="space-y-2">
          {reasons.map((option) => (
            <label
              key={option.value}
              className={clsx(
                "flex cursor-pointer items-center gap-3 rounded-xl border p-3 text-sm transition-colors",
                reason === option.value
                  ? "border-stone-900 bg-stone-50"
                  : "border-stone-200 bg-white hover:border-stone-300",
              )}
            >
              <input
                type="radio"
                name="withdraw-reason"
                value={option.value}
                checked={reason === option.value}
                onChange={() => setReason(option.value)}
              />
              <span className="font-medium text-stone-800">{option.label}</span>
            </label>
          ))}
        </div>
      </fieldset>

      <label className="mt-4 block text-sm font-medium text-stone-700" htmlFor="withdraw-note">
        {noteRequired ? copy.workflow.withdrawNoteRequired : copy.workflow.withdrawNote}
      </label>
      <textarea
        id="withdraw-note"
        rows={2}
        value={note}
        onChange={(event) => setNote(event.target.value)}
        className="mt-1 w-full rounded-xl border border-stone-300 px-3 py-2 text-sm outline-none focus:border-blue-500 focus:ring-2 focus:ring-blue-100"
      />

      <div className="mt-6 flex flex-wrap justify-end gap-3">
        <button
          type="button"
          onClick={onCancel}
          className="rounded-xl border border-stone-300 bg-white px-4 py-2 text-sm font-medium text-stone-700 hover:bg-stone-50"
        >
          {copy.workflow.withdrawKeep}
        </button>
        <button
          type="button"
          disabled={busy || !ready}
          onClick={() => onConfirm(reason, note.trim())}
          className="rounded-xl bg-rose-600 px-4 py-2 text-sm font-semibold text-white hover:bg-rose-700 disabled:opacity-40"
        >
          {copy.workflow.withdrawConfirm}
        </button>
      </div>
    </Modal>
  );
}

/** The approval decision, inside the delivery's own context. */
function ApprovalPanel({
  step,
  busy,
  onPublish,
  onHold,
  onWithdraw,
  canWithdraw,
}: {
  step: WorkflowStep;
  busy: boolean;
  onPublish: (scope: string) => void;
  onHold: () => void;
  onWithdraw: () => void;
  canWithdraw: boolean;
}) {
  const approval = step.approval;
  const [scope, setScope] = useState(approval?.default_scope ?? "delivery");
  const [confirming, setConfirming] = useState(false);
  if (!approval) return null;

  const consequence = [approval.consequence, approval.scope_consequences[scope]]
    .filter(Boolean)
    .join(" ");

  return (
    <div className="space-y-5">
      <div className="rounded-2xl border border-violet-200 bg-violet-50/50 p-5">
        <p className="text-base font-semibold text-stone-900">{approval.headline}</p>
        <ul className="mt-2 space-y-1">
          {approval.evidence_lines.map((line, i) => (
            <li key={i} className="flex items-start gap-2 text-sm text-stone-700">
              <Check className="mt-0.5 h-3.5 w-3.5 shrink-0 text-violet-600" aria-hidden />
              {line}
            </li>
          ))}
        </ul>
      </div>

      {approval.available && (
        <>
          <h4 className="text-base font-semibold leading-relaxed text-stone-900">
            {approval.question}
          </h4>

          <div>
            <h5 className="text-sm font-semibold text-stone-700">{approval.scope_question}</h5>
            {approval.scope_note && (
              <p className="mb-2 mt-1 text-xs leading-relaxed text-stone-500">
                {approval.scope_note}
              </p>
            )}
            <div className="mt-2 space-y-2">
              {approval.scopes.map((option) => (
                <label
                  key={option.value}
                  className={clsx(
                    "flex cursor-pointer items-start gap-3 rounded-xl border p-3.5 transition-colors",
                    scope === option.value
                      ? "border-blue-600 bg-blue-50/50"
                      : "border-stone-200 bg-white hover:border-stone-300",
                  )}
                >
                  <input
                    type="radio"
                    name="approval-scope"
                    value={option.value}
                    checked={scope === option.value}
                    onChange={() => setScope(option.value)}
                    className="mt-0.5"
                  />
                  <span>
                    <span className="block text-sm font-medium text-stone-800">{option.label}</span>
                    <span className="mt-0.5 block text-xs text-stone-500">
                      {option.explanation}
                    </span>
                  </span>
                </label>
              ))}
            </div>
          </div>

          <p className="rounded-xl border border-stone-200 bg-stone-50 px-4 py-3 text-sm leading-relaxed text-stone-700">
            {consequence}
          </p>

          <div className="flex flex-wrap items-center gap-3">
            <button
              type="button"
              disabled={busy}
              onClick={() => setConfirming(true)}
              className="rounded-xl bg-blue-600 px-6 py-2.5 text-sm font-semibold text-white transition-colors hover:bg-blue-700 disabled:opacity-40"
            >
              {copy.workflow.approvePublish}
            </button>
            <button
              type="button"
              disabled={busy}
              onClick={onHold}
              className="rounded-xl border border-stone-300 bg-white px-4 py-2.5 text-sm font-medium text-stone-700 transition-colors hover:bg-stone-50 disabled:opacity-40"
            >
              {copy.workflow.hold}
            </button>
          </div>
        </>
      )}

      {/* Tertiary, and outside the approval block so it is reachable at every
          active stage — a test delivery rarely gets as far as approval. */}
      {canWithdraw && (
        <div className="border-t border-stone-100 pt-4">
          <button
            type="button"
            disabled={busy}
            onClick={onWithdraw}
            className="rounded-xl px-3 py-2 text-sm font-medium text-stone-500 underline-offset-4 transition-colors hover:text-rose-700 hover:underline disabled:opacity-40"
          >
            {copy.workflow.withdraw}
          </button>
        </div>
      )}

      {confirming && (
        <Modal>
          <p className="text-base font-semibold text-stone-900">{approval.question}</p>
          <p className="mt-3 text-sm leading-relaxed text-stone-700">{consequence}</p>
          <div className="mt-6 flex justify-end gap-3">
            <button
              type="button"
              onClick={() => setConfirming(false)}
              className="rounded-xl border border-stone-300 bg-white px-4 py-2 text-sm font-medium text-stone-700 hover:bg-stone-50"
            >
              {copy.common.cancel}
            </button>
            <button
              type="button"
              disabled={busy}
              onClick={() => {
                setConfirming(false);
                onPublish(scope);
              }}
              className="rounded-xl bg-blue-600 px-4 py-2 text-sm font-semibold text-white hover:bg-blue-700 disabled:opacity-40"
            >
              {copy.workflow.approveConfirm}
            </button>
          </div>
        </Modal>
      )}
    </div>
  );
}

function StepBody({
  step,
  busy,
  onPublish,
  onHold,
  onWithdraw,
  canWithdraw,
  onDecisionResolved,
}: {
  step: WorkflowStep;
  busy: boolean;
  onPublish: (scope: string) => void;
  onHold: () => void;
  onWithdraw: () => void;
  canWithdraw: boolean;
  onDecisionResolved: () => void;
}) {
  return (
    <div className="space-y-5">
      {step.summary && <p className="text-base leading-relaxed text-stone-800">{step.summary}</p>}
      {step.status === "pending" && (
        <p className="text-sm text-stone-500">{copy.workflow.nextPending}</p>
      )}
      {step.status === "not_applicable" && (
        <p className="text-sm text-stone-500">{copy.workflow.nextNotApplicable}</p>
      )}
      <Facts step={step} />
      <FileList step={step} />
      <Notes items={step.blockers} tone="blocker" />
      <Notes items={step.warnings} tone="warning" />

      {step.key === ISSUES_STEP && (
        <>
          <DecisionSummary decisions={step.resolved ?? []} heading={copy.workflow.answered} />
          {(step.unresolved ?? []).length > 0 && (
            <div className="space-y-3">
              <h4 className="text-sm font-semibold text-stone-700">{copy.workflow.stillOpen}</h4>
              {(step.unresolved ?? []).map((decision) => (
                <DecisionPanel
                  key={decision.decision_id}
                  decisionId={decision.decision_id}
                  onResolved={onDecisionResolved}
                />
              ))}
            </div>
          )}
        </>
      )}

      {step.key === APPROVAL_STEP && (
        <ApprovalPanel
          step={step}
          busy={busy}
          onPublish={onPublish}
          onHold={onHold}
          onWithdraw={onWithdraw}
          canWithdraw={canWithdraw}
        />
      )}

      {step.outputs && step.outputs.length > 0 && (
        <div>
          <h4 className="mb-2 text-sm font-semibold text-stone-700">
            {copy.workflow.outputsHeading}
          </h4>
          <ul className="flex flex-wrap gap-2">
            {step.outputs.map((output) => (
              <li
                key={output}
                className="rounded-full border border-emerald-200 bg-emerald-50 px-3 py-1 text-xs font-medium text-emerald-700"
              >
                {output}
              </li>
            ))}
          </ul>
        </div>
      )}

      {step.evidence && step.evidence.length > 0 && <EvidenceView items={step.evidence} />}
    </div>
  );
}

export function WorkflowDetailScreen() {
  const { id = "" } = useParams();
  const client = useOpsClient();
  const navigate = useNavigate();
  const toast = useToast();
  const [searchParams] = useSearchParams();

  const { data: workflow, error, loading, reload } = useLoad(() => client.getWorkflow(id), [id]);
  const [openStep, setOpenStep] = useState<string | null>(null);
  const [holdOpen, setHoldOpen] = useState(false);
  const [holdReason, setHoldReason] = useState("");
  const [withdrawOpen, setWithdrawOpen] = useState(false);
  const [withdrawReasons, setWithdrawReasons] = useState<WithdrawalReason[]>([]);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    client
      .getWithdrawalReasons()
      .then(setWithdrawReasons)
      .catch(() => setWithdrawReasons([]));
  }, [client]);

  // Poll while running.
  useEffect(() => {
    if (workflow?.status !== "running") return;
    const timer = setInterval(() => void reload({ quiet: true }), 2500);
    return () => clearInterval(timer);
  }, [workflow?.status, reload]);

  const steps = useMemo<WorkflowStep[]>(() => workflow?.steps ?? [], [workflow]);
  // Deep links land on a step: ?step= names one directly, ?decision= (an
  // existing approval/review link) lands on the step that owns the question.
  const requestedStep = searchParams.get("step");
  const requestedDecision = searchParams.get("decision");
  const landingStep =
    (requestedStep && steps.some((s) => s.key === requestedStep) && requestedStep) ||
    (requestedDecision ? ISSUES_STEP : "") ||
    workflow?.current_step ||
    steps[0]?.key ||
    "";
  const shown = openStep ?? landingStep;

  async function run(action: () => Promise<unknown>, successMessage?: string) {
    setBusy(true);
    try {
      await action();
      if (successMessage) toast.show(successMessage, "success");
      await reload({ quiet: true });
    } catch (err) {
      toast.show(errorMessage(err), "error");
    } finally {
      setBusy(false);
    }
  }

  const canRerun =
    workflow &&
    (["blocked", "failed", "held"].includes(workflow.status) || workflow.interrupted);
  // A delivery that has been published, cancelled or already withdrawn is
  // finished; there is nothing left to take out of active work.
  const canWithdraw = Boolean(
    workflow && !["published", "cancelled", "withdrawn"].includes(workflow.status),
  );

  return (
    <Page
      title={workflow ? `${workflow.client_id} · ${workflow.portfolio_id}` : copy.workflow.caseFile}
      subtitle={
        workflow
          ? `${workflow.outcome_label} · ${formatPeriod(workflow.reporting_period)}`
          : undefined
      }
      actions={
        workflow && (
          <span className="rounded-full border border-stone-200 bg-white px-3 py-1 text-xs font-medium text-stone-600">
            {workflow.workflow_type_label}
          </span>
        )
      }
    >
      {loading && !workflow && <Loading />}
      {error && !workflow && <ErrorNote message={error} onRetry={() => void reload()} />}
      {workflow && (
        <div className="space-y-8">
          <div className="flex flex-wrap items-center gap-3">
            <StatusChip status={workflow.status} />
            <p className="text-base leading-relaxed text-stone-700">{workflow.status_sentence}</p>
          </div>

          {workflow.blockers.length > 0 && (
            <div className="space-y-2">
              {workflow.blockers.map((blocker, i) => (
                <p
                  key={i}
                  className="flex items-start gap-2 rounded-xl border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-900"
                >
                  <OctagonAlert className="mt-0.5 h-4 w-4 shrink-0" aria-hidden />
                  {blocker}
                </p>
              ))}
            </div>
          )}

          {/* The case file. Every stage stays visible and expandable, including
              the ones already behind us. */}
          <section>
            <h2 className="mb-3 text-sm font-semibold text-stone-700">
              {copy.workflow.stepsHeading}
            </h2>
            <ol className="space-y-2">
              {steps.map((step, index) => {
                const expanded = shown === step.key;
                return (
                  <li
                    key={step.key}
                    data-step={step.key}
                    data-step-status={step.status}
                    className={clsx(
                      "rounded-2xl border bg-white transition-colors",
                      expanded ? "border-stone-300 shadow-sm" : "border-stone-200",
                    )}
                  >
                    <button
                      type="button"
                      aria-expanded={expanded}
                      onClick={() => setOpenStep(expanded ? "" : step.key)}
                      className="flex w-full items-center gap-3 px-4 py-3.5 text-left sm:px-6"
                    >
                      <span
                        className={clsx(
                          "flex h-7 w-7 shrink-0 items-center justify-center rounded-full text-xs font-semibold",
                          step.status === "complete"
                            ? "bg-emerald-100 text-emerald-700"
                            : step.status === "current"
                              ? "bg-blue-600 text-white"
                              : step.status === "blocked"
                                ? "bg-rose-100 text-rose-700"
                                : "bg-stone-100 text-stone-400",
                        )}
                      >
                        {step.status === "complete" ? (
                          <Check className="h-4 w-4" aria-hidden />
                        ) : (
                          index + 1
                        )}
                      </span>
                      <span className="min-w-0 flex-1">
                        {/* The step name always reads in full — it wraps on a
                            narrow screen rather than being cut off. */}
                        <span className="block text-sm font-semibold text-stone-900">
                          {step.label}
                        </span>
                        {!expanded && step.summary && (
                          <span className="mt-0.5 block truncate text-xs text-stone-500">
                            {step.summary}
                          </span>
                        )}
                      </span>
                      <StatusChip status={step.status} label={step.status_label} />
                      <ChevronDown
                        className={clsx(
                          "h-4 w-4 shrink-0 text-stone-400 transition-transform",
                          expanded && "rotate-180",
                        )}
                        aria-hidden
                      />
                    </button>
                    {expanded && (
                      <div className="border-t border-stone-100 px-4 py-5 sm:px-6">
                        <StepBody
                          step={step}
                          busy={busy}
                          onPublish={(scope) =>
                            void run(() => client.publishWorkflow(workflow.workflow_id, scope))
                          }
                          onHold={() => setHoldOpen(true)}
                          onWithdraw={() => setWithdrawOpen(true)}
                          canWithdraw={canWithdraw}
                          onDecisionResolved={() => void reload({ quiet: true })}
                        />
                      </div>
                    )}
                  </li>
                );
              })}
            </ol>
          </section>

          {canRerun && (
            <div className="flex flex-wrap items-center gap-3">
              <button
                type="button"
                disabled={busy}
                onClick={() => void run(() => client.rerunWorkflow(workflow.workflow_id))}
                className="flex items-center gap-2 rounded-xl bg-stone-900 px-4 py-2.5 text-sm font-semibold text-white transition-colors hover:bg-stone-700 disabled:opacity-40"
              >
                <RefreshCw className="h-4 w-4" aria-hidden />
                {copy.workflow.runAgain}
              </button>
            </div>
          )}
        </div>
      )}

      {withdrawOpen && workflow && (
        <WithdrawDialog
          busy={busy}
          reasons={withdrawReasons}
          onCancel={() => setWithdrawOpen(false)}
          onConfirm={(reasonCode, note) => {
            setWithdrawOpen(false);
            void run(() => client.withdrawWorkflow(workflow.workflow_id, reasonCode, note));
          }}
        />
      )}

      {holdOpen && workflow && (
        <Modal>
          <label className="mb-2 block text-base text-stone-800" htmlFor="hold-reason">
            {copy.workflow.holdPrompt}
          </label>
          <textarea
            id="hold-reason"
            value={holdReason}
            onChange={(event) => setHoldReason(event.target.value)}
            rows={3}
            className="mb-4 w-full rounded-xl border border-stone-300 px-3 py-2 text-sm outline-none focus:border-blue-500 focus:ring-2 focus:ring-blue-100"
          />
          <div className="flex justify-end gap-3">
            <button
              type="button"
              onClick={() => setHoldOpen(false)}
              className="rounded-xl border border-stone-300 bg-white px-4 py-2 text-sm font-medium text-stone-700 hover:bg-stone-50"
            >
              {copy.common.cancel}
            </button>
            <button
              type="button"
              disabled={busy || !holdReason.trim()}
              onClick={() => {
                setHoldOpen(false);
                void run(() => client.holdWorkflow(workflow.workflow_id, holdReason.trim()));
              }}
              className="rounded-xl bg-stone-900 px-4 py-2 text-sm font-semibold text-white hover:bg-stone-700 disabled:opacity-40"
            >
              {copy.workflow.holdButton}
            </button>
          </div>
        </Modal>
      )}

      {!loading && !workflow && !error && (
        <div className="text-sm text-stone-500">
          {copy.workflow.notFound}{" "}
          <button type="button" className="text-blue-700 hover:underline" onClick={() => navigate(-1)}>
            {copy.common.close}
          </button>
        </div>
      )}
    </Page>
  );
}
