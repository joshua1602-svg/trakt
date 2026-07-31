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
import type { StepDecision, WorkflowStep } from "@/api/types";
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

function Modal({ children, labelledBy }: { children: ReactNode; labelledBy?: string }) {
  return (
    <div className="fixed inset-0 z-40 flex items-center justify-center bg-stone-900/30 px-6">
      <div
        role="dialog"
        aria-modal="true"
        aria-labelledby={labelledBy}
        className="max-h-[85vh] w-full max-w-md overflow-y-auto rounded-2xl border border-stone-200 bg-white p-6 shadow-xl"
      >
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

/** The approval decision, inside the delivery's own context. */
function ApprovalPanel({
  step,
  busy,
  onPublish,
  onHold,
}: {
  step: WorkflowStep;
  busy: boolean;
  onPublish: (scope: string) => void;
  onHold: () => void;
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
  onDecisionResolved,
}: {
  step: WorkflowStep;
  busy: boolean;
  onPublish: (scope: string) => void;
  onHold: () => void;
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
        <ApprovalPanel step={step} busy={busy} onPublish={onPublish} onHold={onHold} />
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
  const [cancelOpen, setCancelOpen] = useState(false);
  const [cancelReason, setCancelReason] = useState("");
  const [busy, setBusy] = useState(false);

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
  // The engine allows cancelling from every status except the two terminal
  // ones. Anything else is a delivery somebody may still be waiting on.
  const canCancel = workflow && !["published", "cancelled"].includes(workflow.status);

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

          {workflow.status === "cancelled" && (
            <p className="rounded-xl border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-900">
              {copy.workflow.cancelledNote}
            </p>
          )}

          {/* Low emphasis and below everything else: cancelling is always
              available on a live delivery, and never the suggested move. */}
          {canCancel && (
            <div className="border-t border-stone-200 pt-6">
              <button
                type="button"
                disabled={busy}
                onClick={() => setCancelOpen(true)}
                className="text-sm font-medium text-stone-500 underline-offset-4 transition-colors hover:text-stone-900 hover:underline disabled:opacity-40"
              >
                {copy.workflow.cancel}
              </button>
            </div>
          )}
        </div>
      )}

      {cancelOpen && workflow && (
        <Modal labelledBy="cancel-heading">
          <p id="cancel-heading" className="text-base font-semibold text-stone-900">
            {copy.workflow.cancelHeading}
          </p>
          <p className="mt-3 text-sm leading-relaxed text-stone-700">
            {copy.workflow.cancelExplain}
          </p>
          <label className="mb-2 mt-5 block text-sm text-stone-800" htmlFor="cancel-reason">
            {copy.workflow.cancelPrompt}
          </label>
          <textarea
            id="cancel-reason"
            value={cancelReason}
            onChange={(event) => setCancelReason(event.target.value)}
            rows={3}
            className="mb-4 w-full rounded-xl border border-stone-300 px-3 py-2 text-sm outline-none focus:border-blue-500 focus:ring-2 focus:ring-blue-100"
          />
          <div className="flex justify-end gap-3">
            <button
              type="button"
              onClick={() => setCancelOpen(false)}
              className="rounded-xl border border-stone-300 bg-white px-4 py-2 text-sm font-medium text-stone-700 hover:bg-stone-50"
            >
              {copy.common.cancel}
            </button>
            <button
              type="button"
              disabled={busy || !cancelReason.trim()}
              onClick={() => {
                setCancelOpen(false);
                void run(
                  () => client.cancelWorkflow(workflow.workflow_id, cancelReason.trim()),
                  copy.workflow.cancelledToast,
                );
              }}
              className="rounded-xl bg-stone-900 px-4 py-2 text-sm font-semibold text-white hover:bg-stone-700 disabled:opacity-40"
            >
              {copy.workflow.cancelButton}
            </button>
          </div>
        </Modal>
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
