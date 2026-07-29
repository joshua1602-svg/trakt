import { useEffect, useMemo, useState, type ReactNode } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { AlertTriangle, OctagonAlert, RefreshCw } from "lucide-react";
import clsx from "clsx";
import { useOpsClient } from "@/api/context";
import type { Stage, Workflow } from "@/api/types";
import { ErrorNote, Loading } from "@/components/ErrorNote";
import { EvidenceView } from "@/components/EvidenceView";
import { Page } from "@/components/Page";
import { StatusChip } from "@/components/StatusChip";
import { useToast } from "@/components/Toast";
import { copy, decisionsLabel } from "@/lib/copy";
import { formatPeriod } from "@/lib/format";
import { errorMessage, useLoad } from "@/lib/useLoad";

const STAGE_CHIP_STYLES: Record<string, string> = {
  completed: "border-emerald-200 bg-emerald-50 text-emerald-800",
  approved: "border-emerald-400 bg-white text-emerald-700",
  needs_review: "border-amber-200 bg-amber-50 text-amber-800",
  blocked: "border-rose-200 bg-rose-50 text-rose-800",
  running: "border-blue-200 bg-blue-50 text-blue-800 animate-pulse",
  waiting: "border-stone-200 bg-stone-100 text-stone-500",
  ready: "border-violet-200 bg-violet-50 text-violet-800",
  rejected: "border-rose-200 bg-rose-50 text-rose-800",
};

function Modal({ children }: { children: ReactNode }) {
  return (
    <div className="fixed inset-0 z-40 flex items-center justify-center bg-stone-900/30 px-6">
      <div className="w-full max-w-md rounded-2xl border border-stone-200 bg-white p-6 shadow-xl">
        {children}
      </div>
    </div>
  );
}

function StagePanel({ stage, workflowId }: { stage: Stage; workflowId: string }) {
  return (
    <div className="rounded-2xl border border-stone-200 bg-white p-6">
      <div className="mb-4 flex items-center justify-between gap-3">
        <h3 className="text-base font-semibold text-stone-900">{stage.label}</h3>
        <StatusChip status={stage.status} label={stage.status_label} />
      </div>
      {stage.summary && <p className="text-lg leading-relaxed text-stone-800">{stage.summary}</p>}
      {stage.why_it_matters && (
        <p className="mt-2 text-sm leading-relaxed text-stone-500">{stage.why_it_matters}</p>
      )}
      {stage.warnings && stage.warnings.length > 0 && (
        <div className="mt-4 space-y-2">
          {stage.warnings.map((warning, i) => (
            <p
              key={i}
              className="flex items-start gap-2 rounded-xl bg-amber-50 px-3 py-2 text-sm text-amber-900"
            >
              <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" aria-hidden />
              {warning}
            </p>
          ))}
        </div>
      )}
      {stage.blockers && stage.blockers.length > 0 && (
        <div className="mt-4 space-y-2">
          {stage.blockers.map((blocker, i) => (
            <p
              key={i}
              className="flex items-start gap-2 rounded-xl bg-rose-50 px-3 py-2 text-sm text-rose-900"
            >
              <OctagonAlert className="mt-0.5 h-4 w-4 shrink-0" aria-hidden />
              {blocker}
            </p>
          ))}
        </div>
      )}
      {typeof stage.decision_count === "number" && stage.decision_count > 0 && (
        <p className="mt-4 text-sm text-stone-600">
          <Link
            to={`/reviews?workflow=${workflowId}`}
            className="font-medium text-blue-700 hover:underline"
          >
            {decisionsLabel(stage.decision_count)}
          </Link>
        </p>
      )}
      {stage.evidence && stage.evidence.length > 0 && (
        <div className="mt-5">
          <EvidenceView items={stage.evidence} />
        </div>
      )}
    </div>
  );
}

function pickActiveStage(workflow: Workflow): string {
  const active = workflow.stages.find((s) =>
    ["needs_review", "blocked", "running", "ready"].includes(s.status),
  );
  if (active) return active.stage;
  const lastDone = [...workflow.stages].reverse().find((s) => s.status !== "waiting");
  return (lastDone ?? workflow.stages[0])?.stage ?? "";
}

export function WorkflowDetailScreen() {
  const { id = "" } = useParams();
  const client = useOpsClient();
  const navigate = useNavigate();
  const toast = useToast();

  const { data: workflow, error, loading, reload } = useLoad(() => client.getWorkflow(id), [id]);
  const [selectedStage, setSelectedStage] = useState<string | null>(null);
  const [confirmPublish, setConfirmPublish] = useState(false);
  const [holdOpen, setHoldOpen] = useState(false);
  const [holdReason, setHoldReason] = useState("");
  const [busy, setBusy] = useState(false);

  // Poll while running.
  useEffect(() => {
    if (workflow?.status !== "running") return;
    const timer = setInterval(() => void reload({ quiet: true }), 2500);
    return () => clearInterval(timer);
  }, [workflow?.status, reload]);

  const activeStage = useMemo(() => (workflow ? pickActiveStage(workflow) : ""), [workflow]);
  const shownStage = workflow?.stages.find((s) => s.stage === (selectedStage ?? activeStage));

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
  const canPublish = workflow?.status === "awaiting_publication";

  return (
    <Page
      title={workflow ? `${workflow.client_id} · ${workflow.portfolio_id}` : copy.workflows.title}
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

          {/* Stage tracker */}
          <div className="flex gap-2 overflow-x-auto pb-1">
            {workflow.stages.map((stage) => (
              <button
                key={stage.stage}
                type="button"
                onClick={() => setSelectedStage(stage.stage)}
                className={clsx(
                  "flex shrink-0 flex-col items-start gap-0.5 rounded-xl border px-4 py-2.5 text-left transition-shadow",
                  STAGE_CHIP_STYLES[stage.status] ?? STAGE_CHIP_STYLES.waiting,
                  (selectedStage ?? activeStage) === stage.stage
                    ? "ring-2 ring-stone-900/20"
                    : "hover:shadow-sm",
                )}
              >
                <span className="text-sm font-medium">{stage.label}</span>
                <span className="text-xs opacity-75">{stage.status_label}</span>
              </button>
            ))}
          </div>

          {shownStage && <StagePanel stage={shownStage} workflowId={workflow.workflow_id} />}

          {/* Contextual actions */}
          <div className="flex flex-wrap items-center gap-3">
            {workflow.open_decisions > 0 && (
              <Link
                to={`/reviews?workflow=${workflow.workflow_id}`}
                className="rounded-xl bg-blue-600 px-4 py-2.5 text-sm font-semibold text-white transition-colors hover:bg-blue-700"
              >
                {copy.workflow.reviewDecisions}
              </Link>
            )}
            {canPublish && (
              <>
                <button
                  type="button"
                  disabled={busy}
                  onClick={() => setConfirmPublish(true)}
                  className="rounded-xl bg-blue-600 px-4 py-2.5 text-sm font-semibold text-white transition-colors hover:bg-blue-700 disabled:opacity-40"
                >
                  {copy.workflow.approvePublish}
                </button>
                <button
                  type="button"
                  disabled={busy}
                  onClick={() => setHoldOpen(true)}
                  className="rounded-xl border border-stone-300 bg-white px-4 py-2.5 text-sm font-medium text-stone-700 transition-colors hover:bg-stone-50 disabled:opacity-40"
                >
                  {copy.workflow.hold}
                </button>
              </>
            )}
            {canRerun && (
              <button
                type="button"
                disabled={busy}
                onClick={() => void run(() => client.rerunWorkflow(workflow.workflow_id))}
                className="flex items-center gap-2 rounded-xl bg-stone-900 px-4 py-2.5 text-sm font-semibold text-white transition-colors hover:bg-stone-700 disabled:opacity-40"
              >
                <RefreshCw className="h-4 w-4" aria-hidden />
                {copy.workflow.runAgain}
              </button>
            )}
          </div>
        </div>
      )}

      {confirmPublish && workflow && (
        <Modal>
          <p className="mb-6 text-base text-stone-800">{copy.workflow.publishConfirm}</p>
          <div className="flex justify-end gap-3">
            <button
              type="button"
              onClick={() => setConfirmPublish(false)}
              className="rounded-xl border border-stone-300 bg-white px-4 py-2 text-sm font-medium text-stone-700 hover:bg-stone-50"
            >
              {copy.common.cancel}
            </button>
            <button
              type="button"
              disabled={busy}
              onClick={() => {
                setConfirmPublish(false);
                void run(() => client.publishWorkflow(workflow.workflow_id));
              }}
              className="rounded-xl bg-blue-600 px-4 py-2 text-sm font-semibold text-white hover:bg-blue-700 disabled:opacity-40"
            >
              {copy.workflow.publishButton}
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
