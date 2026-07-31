import { useState } from "react";
import { Link, useParams } from "react-router-dom";
import { ArrowLeft, CheckCircle2, CircleDot, ExternalLink } from "lucide-react";
import clsx from "clsx";
import { useOpsClient } from "@/api/context";
import type {
  AgentDisclosure,
  AgentProposal,
  AgentStatus,
  DecisionCard,
  ReadinessCriterion,
} from "@/api/agentTypes";
import type { ChecklistRow, InformationRequest } from "@/api/onboardingTypes";
import { ErrorNote, Loading } from "@/components/ErrorNote";
import { Page } from "@/components/Page";
import { StatusChip } from "@/components/StatusChip";
import { useToast } from "@/components/Toast";
import { copy } from "@/lib/copy";
import { humanize } from "@/lib/format";
import { errorMessage, useLoad } from "@/lib/useLoad";
import { Empty, Field, Panel, SyntheticBanner, stateTone } from "./shared";

/**
 * The case workspace: a conversation on the left, the governed state on the
 * right.
 *
 * It renders decisions the backend made — the onboarding's own status and
 * checklist, the practice run's stage, the controls, the readiness criteria and
 * the decision cards — and never computes them. It also deliberately does NOT
 * reproduce the onboarding wizard, the OCC's execution screens or its approval
 * screens: where one of those is the right place to look, the case links to it.
 */

type AgentStep = Parameters<ReturnType<typeof useOpsClient>["runAgentStep"]>[1];

/** The governed steps offered beside the conversation, in the order they arise. */
const STEPS: { action: string; step: AgentStep; label: string }[] = [
  {
    action: "request_client_information",
    step: "information-requests",
    label: "Ask the client for what is outstanding",
  },
  { action: "submit_for_approval", step: "submit", label: "Submit for approval" },
  { action: "approve_onboarding", step: "approve", label: "Approve the onboarding" },
  { action: "run_synthetic_onboarding", step: "run", label: "Run the practice onboarding" },
  { action: "generate_orchestration_plan", step: "plan", label: "Prepare the execution plan" },
  { action: "approve_execution_readiness", step: "readiness/approve", label: "Approve readiness" },
  { action: "request_activation", step: "review", label: "Submit for review" },
];

/** Which onboarding actions are worth a button, given where the case is. */
function onboardingActions(status: AgentStatus): string[] {
  const out: string[] = [];
  const onboarding = status.onboarding;
  if (onboarding.client_checklist.length > 0) out.push("request_client_information");
  if (onboarding.ready && ["draft", "in_review", "changes_required"].includes(onboarding.status)) {
    out.push("submit_for_approval");
  }
  if (onboarding.ready && ["ready_for_approval", "in_review"].includes(onboarding.status)) {
    out.push("approve_onboarding");
  }
  return out;
}

export function AgentCaseScreen() {
  const { caseId = "" } = useParams();
  const client = useOpsClient();
  const toast = useToast();
  const [text, setText] = useState("");
  const [busy, setBusy] = useState(false);
  const [proposal, setProposal] = useState<AgentProposal | null>(null);
  const [showPackage, setShowPackage] = useState(false);
  const [showPreview, setShowPreview] = useState(false);

  const view = useLoad<AgentStatus>(() => client.getAgentCase(caseId), [caseId]);
  const version = view.data?.run.version;
  const readiness = useLoad(() => client.getAgentReadiness(caseId), [caseId, version]);
  const preview = useLoad(() => client.getAgentPreview(caseId), [caseId, version]);

  async function act<T>(run: () => Promise<T>): Promise<T | undefined> {
    if (busy) return undefined;
    setBusy(true);
    try {
      const result = await run();
      await view.reload({ quiet: true });
      return result;
    } catch (err) {
      toast.show(errorMessage(err), "error");
      return undefined;
    } finally {
      setBusy(false);
    }
  }

  async function send(confirm = false) {
    const message = confirm ? (proposal?.summary ?? text) : text;
    if (!message.trim()) return;
    const turn = await act(() => client.instructAgent(caseId, message.trim(), confirm));
    if (!turn) return;
    setProposal(turn.proposal);
    if (turn.applied || !turn.proposal) setText("");
  }

  if (view.loading && !view.data) return <Loading />;
  if (view.error) {
    return (
      <Page title={copy.agent.title}>
        <ErrorNote message={view.error} onRetry={() => void view.reload()} />
      </Page>
    );
  }
  if (!view.data) {
    return (
      <Page title={copy.agent.title}>
        <p className="text-sm text-stone-500">{copy.agent.notFound}</p>
      </Page>
    );
  }

  const status = view.data;
  const run = status.run;
  const onboarding = status.onboarding;
  const facts = status.facts;
  // The rehearsal's verdict, not the run's current position: READY_FOR_EXECUTION
  // is a waypoint, and a case that has gone on to review still passed it.
  const isReady = run.readiness_status === "READY_FOR_EXECUTION";
  const available = new Set([
    ...(status.state.allowed_human_actions ?? []),
    ...onboardingActions(status),
  ]);

  return (
    <Page
      title={onboarding.client_name || status.case_ref}
      subtitle={[status.case_ref, facts.portfolio_id, humanize(facts.asset_class)]
        .filter(Boolean)
        .join(" · ")}
      actions={
        <Link
          to="/agent"
          className="inline-flex items-center gap-1 text-sm font-medium text-stone-600 hover:text-stone-900"
        >
          <ArrowLeft className="h-4 w-4" aria-hidden />
          {copy.agent.casesHeading}
        </Link>
      }
    >
      <SyntheticBanner />

      {isReady && <ReadyBanner />}

      <div className="mt-6 grid gap-6 lg:grid-cols-[minmax(0,1fr)_22rem]">
        <div className="space-y-6">
          <Panel title={copy.agent.conversationHeading}>
            <ol className="space-y-3">
              {run.messages.map((message, index) => (
                <li
                  key={`${message.at}-${index}`}
                  className={clsx(
                    "rounded-xl px-3 py-2 text-sm whitespace-pre-wrap",
                    message.role === "operator"
                      ? "bg-stone-100 text-stone-800"
                      : "border border-stone-200 bg-white text-stone-700",
                  )}
                >
                  {message.text}
                </li>
              ))}
            </ol>

            {proposal && (
              <div className="mt-4 rounded-xl border border-amber-200 bg-amber-50 p-3">
                <p className="text-sm font-semibold text-amber-900">{copy.agent.proposalHeading}</p>
                <p className="mt-1 text-sm text-amber-900">{proposal.summary}</p>
                {proposal.disclosure && <Disclosure disclosure={proposal.disclosure} />}
                <div className="mt-3 flex gap-2">
                  <button
                    type="button"
                    disabled={busy}
                    onClick={() => void send(true)}
                    className="rounded-lg bg-stone-900 px-3 py-1.5 text-sm font-semibold text-white disabled:opacity-50"
                  >
                    {copy.agent.proposalConfirm}
                  </button>
                  <button
                    type="button"
                    onClick={() => setProposal(null)}
                    className="rounded-lg border border-stone-300 bg-white px-3 py-1.5 text-sm font-medium text-stone-700"
                  >
                    {copy.agent.proposalDismiss}
                  </button>
                </div>
              </div>
            )}

            <div className="mt-4 flex gap-2">
              <input
                aria-label={copy.agent.conversationHeading}
                className="min-w-0 flex-1 rounded-xl border border-stone-300 px-3 py-2 text-sm"
                placeholder={copy.agent.conversationPlaceholder}
                value={text}
                disabled={busy}
                onChange={(event) => setText(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === "Enter") void send();
                }}
              />
              <button
                type="button"
                disabled={busy || !text.trim()}
                onClick={() => void send()}
                className="rounded-xl bg-blue-600 px-4 py-2 text-sm font-semibold text-white disabled:opacity-50"
              >
                {busy ? copy.agent.sending : copy.agent.send}
              </button>
            </div>
          </Panel>

          {status.open_decisions.filter((d) => d.status === "open").length > 0 && (
            <Panel title={copy.agent.decisionsHeading}>
              <ul className="space-y-4">
                {status.open_decisions
                  .filter((decision) => decision.status === "open")
                  .map((decision) => (
                    <DecisionCardView
                      key={decision.decision_id}
                      decision={decision}
                      busy={busy}
                      onAnswer={(action, value) =>
                        void act(() =>
                          client.answerAgentDecision(caseId, {
                            decision_id: decision.decision_id,
                            action,
                            value,
                          }),
                        )
                      }
                    />
                  ))}
              </ul>
            </Panel>
          )}

          <ClientPanel
            checklist={onboarding.client_checklist}
            requests={onboarding.information_requests}
            busy={busy}
            onAsk={() => void act(() => client.runAgentStep(caseId, "information-requests"))}
          />

          <Panel title={copy.agent.artefactsHeading}>
            {run.received_artefacts.length === 0 ? (
              <>
                <Empty />
                <p className="mt-2 text-sm text-stone-500">{copy.agent.uploadHelp}</p>
                <div className="mt-3 flex flex-wrap gap-2">
                  <label className="cursor-pointer rounded-xl border border-stone-300 px-3 py-1.5 text-sm font-medium text-stone-700 hover:bg-stone-50">
                    {copy.agent.uploadButton}
                    <input
                      type="file"
                      multiple
                      className="hidden"
                      onChange={(event) => {
                        const files = Array.from(event.target.files ?? []);
                        if (files.length > 0) {
                          void act(() => client.uploadAgentArtefacts(caseId, files));
                        }
                      }}
                    />
                  </label>
                  <button
                    type="button"
                    disabled={busy}
                    onClick={() => void act(() => client.generateAgentResponse(caseId))}
                    className="rounded-xl border border-stone-300 px-3 py-1.5 text-sm font-medium text-stone-700 hover:bg-stone-50 disabled:opacity-50"
                  >
                    {copy.agent.uploadGenerate}
                  </button>
                  {run.fixture_id && (
                    <button
                      type="button"
                      disabled={busy}
                      onClick={() =>
                        void act(() => client.loadAgentFixtureArtefacts(caseId, run.fixture_id))
                      }
                      className="rounded-xl border border-stone-300 px-3 py-1.5 text-sm font-medium text-stone-700 hover:bg-stone-50 disabled:opacity-50"
                    >
                      {copy.agent.uploadFixture}
                    </button>
                  )}
                </div>
              </>
            ) : (
              <ul className="space-y-3">
                {run.received_artefacts.map((artefact) => (
                  <li
                    key={artefact.artefact_id}
                    className="rounded-xl border border-stone-200 px-3 py-2"
                  >
                    <p className="text-sm font-medium text-stone-900">{artefact.source_file}</p>
                    <p className="text-xs text-stone-500">
                      {artefact.artefact_type
                        ? humanize(artefact.artefact_type)
                        : copy.workflow.fileKind}
                      {artefact.row_count > 0 && ` · ${artefact.row_count} records`}
                    </p>
                    <p className="mt-1 break-all text-xs text-stone-400">
                      {copy.agent.artefactIntended}: {artefact.intended_live_uri || "—"}
                    </p>
                    <p className="text-xs font-medium text-violet-700">
                      {copy.agent.artefactNotWritten}
                    </p>
                  </li>
                ))}
              </ul>
            )}
          </Panel>

          <PackPanel
            status={status}
            busy={busy}
            caseId={caseId}
            onDraft={() => void act(() => client.draftAgentPack(caseId))}
            onApprove={() => void act(() => client.approveAgentPack(caseId))}
            onSend={(to) => void act(() => client.sendAgentPack(caseId, to))}
          />

          <ReviewPanel
            status={status}
            busy={busy}
            caseId={caseId}
            onSubmit={() => void act(() => client.requestAgentReview(caseId))}
            onApprove={() => void act(() => client.approveAgentActivation(caseId))}
          />

          <ActivationPanel
            status={status}
            busy={busy}
            caseId={caseId}
            onConfirm={(confirmation) =>
              void act(() => client.confirmAgentActivation(caseId, confirmation))
            }
          />

          <Panel
            title={copy.agent.previewHeading}
            action={
              <button
                type="button"
                onClick={() => setShowPreview((prev) => !prev)}
                className="text-sm font-medium text-blue-700"
              >
                {showPreview ? copy.agent.hidePackage : copy.agent.downloadPackage}
              </button>
            }
          >
            <p className="text-sm text-stone-600">{copy.agent.previewDescription}</p>
            <p className="mt-1 text-xs font-medium text-violet-700">
              {copy.agent.previewNothingWritten}
            </p>
            {(preview.data?.preview.artefacts ?? []).length === 0 ? (
              <p className="mt-3 text-sm text-stone-400">{copy.agent.previewNone}</p>
            ) : showPreview ? (
              <pre className="mt-3 max-h-96 overflow-auto rounded-xl bg-stone-900 p-3 text-xs text-stone-100">
                {JSON.stringify(preview.data?.preview, null, 2)}
              </pre>
            ) : (
              <ul className="mt-3 space-y-1 text-sm text-stone-700">
                {(preview.data?.preview.artefacts ?? []).map((artefact) => (
                  <li key={artefact.rel}>{artefact.label}</li>
                ))}
              </ul>
            )}
          </Panel>

          {isReady && (
            <Panel
              title={copy.agent.readinessHeading}
              action={
                <button
                  type="button"
                  onClick={() => setShowPackage((prev) => !prev)}
                  className="text-sm font-medium text-blue-700"
                >
                  {showPackage ? copy.agent.hidePackage : copy.agent.downloadPackage}
                </button>
              }
            >
              {showPackage && readiness.data?.package ? (
                <pre className="max-h-96 overflow-auto rounded-xl bg-stone-900 p-3 text-xs text-stone-100">
                  {JSON.stringify(readiness.data.package, null, 2)}
                </pre>
              ) : (
                <p className="text-sm text-stone-600">{copy.agent.readyHeadline}</p>
              )}
            </Panel>
          )}
        </div>

        <aside className="space-y-6">
          <Panel title={copy.agent.statusHeading}>
            <Field label={copy.agent.onboardingStageHeading}>
              <StatusChip
                status={onboarding.status === "approved" ? "ready" : "waiting"}
                label={onboarding.status_label}
              />
            </Field>
            <Field label={copy.agent.stageHeading}>
              <StatusChip status={stateTone(run.state)} label={status.state.label} />
            </Field>
            <Field label={copy.agent.readinessHeading}>
              {status.readiness.ready ? copy.agent.readyStatus : copy.agent.notReady}
            </Field>
          </Panel>

          <Panel title={copy.agent.factsHeading}>
            {facts.client_id && <Field label="Client">{facts.client_id}</Field>}
            {facts.portfolio_id && <Field label="Portfolio">{facts.portfolio_id}</Field>}
            {facts.products.length > 0 && (
              <Field label="Products">
                {(facts.product_labels.length > 0 ? facts.product_labels : facts.products).join(
                  ", ",
                )}
              </Field>
            )}
            {run.reporting_period && (
              <Field label="Reporting period">{run.reporting_period}</Field>
            )}
          </Panel>

          <Panel title={copy.agent.gatesHeading}>
            <ol className="space-y-1">
              {status.lifecycle
                .filter((entry) => !["BLOCKED", "CANCELLED"].includes(entry.state))
                .map((entry) => (
                  <li
                    key={entry.state}
                    data-state={entry.state}
                    data-current={entry.current ? "true" : "false"}
                    className={clsx(
                      "flex items-center gap-2 rounded-lg px-2 py-1 text-sm",
                      entry.current && "bg-stone-100 font-medium text-stone-900",
                      !entry.current && entry.reached && "text-stone-600",
                      !entry.current && !entry.reached && "text-stone-400",
                    )}
                  >
                    {entry.reached && !entry.current ? (
                      <CheckCircle2 className="h-4 w-4 text-emerald-500" aria-hidden />
                    ) : (
                      <CircleDot
                        className={clsx(
                          "h-4 w-4",
                          entry.current ? "text-blue-500" : "text-stone-300",
                        )}
                        aria-hidden
                      />
                    )}
                    <span>{entry.label}</span>
                  </li>
                ))}
            </ol>
          </Panel>

          {Object.keys(status.stage_outcomes).length > 0 && (
            <Panel title={copy.agent.executionHeading}>
              <ul className="space-y-1">
                {Object.entries(status.stage_outcomes).map(([stage, outcome]) => (
                  <li key={stage} className="flex items-center justify-between gap-2 text-sm">
                    <span className="text-stone-600">{humanize(stage)}</span>
                    <span className="text-xs font-medium text-stone-500">
                      {copy.agent.stageOutcomes[outcome] ?? humanize(outcome)}
                    </span>
                  </li>
                ))}
              </ul>
            </Panel>
          )}

          <Panel title={copy.agent.criteriaHeading}>
            <CriteriaList criteria={status.readiness.criteria} />
          </Panel>

          {status.blockers.length > 0 && (
            <Panel title={copy.agent.blockersHeading}>
              <ul className="list-disc space-y-1 pl-4 text-sm text-rose-700">
                {status.blockers.map((blocker) => (
                  <li key={blocker}>{blocker}</li>
                ))}
              </ul>
            </Panel>
          )}

          {onboarding.blocking.length > 0 && (
            <Panel title={copy.agent.missingHeading}>
              <ul className="list-disc space-y-1 pl-4 text-sm text-stone-600">
                {onboarding.blocking.map((problem) => (
                  <li key={`${problem.section}-${problem.field}-${problem.index}`}>
                    {problem.message}
                  </li>
                ))}
              </ul>
            </Panel>
          )}

          {status.observations.length > 0 && (
            <Panel title={copy.agent.observationsHeading}>
              <ul className="list-disc space-y-1 pl-4 text-sm text-stone-600">
                {status.observations.map((observation) => (
                  <li key={observation}>{observation}</li>
                ))}
              </ul>
            </Panel>
          )}

          <Panel title={copy.agent.actionsHeading}>
            <ControlActions
              available={available}
              busy={busy}
              onRun={(step) => void act(() => client.runAgentStep(caseId, step))}
            />
          </Panel>

          <Panel title={copy.agent.occLinksHeading}>
            <ul className="space-y-2">
              {status.occ_links.map((link) => (
                <li key={link.to}>
                  <Link
                    to={link.to}
                    className="inline-flex items-center gap-1 text-sm font-medium text-blue-700 hover:underline"
                  >
                    {link.label}
                    <ExternalLink className="h-3 w-3" aria-hidden />
                  </Link>
                  <p className="text-xs text-stone-500">{link.why}</p>
                </li>
              ))}
            </ul>
          </Panel>
        </aside>
      </div>
    </Page>
  );
}

/**
 * The four populations a turn must report.
 *
 * Rendered together and always in the same order, so "and what did you NOT
 * understand?" has one place to look. A proposal carrying questions or
 * unrecognised text has applied NOTHING — the panel says so rather than
 * leaving it to be inferred from a missing tick.
 */
function Disclosure({ disclosure }: { disclosure: AgentDisclosure }) {
  const incomplete = disclosure.questions.length > 0 || disclosure.unrecognised.length > 0;
  return (
    <div className="mt-3 space-y-2 text-sm text-amber-900">
      {disclosure.understood.length > 0 && (
        <div>
          <p className="text-xs font-semibold uppercase tracking-wide text-amber-700">
            {copy.agent.disclosureUnderstood}
          </p>
          <ul className="mt-1 list-disc space-y-0.5 pl-4">
            {disclosure.understood.map((line) => (
              <li key={line}>{line}</li>
            ))}
          </ul>
        </div>
      )}
      {disclosure.questions.length > 0 && (
        <div>
          <p className="text-xs font-semibold uppercase tracking-wide text-amber-700">
            {copy.agent.disclosureQuestions}
          </p>
          <ul className="mt-1 list-disc space-y-0.5 pl-4">
            {disclosure.questions.map((line) => (
              <li key={line}>{line}</li>
            ))}
          </ul>
        </div>
      )}
      {disclosure.unrecognised.length > 0 && (
        <div>
          <p className="text-xs font-semibold uppercase tracking-wide text-rose-700">
            {copy.agent.disclosureUnrecognised}
          </p>
          <ul className="mt-1 list-disc space-y-0.5 pl-4 text-rose-800">
            {disclosure.unrecognised.map((line) => (
              <li key={line}>“{line}”</li>
            ))}
          </ul>
        </div>
      )}
      {incomplete && (
        <p className="font-medium text-rose-800">{copy.agent.disclosureNothingApplied}</p>
      )}
    </div>
  );
}

/**
 * The client pack, and its own four-state workflow.
 *
 * Every question in it is a field the governed catalogue declares — this panel
 * projects what the server built and adds nothing. The receipt's `sent` is
 * reported as it stands: in an environment with no mail integration it is
 * false, and the panel says "recorded, not sent" rather than implying delivery.
 */
function PackPanel({
  status,
  busy,
  caseId,
  onDraft,
  onApprove,
  onSend,
}: {
  status: AgentStatus;
  busy: boolean;
  caseId: string;
  onDraft: () => void;
  onApprove: () => void;
  onSend: (to?: string[]) => void;
}) {
  const client = useOpsClient();
  const [showDocument, setShowDocument] = useState(false);
  const [recipient, setRecipient] = useState("");
  const pack = status.pack;
  const allowed = new Set(status.state.allowed_human_actions ?? []);
  const document = useLoad(() => client.getAgentPack(caseId), [caseId, status.run.version]);
  const recipients = pack.email?.to ?? [];

  return (
    <Panel
      title={copy.agent.packHeading}
      action={
        pack.sections.length > 0 ? (
          <button
            type="button"
            onClick={() => setShowDocument((prev) => !prev)}
            className="text-sm font-medium text-blue-700"
          >
            {showDocument ? copy.agent.packHide : copy.agent.packDocument}
          </button>
        ) : undefined
      }
    >
      <p className="text-sm text-stone-600">{copy.agent.packDescription}</p>

      {pack.sections.length === 0 ? (
        <p className="mt-3 text-sm text-stone-400">{copy.agent.packNone}</p>
      ) : (
        <>
          <p className="mt-3 text-sm text-stone-700">
            {pack.questions - pack.outstanding} {copy.agent.packAnswered} ·{" "}
            <span className="font-medium">{pack.outstanding}</span> {copy.agent.packOutstanding}
          </p>
          <ul className="mt-2 space-y-1 text-sm text-stone-600">
            {pack.sections.map((section) => (
              <li key={section.key} className="flex justify-between gap-2">
                <span>{section.label}</span>
                <span className="text-xs text-stone-500">
                  {section.outstanding} {copy.agent.packOutstanding}
                </span>
              </li>
            ))}
          </ul>
          <p className="mt-3 text-xs text-stone-500">{pack.mapping_statement}</p>
          {showDocument && (
            <pre className="mt-3 max-h-96 overflow-auto rounded-xl bg-stone-900 p-3 text-xs text-stone-100">
              {document.data?.document ?? ""}
            </pre>
          )}
        </>
      )}

      {pack.status && (
        <p className="mt-3 text-xs uppercase tracking-wide text-stone-400">
          {copy.agent.packStatusHeading}: {humanize(pack.status)}
        </p>
      )}

      {/* The honest answer to "did this leave Trakt?". */}
      {pack.status === "SENT" && (
        <p className="mt-2 rounded-lg bg-violet-50 px-3 py-2 text-sm font-medium text-violet-800">
          {pack.sent ? copy.agent.packIssued : copy.agent.packNotSent}
        </p>
      )}

      <div className="mt-3 flex flex-wrap items-center gap-2">
        {allowed.has("draft_onboarding_pack") && (
          <button
            type="button"
            disabled={busy}
            onClick={onDraft}
            className="rounded-xl border border-stone-300 px-3 py-1.5 text-sm font-medium text-stone-700 hover:bg-stone-50 disabled:opacity-50"
          >
            {pack.sections.length > 0 ? copy.agent.packRedraft : copy.agent.packDraft}
          </button>
        )}
        {allowed.has("approve_pack_to_send") && (
          <button
            type="button"
            disabled={busy}
            onClick={onApprove}
            className="rounded-xl bg-stone-900 px-3 py-1.5 text-sm font-semibold text-white disabled:opacity-50"
          >
            {copy.agent.packApprove}
          </button>
        )}
        {allowed.has("send_onboarding_pack") && (
          <>
            {recipients.length === 0 && (
              <input
                aria-label={copy.agent.packRecipients}
                className="min-w-0 flex-1 rounded-xl border border-stone-300 px-3 py-1.5 text-sm"
                placeholder={copy.agent.packNoRecipient}
                value={recipient}
                onChange={(event) => setRecipient(event.target.value)}
              />
            )}
            <button
              type="button"
              disabled={busy || (recipients.length === 0 && !recipient.trim())}
              onClick={() => onSend(recipients.length > 0 ? undefined : [recipient.trim()])}
              className="rounded-xl bg-stone-900 px-3 py-1.5 text-sm font-semibold text-white disabled:opacity-50"
            >
              {copy.agent.packSend}
            </button>
          </>
        )}
      </div>
      {recipients.length > 0 && (
        <p className="mt-2 text-xs text-stone-500">
          {copy.agent.packRecipients}: {recipients.join(", ")}
        </p>
      )}
    </Panel>
  );
}

/**
 * The review package, and the approval of the configuration.
 *
 * Approving records a decision and prepares the confirmation. It does not start
 * anything, and the panel says so beside the button rather than after the fact.
 */
function ReviewPanel({
  status,
  busy,
  caseId,
  onSubmit,
  onApprove,
}: {
  status: AgentStatus;
  busy: boolean;
  caseId: string;
  onSubmit: () => void;
  onApprove: () => void;
}) {
  const client = useOpsClient();
  const [show, setShow] = useState(false);
  const allowed = new Set(status.state.allowed_human_actions ?? []);
  const submitted = Boolean(status.review_package_ref);
  const review = useLoad(
    () => (submitted ? client.getAgentReview(caseId) : Promise.resolve(null)),
    [caseId, submitted, status.run.version],
  );
  const pkg = review.data?.package as
    | { operator_actions?: { kind: string; subject: string; detail: string; status: string }[] }
    | undefined;

  if (!submitted && !allowed.has("request_activation")) return null;

  return (
    <Panel
      title={copy.agent.reviewHeading}
      action={
        submitted ? (
          <button
            type="button"
            onClick={() => setShow((prev) => !prev)}
            className="text-sm font-medium text-blue-700"
          >
            {show ? copy.agent.packHide : copy.agent.reviewShow}
          </button>
        ) : undefined
      }
    >
      <p className="text-sm text-stone-600">{copy.agent.reviewDescription}</p>

      {!submitted ? (
        <p className="mt-3 text-sm text-stone-400">{copy.agent.reviewNone}</p>
      ) : (
        <>
          {(pkg?.operator_actions ?? []).length > 0 && (
            <div className="mt-3">
              <p className="text-xs uppercase tracking-wide text-stone-400">
                {copy.agent.reviewOperatorActions}
              </p>
              <ul className="mt-1 space-y-1 text-sm text-stone-700">
                {(pkg?.operator_actions ?? []).map((action) => (
                  <li key={`${action.kind}-${action.subject}`}>
                    <span className="font-medium">{action.subject}</span> — {action.detail}{" "}
                    <span className="text-xs font-medium text-amber-700">
                      {copy.agent.reviewNotProvisioned}
                    </span>
                  </li>
                ))}
              </ul>
            </div>
          )}
          {show && (
            <pre className="mt-3 max-h-96 overflow-auto rounded-xl bg-stone-900 p-3 text-xs text-stone-100">
              {review.data?.document ?? ""}
            </pre>
          )}
        </>
      )}

      <div className="mt-3 flex flex-wrap items-center gap-2">
        {allowed.has("request_activation") && (
          <button
            type="button"
            disabled={busy}
            onClick={onSubmit}
            className="rounded-xl border border-stone-300 px-3 py-1.5 text-sm font-medium text-stone-700 hover:bg-stone-50 disabled:opacity-50"
          >
            {copy.agent.reviewSubmit}
          </button>
        )}
        {allowed.has("approve_activation") && (
          <>
            <button
              type="button"
              disabled={busy}
              onClick={onApprove}
              className="rounded-xl bg-stone-900 px-3 py-1.5 text-sm font-semibold text-white disabled:opacity-50"
            >
              {copy.agent.reviewApprove}
            </button>
            <span className="text-xs text-stone-500">{copy.agent.reviewApproveNote}</span>
          </>
        )}
      </div>
    </Panel>
  );
}

/**
 * The confirmation immediately before production.
 *
 * Deliberately concrete: the client, the portfolio, the files, where they would
 * go, and what would happen. "Activate this client" is not a decision anyone can
 * make on its own. Every reason the server would refuse is listed, so an
 * operator is never left guessing why the button did nothing.
 */
function ActivationPanel({
  status,
  busy,
  caseId,
  onConfirm,
}: {
  status: AgentStatus;
  busy: boolean;
  caseId: string;
  onConfirm: (confirmation: string) => void;
}) {
  const client = useOpsClient();
  const [confirmation, setConfirmation] = useState("");
  const allowed = new Set(status.state.allowed_human_actions ?? []);
  const relevant = allowed.has("confirm_activation") || status.run.state === "INGESTION_STARTED";
  const view = useLoad(
    () => (relevant ? client.getAgentActivation(caseId) : Promise.resolve(null)),
    [caseId, relevant, status.run.version],
  );

  if (!relevant) return null;
  const intent = view.data?.intent;
  const refusals = view.data?.refusals ?? [];

  return (
    <Panel title={copy.agent.activationHeading}>
      <p className="text-sm text-stone-600">{copy.agent.activationDescription}</p>

      {intent && (
        <>
          <p className="mt-3 text-sm font-medium text-stone-900">{intent.statement}</p>
          {intent.files.length > 0 && (
            <div className="mt-3">
              <p className="text-xs uppercase tracking-wide text-stone-400">
                {copy.agent.activationFiles}
              </p>
              <ul className="mt-1 space-y-0.5 text-sm text-stone-700">
                {intent.files.map((file) => (
                  <li key={file.name}>{file.name}</li>
                ))}
              </ul>
            </div>
          )}
          {intent.target_locations.length > 0 && (
            <div className="mt-3">
              <p className="text-xs uppercase tracking-wide text-stone-400">
                {copy.agent.activationTargets}
              </p>
              <ul className="mt-1 space-y-0.5 break-all text-xs text-stone-500">
                {intent.target_locations.map((target) => (
                  <li key={target}>{target}</li>
                ))}
              </ul>
            </div>
          )}
          <div className="mt-3">
            <p className="text-xs uppercase tracking-wide text-stone-400">
              {copy.agent.activationActions}
            </p>
            <ul className="mt-1 list-disc space-y-0.5 pl-4 text-sm text-stone-700">
              {intent.actions.map((action) => (
                <li key={action}>{action}</li>
              ))}
            </ul>
          </div>
        </>
      )}

      {status.run.state === "INGESTION_STARTED" ? (
        <p className="mt-3 rounded-lg bg-emerald-50 px-3 py-2 text-sm font-medium text-emerald-800">
          {copy.agent.activationStarted}
        </p>
      ) : (
        <>
          {refusals.length > 0 && (
            <div className="mt-4 rounded-xl border border-amber-200 bg-amber-50 p-3">
              <p className="text-sm font-semibold text-amber-900">
                {copy.agent.activationRefusedHeading}
              </p>
              <ul className="mt-1 list-disc space-y-0.5 pl-4 text-sm text-amber-900">
                {refusals.map((reason) => (
                  <li key={reason}>{reason}</li>
                ))}
              </ul>
            </div>
          )}
          {!view.data?.live_enabled && (
            <p className="mt-3 text-sm font-medium text-violet-800">
              {copy.agent.activationDisabled}
            </p>
          )}
          <div className="mt-3 flex flex-wrap items-center gap-2">
            <input
              aria-label={copy.agent.activationConfirmLabel}
              className="min-w-0 flex-1 rounded-xl border border-stone-300 px-3 py-1.5 text-sm"
              placeholder={copy.agent.activationConfirmLabel}
              value={confirmation}
              onChange={(event) => setConfirmation(event.target.value)}
            />
            <button
              type="button"
              disabled={busy || !confirmation.trim()}
              onClick={() => onConfirm(confirmation.trim())}
              className="rounded-xl bg-rose-700 px-3 py-1.5 text-sm font-semibold text-white disabled:opacity-50"
            >
              {copy.agent.activationConfirm}
            </button>
          </div>
        </>
      )}
    </Panel>
  );
}

/**
 * What the client still has to answer, and what has already been asked.
 *
 * The list is Client Onboarding's own — only fields a client can actually
 * answer — so this panel shows it rather than working anything out.
 */
function ClientPanel({
  checklist,
  requests,
  busy,
  onAsk,
}: {
  checklist: ChecklistRow[];
  requests: InformationRequest[];
  busy: boolean;
  onAsk: () => void;
}) {
  return (
    <Panel title={copy.agent.checklistHeading}>
      {checklist.length === 0 ? (
        <p className="text-sm text-stone-500">{copy.agent.checklistEmpty}</p>
      ) : (
        <>
          <ul className="list-disc space-y-1 pl-4 text-sm text-stone-700">
            {checklist.map((row) => (
              <li key={`${row.section}-${row.field}-${row.index}`}>{row.label}</li>
            ))}
          </ul>
          <button
            type="button"
            disabled={busy}
            onClick={onAsk}
            className="mt-3 rounded-xl border border-stone-300 px-3 py-1.5 text-sm font-medium text-stone-700 hover:bg-stone-50 disabled:opacity-50"
          >
            {copy.agent.checklistAsk}
          </button>
        </>
      )}

      {requests.length > 0 && (
        <>
          <p className="mt-4 text-xs uppercase tracking-wide text-stone-400">
            {copy.agent.requestsHeading}
          </p>
          <ul className="mt-1 space-y-1 text-sm text-stone-600">
            {requests.map((request) => (
              <li key={request.request_id} className="flex justify-between gap-2">
                <span>
                  {request.items.length} item{request.items.length === 1 ? "" : "s"}
                </span>
                <span className="text-xs text-stone-500">
                  {["open", "sent"].includes(request.status)
                    ? copy.agent.requestOutstanding
                    : copy.agent.requestAnswered}
                </span>
              </li>
            ))}
          </ul>
        </>
      )}
    </Panel>
  );
}

/** Readiness criteria, grouped by which half of the process they belong to. */
function CriteriaList({ criteria }: { criteria: ReadinessCriterion[] }) {
  const groups: { key: ReadinessCriterion["stage"]; label: string }[] = [
    { key: "onboarding", label: copy.agent.criteriaOnboarding },
    { key: "execution", label: copy.agent.criteriaExecution },
    { key: "boundary", label: copy.agent.criteriaBoundary },
  ];
  return (
    <>
      {groups.map((group) => {
        const rows = criteria.filter((c) => c.stage === group.key);
        if (rows.length === 0) return null;
        return (
          <div key={group.key} className="mb-3 last:mb-0">
            <p className="mb-1 text-xs uppercase tracking-wide text-stone-400">{group.label}</p>
            <ul className="space-y-1">
              {rows.map((criterion) => (
                <li key={criterion.key} className="flex items-start justify-between gap-2 text-sm">
                  <span className={criterion.passed ? "text-stone-600" : "text-stone-900"}>
                    {criterion.label}
                  </span>
                  <span
                    className={clsx(
                      "shrink-0 text-xs font-medium",
                      criterion.passed ? "text-emerald-600" : "text-amber-700",
                    )}
                  >
                    {criterion.passed ? copy.agent.gateDone : copy.agent.gateBlocked}
                  </span>
                </li>
              ))}
            </ul>
          </div>
        );
      })}
    </>
  );
}

/**
 * The governed steps available from here.
 *
 * Some allowed actions — answering a question, resolving a mapping — need a
 * detail that no button can carry, so they are reachable only through the
 * conversation. Saying "nothing yet" in that case would be wrong: there IS
 * something to do. The panel distinguishes the two.
 */
function ControlActions({
  available,
  busy,
  onRun,
}: {
  available: Set<string>;
  busy: boolean;
  onRun: (step: AgentStep) => void;
}) {
  const buttons = STEPS.filter((entry) => available.has(entry.action));
  const conversational = [...available].filter(
    (action) => !STEPS.some((entry) => entry.action === action),
  );

  if (buttons.length === 0 && conversational.length === 0) {
    return <p className="text-sm text-stone-400">{copy.agent.actionsNone}</p>;
  }

  return (
    <>
      {buttons.length > 0 && (
        <>
          <p className="mb-2 text-xs text-stone-500">{copy.agent.actionsHelp}</p>
          <div className="flex flex-wrap gap-2">
            {buttons.map((entry) => (
              <button
                key={entry.step}
                type="button"
                disabled={busy}
                onClick={() => onRun(entry.step)}
                className="rounded-lg border border-stone-300 px-3 py-1.5 text-sm font-medium text-stone-700 hover:bg-stone-50 disabled:opacity-50"
              >
                {entry.label}
              </button>
            ))}
          </div>
        </>
      )}
      {conversational.length > 0 && (
        <>
          <p className={clsx("text-xs text-stone-500", buttons.length > 0 && "mt-3")}>
            {copy.agent.actionsInConversation}
          </p>
          <ul className="mt-1 list-disc pl-4 text-sm text-stone-600">
            {conversational.map((action) => (
              <li key={action}>{humanize(action)}</li>
            ))}
          </ul>
        </>
      )}
    </>
  );
}

function ReadyBanner() {
  return (
    <div
      role="status"
      className="mt-4 rounded-2xl border border-emerald-200 bg-emerald-50 px-4 py-3"
    >
      <p className="text-sm font-semibold text-emerald-900">{copy.agent.readyHeadline}</p>
      <ul className="mt-1 space-y-0.5 text-sm text-emerald-800">
        {copy.agent.readyNotDone.map((line) => (
          <li key={line}>{line}</li>
        ))}
      </ul>
    </div>
  );
}

function DecisionCardView({
  decision,
  busy,
  onAnswer,
}: {
  decision: DecisionCard;
  busy: boolean;
  onAnswer: (action: string, value: string) => void;
}) {
  return (
    <li className="rounded-xl border border-amber-200 bg-amber-50 p-4">
      <p className="text-sm font-semibold text-stone-900">{decision.title}</p>
      <p className="mt-1 text-sm text-stone-700">{decision.question}</p>

      <dl className="mt-3 space-y-1 text-sm">
        <Detail label={copy.agent.decisionIssue}>{decision.issue}</Detail>
        {decision.evidence.map((item, index) => (
          <Detail key={index} label={copy.agent.decisionEvidence}>
            {typeof item.data === "object" && item.data !== null
              ? Object.values(item.data as Record<string, unknown>)
                  .filter(Boolean)
                  .join(" ")
              : String(item.data ?? "")}
          </Detail>
        ))}
        {decision.recommendation && (
          <Detail label={copy.agent.decisionRecommendation}>{decision.recommendation}</Detail>
        )}
        {decision.confidence !== null && decision.confidence !== undefined && (
          <Detail label={copy.agent.decisionConfidence}>
            {Math.round(decision.confidence * 100)}%
          </Detail>
        )}
        <Detail label={copy.agent.decisionMateriality}>{decision.materiality}</Detail>
        <Detail label={copy.agent.decisionConsequence}>{decision.downstream_consequence}</Detail>
      </dl>

      <div className="mt-3 flex flex-wrap gap-2">
        {decision.options.map((option) => (
          <button
            key={option.value}
            type="button"
            disabled={busy}
            onClick={() => onAnswer("amend", option.value)}
            className="rounded-lg border border-stone-300 bg-white px-3 py-1.5 text-sm font-medium text-stone-700 disabled:opacity-50"
          >
            {option.label}
          </button>
        ))}
        <button
          type="button"
          disabled={busy}
          onClick={() => onAnswer("approve", decision.recommendation)}
          className="rounded-lg bg-stone-900 px-3 py-1.5 text-sm font-semibold text-white disabled:opacity-50"
        >
          {copy.agent.decisionApprove}
        </button>
      </div>
    </li>
  );
}

function Detail({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div>
      <dt className="text-xs uppercase tracking-wide text-stone-400">{label}</dt>
      <dd className="text-stone-700">{children}</dd>
    </div>
  );
}
