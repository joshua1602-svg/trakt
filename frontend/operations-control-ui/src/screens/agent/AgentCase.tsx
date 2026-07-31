import { useState } from "react";
import { Link, useParams } from "react-router-dom";
import { ArrowLeft, CheckCircle2, CircleDot, ExternalLink } from "lucide-react";
import clsx from "clsx";
import { useOpsClient } from "@/api/context";
import type {
  AgentProposal,
  AgentStatus,
  DecisionCard,
  ProposedValue,
} from "@/api/agentTypes";
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
 * It renders decisions the backend made — the stage, the controls, the readiness
 * criteria, the decision cards — and never computes them. It also deliberately
 * does NOT reproduce the OCC's execution, evidence or approval screens: where
 * one of those is the right place to look, the case links to it.
 */

/** The governed steps offered beside the conversation, in lifecycle order. */
const STEPS: { action: string; step: Parameters<
  ReturnType<typeof useOpsClient>["runAgentStep"]
>[1]; label: string }[] = [
  { action: "confirm_requirements", step: "requirements/confirm", label: "Confirm the interpretation" },
  { action: "generate_onboarding_pack", step: "pack/generate", label: "Generate the onboarding pack" },
  { action: "approve_onboarding_pack", step: "pack/approve", label: "Approve the pack" },
  { action: "classify_artefacts", step: "artefacts/classify", label: "Recognise the files" },
  { action: "draft_client_config", step: "configuration/draft", label: "Draft the configuration" },
  { action: "approve_client_config", step: "configuration/approve", label: "Approve the configuration" },
  { action: "run_synthetic_onboarding", step: "run", label: "Run the onboarding" },
  { action: "generate_orchestration_plan", step: "plan", label: "Prepare the execution plan" },
  { action: "approve_execution_readiness", step: "readiness/approve", label: "Approve readiness" },
];

export function AgentCaseScreen() {
  const { caseId = "" } = useParams();
  const client = useOpsClient();
  const toast = useToast();
  const [text, setText] = useState("");
  const [busy, setBusy] = useState(false);
  const [proposal, setProposal] = useState<AgentProposal | null>(null);
  const [showPackage, setShowPackage] = useState(false);

  const view = useLoad<AgentStatus>(() => client.getAgentCase(caseId), [caseId]);
  const readiness = useLoad(
    () => client.getAgentReadiness(caseId),
    [caseId, view.data?.case.case_version],
  );

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
    const message = confirm ? proposal?.summary ?? text : text;
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
  const doc = status.case;
  const isReady = doc.state === "READY_FOR_EXECUTION";
  const available = new Set(status.state.allowed_human_actions ?? []);

  return (
    <Page
      title={doc.client_name || doc.case_id}
      subtitle={[doc.portfolio_id, humanize(doc.asset_type)].filter(Boolean).join(" · ")}
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
              {doc.messages.map((message, index) => (
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
                <p className="text-sm font-semibold text-amber-900">
                  {copy.agent.proposalHeading}
                </p>
                <p className="mt-1 text-sm text-amber-900">{proposal.summary}</p>
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

          <Panel title={copy.agent.artefactsHeading}>
            {doc.received_artefacts.length === 0 ? (
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
                  {doc.fixture_id && (
                    <button
                      type="button"
                      disabled={busy}
                      onClick={() =>
                        void act(() =>
                          client.loadAgentFixtureArtefacts(caseId, doc.fixture_id),
                        )
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
                {doc.received_artefacts.map((artefact) => (
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

          {doc.configuration_provenance.length > 0 && (
            <Panel title={copy.agent.configHeading}>
              <p className="mb-2 text-xs uppercase tracking-wide text-stone-400">
                {copy.agent.configProvenance}
              </p>
              <ul className="space-y-2">
                {doc.configuration_provenance.map((value) => (
                  <ProvenanceRow key={value.key} value={value} />
                ))}
              </ul>
            </Panel>
          )}

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
            <Field label={copy.agent.stageHeading}>
              <StatusChip status={stateTone(doc.state)} label={status.state.label} />
            </Field>
            <Field label={copy.agent.readinessHeading}>
              {status.readiness.ready ? copy.agent.readyStatus : copy.agent.notReady}
            </Field>
            {doc.client_id && <Field label="Client">{doc.client_id}</Field>}
            {doc.portfolio_id && <Field label="Portfolio">{doc.portfolio_id}</Field>}
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
            <ul className="space-y-1">
              {status.readiness.criteria.map((criterion) => (
                <li
                  key={criterion.key}
                  className="flex items-start justify-between gap-2 text-sm"
                >
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

          {doc.unresolved_questions.length > 0 && (
            <Panel title={copy.agent.missingHeading}>
              <ul className="list-disc space-y-1 pl-4 text-sm text-stone-600">
                {doc.unresolved_questions.map((question) => (
                  <li key={question}>{question}</li>
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
 * The governed steps available from here.
 *
 * Some allowed actions — correcting a fact or a configuration value, answering
 * a mapping — need a detail that no button can carry, so they are reachable
 * only through the conversation. Saying "nothing yet" in that case would be
 * wrong: there IS something to do. The panel distinguishes the two.
 */
function ControlActions({
  available,
  busy,
  onRun,
}: {
  available: Set<string>;
  busy: boolean;
  onRun: (step: (typeof STEPS)[number]["step"]) => void;
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
        <Detail label={copy.agent.decisionConsequence}>
          {decision.downstream_consequence}
        </Detail>
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

function ProvenanceRow({ value }: { value: ProposedValue }) {
  return (
    <li className="rounded-xl border border-stone-200 px-3 py-2">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <span className="text-sm font-medium text-stone-900">{humanize(value.key)}</span>
        <span className="text-sm text-stone-700">{formatValue(value.value)}</span>
      </div>
      <p className="text-xs text-stone-500">
        {humanize(value.source)}
        {value.evidence && ` — ${value.evidence}`}
      </p>
      {value.downstream_impact && (
        <p className="text-xs text-stone-400">{value.downstream_impact}</p>
      )}
      {value.requires_human_confirmation && !value.confirmed && (
        <p className="text-xs font-medium text-amber-700">{copy.agent.configNeedsConfirm}</p>
      )}
    </li>
  );
}

function formatValue(value: unknown): string {
  if (value === null || value === undefined || value === "") return "—";
  if (Array.isArray(value)) return value.join(", ") || "—";
  if (typeof value === "boolean") return value ? "Yes" : "No";
  return String(value);
}
