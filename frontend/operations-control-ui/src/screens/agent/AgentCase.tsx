import { useCallback, useEffect, useState, type ReactNode } from "react";
import { Link, useParams } from "react-router-dom";
import {
  ArrowLeft,
  CheckCircle2,
  Circle,
  CircleDot,
  ExternalLink,
} from "lucide-react";
import clsx from "clsx";
import { useOpsClient } from "@/api/context";
import type {
  AgentDisclosure,
  AgentProposal,
  AgentStatus,
  DecisionCard,
  FieldRequest,
  MappingOverview,
  MappingRow,
  ReadinessCriterion,
  RegistryField,
  StreamSummary,
} from "@/api/agentTypes";
import type { CaseProblem, ChecklistRow, InformationRequest } from "@/api/onboardingTypes";
import { AgentCancelDialog } from "./AgentCancelDialog";
import { ClientMailPanel } from "./AgentClientMail";
import { ClientQuestionsPanel } from "./AgentClientQuestions";
import { DialogButtons, Modal } from "@/components/admin/primitives";
import { ErrorNote, Loading } from "@/components/ErrorNote";
import { Page } from "@/components/Page";
import { StatusChip } from "@/components/StatusChip";
import { useToast } from "@/components/Toast";
import { copy } from "@/lib/copy";
import { humanize } from "@/lib/format";
import { errorMessage, useLoad } from "@/lib/useLoad";
import { Empty, Field, Panel, SyntheticBanner, stateTone } from "./shared";
import { deriveStages, type StageInfo, type StageKey } from "./stages";

/**
 * The case workspace, rebuilt around chronology.
 *
 * The spine of the page is the operator journey — define, scope, pack, client
 * response, configuration, rehearsal, readiness, approval, activation — with
 * exactly one current stage expanded, completed stages collapsed into the
 * timeline, and future stages listed without their panels. The conversation
 * stays beside it at every stage.
 *
 * It still renders only decisions the backend made — the onboarding's own
 * status and checklist, the run's state, the readiness criteria, the decision
 * cards — and never computes them. Where an existing OCC view is the right
 * place to look, the case links to it rather than reproducing it.
 */

type AgentStep = Parameters<ReturnType<typeof useOpsClient>["runAgentStep"]>[1];

/**
 * Ending the case. Named once, because two places have to agree about it: the
 * foot-of-page control that offers it, and the conversational list that must
 * NOT, or the same act is offered twice — once as a real control with a
 * required reason, and once as a bullet reading "cancel run".
 */
const CANCEL_ACTION = "cancel_run";

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

/**
 * The blocking problems that are the OPERATOR'S to solve.
 *
 * `onboarding.blocking` and the client's outstanding list overlap almost
 * entirely — every unanswered client field is both — so rendering both in full
 * printed the same eleven items twice on one screen, once as a checklist and
 * once as sentences, and buried anything that was genuinely different. The
 * residue is the real answer to "why is this stuck": what nobody has asked the
 * client for, because it is not theirs to answer.
 *
 * The split is the SERVER'S, not this screen's: validation stamps every
 * problem with `owner`, from whether the catalogue asks that field of a client
 * at all. Deriving it here instead — "blocking items not in the checklist" —
 * would have been wrong in a way that matters, because `client_checklist`
 * excludes anything sitting in an open request, so pressing "ask the client"
 * would have emptied the checklist and tipped the client's own items into this
 * panel at the exact moment it became most true that we were waiting on them.
 * That is the bug already fixed once in the agent's `pending()`; reading
 * `owner` cannot reproduce it, because ownership does not depend on what has
 * been asked.
 */
export function operatorBlocking(blocking: CaseProblem[]): CaseProblem[] {
  return blocking.filter((problem) => problem.owner !== "client");
}

/** Which pack stage currently owns the pack panel, so it renders exactly once. */
function packOwner(status: AgentStatus): StageKey {
  const packStatus = status.pack.status ?? "";
  if (!packStatus || packStatus === "DRAFTED" || packStatus === "HUMAN_REVIEW_REQUIRED") {
    return (status.pack.sections ?? []).length > 0 ? "pack_review" : "pack_prepare";
  }
  if (packStatus === "APPROVED_TO_SEND") return "pack_issue";
  return "pack_issue"; // SENT — collapsed under the issue stage
}

export function AgentCaseScreen() {
  const { caseId = "" } = useParams();
  const client = useOpsClient();
  // Stable, because the dialog fetches on it. A fresh closure every render
  // makes the effect that loads the field list re-run on its own result, and
  // an input re-rendering under the cursor loses what is being typed into it.
  const loadFields = useCallback(
    () => client.agentFieldRegistry(caseId),
    [client, caseId],
  );
  const toast = useToast();
  const [text, setText] = useState("");
  const [busy, setBusy] = useState(false);
  const [proposal, setProposal] = useState<AgentProposal | null>(null);
  // The operator's own words that produced the standing proposal. Confirming
  // re-sends THESE, never the proposal's summary: the server re-reads the
  // message to decide what to apply, and a summary is prose about a change
  // rather than an instruction to make one. "Answer the onboarding." read back
  // as an instruction means nothing, so every answer failed on confirmation
  // with "Trakt could not tell what to do with that" — the proposal was right
  // there on screen and the confirm button could not restate it.
  const [proposedFrom, setProposedFrom] = useState("");
  const [showPackage, setShowPackage] = useState(false);
  const [showPreview, setShowPreview] = useState(false);
  const [confirmCancel, setConfirmCancel] = useState(false);

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
    const message = confirm ? proposedFrom || text : text;
    if (!message.trim()) return;
    const turn = await act(() => client.instructAgent(caseId, message.trim(), confirm));
    if (!turn) return;
    // AN APPLIED TURN HAS NO PENDING PROPOSAL, whatever it carries in the
    // field. The server returns `proposal={"disclosure": ...}` on the applied
    // path so the change can be explained, and that object is TRUTHY: keying
    // the banner off its presence left "Trakt is proposing a change" standing
    // over a change that had already been written.
    //
    // The cost of that is not cosmetic. The operator confirms, sees the same
    // banner, concludes nothing was saved, and either confirms again or starts
    // rewording an instruction that already worked — while the case underneath
    // is correct the whole time. `applied` is the only thing that settles it,
    // so it is read first and the field is ignored when it is true.
    const pending = turn.applied ? null : turn.proposal;
    setProposal(pending);
    // Remember what produced this proposal, so confirming can re-send it.
    setProposedFrom(pending ? message.trim() : "");
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
  const stages = deriveStages(status);
  const current = stages.find((stage) => stage.status === "current");
  const packStage = packOwner(status);
  const yours = operatorBlocking(onboarding.blocking);
  // Open questions that are answered HERE. EVERY question about a column is
  // answered in the mapping table instead — the proposals, the weak matches
  // and the ambiguities alike.
  //
  // Not only to avoid rendering seventy cards beside a table listing the same
  // seventy columns. The table is a DRAFT an operator works down and commits
  // in one act; a card that applied its answer the moment it was clicked would
  // be a second route to the same column with different rules, and the column
  // an operator settled from a card could not then be changed back.
  const openDecisions = status.open_decisions.filter(
    (d) =>
      d.status === "open" &&
      !MAPPING_DECISION_TYPES.has(
        String((d.subject as { decision_type?: string })?.decision_type ?? ""),
      ),
  );

  /**
   * Providing the client's files.
   *
   * Rendered on the artefacts stage AND on the responses stage before it,
   * because an operator may already hold the tape. `register_synthetic_artefact`
   * is permitted from PACK_SENT onwards, but the panel lived only on a stage
   * that stays FUTURE — and a future stage renders no panel — until the
   * responses stage completes. So the capability was allowed by the state
   * machine and unreachable on screen: an operator holding the files had to
   * wait for the client to answer questions before Trakt would take them.
   *
   * Gated on the permission rather than on the stage, so it disappears where
   * the state machine would refuse it rather than rendering a button that can
   * only produce an error.
   */
  const artefactsPanel = (
    <ArtefactsPanel
      status={status}
      busy={busy}
      canGenerate={available.has("register_synthetic_artefact")}
      onUpload={(files) => void act(() => client.uploadAgentArtefacts(caseId, files))}
      onRemove={(artefactId) => void act(() => client.removeAgentArtefact(caseId, artefactId))}
      onGenerate={() => void act(() => client.generateAgentResponse(caseId))}
      onFixture={() => void act(() => client.loadAgentFixtureArtefacts(caseId, run.fixture_id))}
    />
  );

  /** One stage's workflow content. Rendered under exactly one stage. */
  function stageBody(key: StageKey): ReactNode {
    switch (key) {
      case "scope":
        return (
          <ScopeBlock status={status} />
        );
      case "pack_prepare":
      case "pack_review":
      case "pack_issue":
        if (key !== packStage) return null;
        return (
          <PackPanel
            status={status}
            busy={busy}
            caseId={caseId}
            onDraft={() => void act(() => client.draftAgentPack(caseId))}
            onApprove={() => void act(() => client.approveAgentPack(caseId))}
            // ISSUING IS THE ONE ACTION THAT EMAILS A CLIENT, and it was the
            // one action that confirmed nothing: `act` toasts on failure and is
            // silent on success, so a send and a no-op looked identical. The
            // receipt already carries the honest answer — say it here, when the
            // operator is asking the question, not only in the panel below.
            onSend={(to) =>
              void act(async () => {
                const after = await client.sendAgentPack(caseId, to);
                toast.show(
                  after.pack?.sent
                    ? copy.agent.packIssuedToast
                    : copy.agent.packRecordedToast,
                  after.pack?.sent ? "success" : "info",
                );
                return after;
              })
            }
          />
        );
      case "responses":
        return (
          <>
            {/* The stage reads in the order the work happens: what came back,
                then what it answers, then what is still outstanding.

                RECEIVE. A reply the client has already sent is the cheapest
                way to answer the questions below it. */}
            <ClientMailPanel
              caseId={caseId}
              busy={busy}
              canRegister={available.has("register_synthetic_artefact")}
              onIngested={() => void view.reload({ quiet: true })}
            />
            {/* CHASE. What is still outstanding and the way to ask again for
                it. Recording the answers is NOT here: it is in the rail, where
                it is reachable at every stage. This stage completes and
                collapses, and a client's corrections do not stop arriving when
                it does. */}
            <div className="mt-4">
              <ResponsesBlock
                requests={onboarding.information_requests}
                checklist={onboarding.client_checklist}
                busy={busy}
                canAsk={available.has("request_client_information")}
                // Both buttons are gated. An action the state machine refuses
                // used to render anyway, so clicking it could only produce an
                // error — which reads as a broken button, not as a rule being
                // enforced.
                canGenerate={onboarding.client_checklist.length > 0}
                onAsk={() => void act(() => client.runAgentStep(caseId, "information-requests"))}
                // Answers the outstanding questions. This panel used to call
                // generateAgentResponse, which makes up the data FILES — so the
                // call succeeded, the checklist was untouched, and the button
                // looked broken.
                onGenerate={() => void act(() => client.generateAgentAnswers(caseId))}
              />
            </div>
            {/* PROVIDE. Files may already be in hand, and waiting for the
                client to answer questions before Trakt will take them helps
                nobody — the state machine permits it here, so the screen does
                too. */}
            {available.has("register_synthetic_artefact") && (
              <div className="mt-4">{artefactsPanel}</div>
            )}
          </>
        );
      case "artefacts":
        return artefactsPanel;
      case "configure":
        return (
          <>
            {available.has("submit_for_approval") && (
              <PrimaryButton busy={busy} onClick={() => void act(() => client.runAgentStep(caseId, "submit"))}>
                Submit for approval
              </PrimaryButton>
            )}
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
          </>
        );
      case "config_review":
        return available.has("approve_onboarding") ? (
          <PrimaryButton busy={busy} onClick={() => void act(() => client.runAgentStep(caseId, "approve"))}>
            Approve the onboarding
          </PrimaryButton>
        ) : null;
      case "rehearsal":
        return (
          <>
            {available.has("run_synthetic_onboarding") && (
              <PrimaryButton busy={busy} onClick={() => void act(() => client.runAgentStep(caseId, "run"))}>
                Start rehearsal
              </PrimaryButton>
            )}
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
          </>
        );
      case "exceptions":
        return null; // decision cards stay at the top, where they cannot be missed
      case "readiness":
        return (
          <>
            <div className="flex flex-wrap gap-2">
              {available.has("generate_orchestration_plan") && (
                <PrimaryButton busy={busy} onClick={() => void act(() => client.runAgentStep(caseId, "plan"))}>
                  Prepare the execution plan
                </PrimaryButton>
              )}
              {available.has("approve_execution_readiness") && (
                <PrimaryButton
                  busy={busy}
                  onClick={() => void act(() => client.runAgentStep(caseId, "readiness/approve"))}
                >
                  Approve readiness
                </PrimaryButton>
              )}
            </div>
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
          </>
        );
      case "approve_activation":
        return (
          <ReviewPanel
            status={status}
            busy={busy}
            caseId={caseId}
            onSubmit={() => void act(() => client.requestAgentReview(caseId))}
            onApprove={() => void act(() => client.approveAgentActivation(caseId))}
          />
        );
      case "confirm_activation":
        return (
          <ActivationPanel
            status={status}
            busy={busy}
            caseId={caseId}
            onConfirm={(confirmation) =>
              void act(() => client.confirmAgentActivation(caseId, confirmation))
            }
          />
        );
      default:
        return null;
    }
  }

  return (
    <Page
      // The case screen's subject is a TABLE — every source column in the
      // delivery and what Trakt read it as. At the reading width that table had
      // 528px for five columns and every row wrapped onto two lines.
      width="wide"
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
      <SyntheticBanner mode={run.mode} />

      {run.state === "BLOCKED" && status.blockers.length > 0 && (
        <div role="alert" className="mt-4 rounded-2xl border border-rose-200 bg-rose-50 px-4 py-3">
          <p className="text-sm font-semibold text-rose-900">{copy.agent.blockersHeading}</p>
          <ul className="mt-1 list-disc space-y-0.5 pl-4 text-sm text-rose-800">
            {status.blockers.map((blocker) => (
              <li key={blocker}>{blocker}</li>
            ))}
          </ul>
        </div>
      )}

      {isReady && <ReadyBanner />}

      <CaseSummaryCard status={status} currentStageLabel={current?.label ?? ""} />

      <div
        // The status rail costs the main column 22rem, and the main column's
        // job is the mapping table. The rail therefore appears only once there
        // is room for BOTH, which the stock breakpoints get wrong here: the
        // shell's navigation already takes ~264px, so `lg` left the table
        // 824px and even `2xl` left it 808px — both under the 960px at which
        // the headings and the widest status chip stop clipping. Measured, not
        // guessed; the threshold is where the arithmetic actually lands.
        className="mt-6 grid gap-6 min-[1700px]:grid-cols-[minmax(0,1fr)_22rem]"
      >
        <div className="space-y-4">
          {openDecisions.length > 0 && (
            <Panel title={copy.agent.decisionsHeading}>
              <ul className="space-y-4">
                {openDecisions.map((decision) => (
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

          {/* Every column, including the ones nobody was asked about. Placed
              under the decisions because the decisions are the work; this is
              the check on everything the mapper did WITHOUT asking. */}
          <MappingPanel
            mapping={status.mapping}
            requests={run.field_requests ?? []}
            live={run.mode === "live"}
            busy={busy}
            loadFields={loadFields}
            onStage={(input) =>
              void act(() => client.stageAgentMapping(caseId, input))
            }
            onDeclareUnit={(field, unit) =>
              void act(() => client.declareSourceUnit(caseId, { field, unit }))
            }
            onApprove={() =>
              void act(async () => {
                const result = await client.approveAgentMappings(caseId);
                toast.show(
                  copy.agent.mappingCommittedToast(
                    status.mapping.to_confirm ?? 0),
                  "success",
                );
                return result;
              })
            }
            onResolveUnmapped={(input) =>
              void act(async () => {
                const result = await client.resolveUnmappedColumn(caseId, input);
                // Said in the words of the act that happened. A request is not
                // a mapping, and a toast that read the same for both would
                // leave an operator believing the column was settled.
                if (input.action === "use_existing") {
                  toast.show(
                    copy.agent.mappingMappedToast(
                      input.source_column,
                      input.target_field ?? "",
                    ),
                    "success",
                  );
                } else if (input.action === "request_field") {
                  toast.show(
                    copy.agent.mappingRequestedToast(input.field_name ?? ""),
                    "success",
                  );
                } else {
                  toast.show(copy.agent.mappingWithdrawnToast, "success");
                }
                return result;
              })
            }
          />


          <section aria-label={copy.agent.timelineHeading}>
            <h2 className="mb-3 text-sm font-semibold text-stone-900">
              {copy.agent.timelineHeading}
            </h2>
            <ol className="space-y-2">
              {stages.map((stage) => (
                <StageSection key={stage.key} stage={stage}>
                  {stage.status !== "future" ? stageBody(stage.key) : null}
                </StageSection>
              ))}
            </ol>
          </section>
        </div>

        <aside className="space-y-6">
          <Panel title={copy.agent.conversationHeading}>
            <ol className="max-h-80 space-y-3 overflow-y-auto">
              {run.messages.map((message, index) => (
                <li
                  key={`${message.at}-${index}`}
                  className={clsx(
                    "rounded-xl px-3 py-2 text-sm whitespace-pre-wrap",
                    message.role === "operator"
                      ? "bg-stone-100 text-stone-800"
                      : message.role === "client"
                        // A client's own words, arrived from outside. Marked so
                        // they can never be read as something Trakt said or an
                        // operator instructed — which is the whole reason the
                        // role exists rather than being folded into "agent".
                        ? "border border-blue-200 bg-blue-50 text-stone-800"
                        : "border border-stone-200 bg-white text-stone-700",
                  )}
                >
                  {message.role === "client" && (
                    <p className="mb-1 text-xs font-semibold text-blue-800">
                      {copy.agent.mailFrom(message.author || copy.agent.mailUnnamed)}
                    </p>
                  )}
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
                    onClick={() => {
                      setProposal(null);
                      setProposedFrom("");
                    }}
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

          <Panel title={copy.agent.statusHeading}>
            <Field label={copy.agent.onboardingStageHeading}>
              <StatusChip
                status={onboarding.status === "approved" ? "ready" : "waiting"}
                label={onboarding.status_label}
              />
            </Field>
            <Field label={copy.agent.stageHeading(run.mode === "live")}>
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
          </Panel>

          <RunTargetPanel
            run={run}
            busy={busy}
            onSave={(input) => void act(() => client.setAgentRunTarget(caseId, input))}
          />

          <ConcentrationPanel
            answers={onboarding.answers as unknown as Record<string, unknown>}
            busy={busy}
            onRecord={(input) => void act(() => client.recordAgentConcentration(caseId, input))}
          />

          <ClientQuestionsPanel
            caseId={caseId}
            version={status.run.version}
            confirmations={status.pack.confirmations ?? []}
            checklist={onboarding.client_checklist}
            requests={onboarding.information_requests}
            busy={busy}
            onSaved={() => void view.reload({ quiet: true })}
          />

          <CriteriaPanel
            criteria={status.readiness.criteria}
            live={run.mode === "live"}
            /* Expanded once the case is at readiness, where the table is the
               work. Before then most rows read "Blocked" only because the case
               has not got there yet, which is not information — it is the wall
               that made "what is pending" unanswerable. */
            open={stages.find((stage) => stage.key === "readiness")?.status !== "future"}
          />

          {/* The blocked banner at the top already lists these when the run is
              BLOCKED; this panel covers blockers recorded in any other state. */}
          {status.blockers.length > 0 && run.state !== "BLOCKED" && (
            <Panel title={copy.agent.blockersHeading}>
              <ul className="list-disc space-y-1 pl-4 text-sm text-rose-700">
                {status.blockers.map((blocker) => (
                  <li key={blocker}>{blocker}</li>
                ))}
              </ul>
            </Panel>
          )}

          {yours.length > 0 && (
            <Panel title={copy.agent.missingHeading}>
              <p className="mb-2 text-xs text-stone-500">{copy.agent.missingHelp}</p>
              <ul className="list-disc space-y-1 pl-4 text-sm text-stone-600">
                {yours.map((problem) => (
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

          <details className="rounded-2xl border border-stone-200 bg-white p-5">
            <summary className="cursor-pointer text-sm font-semibold text-stone-900">
              {copy.agent.gatesHeading}
            </summary>
            <ol className="mt-3 space-y-1">
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
          </details>

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

      {/* ENDING THE CASE. Alone at the foot of the page, below everything that
          moves it forward, because it is not a way forward — the same place and
          the same quiet treatment Client Onboarding gives the identical act.
          Offered only while the case can still be cancelled, so a finished or
          already-cancelled case shows nothing. */}
      {available.has(CANCEL_ACTION) && (
        <div className="mt-10 border-t border-stone-200 pt-6">
          <button
            type="button"
            onClick={() => setConfirmCancel(true)}
            className="text-sm font-medium text-stone-500 underline-offset-4 hover:text-stone-900 hover:underline"
          >
            {copy.agent.cancelLink}
          </button>
        </div>
      )}

      {confirmCancel && (
        <AgentCancelDialog
          live={run.mode === "live"}
          busy={busy}
          onDismiss={() => setConfirmCancel(false)}
          onConfirm={(reason) => {
            void act(async () => {
              const result = await client.runAgentStep(caseId, "cancel", { reason });
              setConfirmCancel(false);
              toast.show(copy.agent.cancelledToast, "success");
              return result;
            });
          }}
        />
      )}
    </Page>
  );
}

/** One clear primary action for a stage. */
function PrimaryButton({
  busy,
  onClick,
  children,
}: {
  busy: boolean;
  onClick: () => void;
  children: ReactNode;
}) {
  return (
    <button
      type="button"
      disabled={busy}
      onClick={onClick}
      className="rounded-xl bg-blue-600 px-4 py-2 text-sm font-semibold text-white disabled:opacity-50"
    >
      {children}
    </button>
  );
}

/**
 * The compact structured summary at the top of every case: who this is for,
 * which mode, where it stands, and — separately — each declared data stream.
 */
function CaseSummaryCard({
  status,
  currentStageLabel,
}: {
  status: AgentStatus;
  currentStageLabel: string;
}) {
  const practice = status.run.mode !== "live";
  return (
    <section className="mt-4 rounded-2xl border border-stone-200 bg-white p-5">
      <dl className="grid gap-x-8 gap-y-1 text-sm sm:grid-cols-3">
        <div>
          <dt className="text-xs uppercase tracking-wide text-stone-400">
            {copy.agent.summaryClient}
          </dt>
          <dd className="font-medium text-stone-900">
            {status.onboarding.client_name || status.case_ref}
          </dd>
        </div>
        <div>
          <dt className="text-xs uppercase tracking-wide text-stone-400">
            {copy.agent.summaryMode}
          </dt>
          <dd className="font-medium text-stone-900">
            {practice ? copy.agent.summaryModePractice : copy.agent.summaryModeLive}
          </dd>
        </div>
        <div>
          <dt className="text-xs uppercase tracking-wide text-stone-400">
            {copy.agent.summaryStage}
          </dt>
          <dd className="font-medium text-stone-900">{currentStageLabel || "—"}</dd>
        </div>
      </dl>
      <div className="mt-4">
        <p className="text-xs uppercase tracking-wide text-stone-400">
          {copy.agent.summaryStreams}
        </p>
        {status.streams.length === 0 ? (
          <p className="mt-1 text-sm text-stone-500">{copy.agent.streamsNone}</p>
        ) : (
          <ol className="mt-1 space-y-2">
            {status.streams.map((stream, index) => (
              <StreamRow key={stream.source_key || index} stream={stream} index={index} />
            ))}
          </ol>
        )}
      </div>
    </section>
  );
}

function StreamRow({ stream, index }: { stream: StreamSummary; index: number }) {
  const regime =
    stream.regime_status === "not_applicable"
      ? copy.agent.streamRegimeNone
      : stream.regime_status === "potential"
        ? copy.agent.streamRegimePotential
        : stream.regime_status === "configured"
          ? copy.agent.streamRegimeConfigured
          : (stream.regime_note || copy.agent.streamRegimeNotEligible);
  return (
    <li className="rounded-xl border border-stone-200 px-3 py-2" data-testid={`stream-${stream.dataset}`}>
      <p className="text-sm font-semibold text-stone-900">
        {index + 1}. {stream.label}
      </p>
      <dl className="mt-1 grid gap-x-6 gap-y-0.5 text-xs text-stone-600 sm:grid-cols-2">
        <div className="flex gap-1">
          <dt className="text-stone-400">{copy.agent.streamPurpose}:</dt>
          <dd>{stream.purpose}</dd>
        </div>
        <div className="flex gap-1">
          <dt className="text-stone-400">{copy.agent.streamCadence}:</dt>
          <dd>
            {stream.cadence ? humanize(stream.cadence) : "—"}
            {stream.cadence && !stream.cadence_confirmed
              ? `, ${copy.agent.streamCadencePending}`
              : ""}
          </dd>
        </div>
        <div className="flex gap-1">
          <dt className="text-stone-400">{copy.agent.streamFile}:</dt>
          <dd>{stream.required_file || "—"}</dd>
        </div>
        <div className="flex gap-1">
          <dt className="text-stone-400">{copy.agent.streamRegime}:</dt>
          <dd>{regime}</dd>
        </div>
      </dl>
    </li>
  );
}

/** One stage of the timeline: collapsed when done, expanded when current. */
function StageSection({ stage, children }: { stage: StageInfo; children: ReactNode }) {
  if (stage.status === "future") {
    return (
      <li
        className="flex items-center gap-3 rounded-xl border border-transparent px-3 py-2"
        data-stage={stage.key}
        data-stage-status="future"
      >
        <Circle className="h-4 w-4 shrink-0 text-stone-300" aria-hidden />
        <span className="text-sm text-stone-400">{stage.label}</span>
        {stage.liveOnly && (
          <span className="text-xs text-stone-400">— {copy.agent.stageLiveOnly}</span>
        )}
      </li>
    );
  }

  if (stage.status === "done") {
    return (
      <li data-stage={stage.key} data-stage-status="done">
        <details className="rounded-xl border border-stone-200 bg-white">
          <summary className="flex cursor-pointer items-center gap-3 px-3 py-2">
            <CheckCircle2 className="h-4 w-4 shrink-0 text-emerald-500" aria-hidden />
            <span className="text-sm font-medium text-stone-700">{stage.label}</span>
            {stage.note && <span className="truncate text-xs text-stone-400">{stage.note}</span>}
            <span className="ml-auto shrink-0 text-xs font-medium text-emerald-600">
              {copy.agent.stageDone}
            </span>
          </summary>
          {children && <div className="space-y-4 border-t border-stone-100 p-3">{children}</div>}
        </details>
      </li>
    );
  }

  return (
    <li
      className="rounded-2xl border-2 border-blue-200 bg-white p-4"
      data-stage={stage.key}
      data-stage-status="current"
    >
      <div className="flex items-center gap-3">
        <CircleDot className="h-5 w-5 shrink-0 text-blue-500" aria-hidden />
        <span className="text-sm font-semibold text-stone-900">{stage.label}</span>
        <span
          className={clsx(
            "ml-auto shrink-0 rounded-full px-2 py-0.5 text-xs font-medium",
            stage.blocked ? "bg-rose-100 text-rose-800" : "bg-blue-100 text-blue-800",
          )}
        >
          {stage.blocked ? copy.agent.stageBlockedChip : copy.agent.stageCurrent}
        </span>
      </div>
      {stage.note && <p className="mt-1 pl-8 text-sm text-stone-500">{stage.note}</p>}
      <div className="mt-3 space-y-4">{children}</div>
    </li>
  );
}

/** What the scope stage still needs, in the catalogue's own words. */
function ScopeBlock({ status }: { status: AgentStatus }) {
  const missing = status.onboarding.blocking.slice(0, 5);
  return (
    <div className="text-sm text-stone-600">
      {status.streams.length === 0 && <p>{copy.agent.streamsNone}</p>}
      {missing.length > 0 && (
        <>
          <p className="mt-2 font-medium text-stone-700">{copy.agent.stageWhatRemains}</p>
          <ul className="mt-1 list-disc space-y-0.5 pl-4">
            {missing.map((problem) => (
              <li key={`${problem.section}-${problem.field}-${problem.index}`}>
                {problem.message}
              </li>
            ))}
          </ul>
        </>
      )}
      <p className="mt-2 text-xs text-stone-500">{copy.agent.newCasePrompt}</p>
    </div>
  );
}

/** Recording what the client sent back, or letting practice mode stand in. */
function ResponsesBlock({
  requests,
  checklist,
  busy,
  canAsk,
  canGenerate,
  onAsk,
  onGenerate,
}: {
  requests: InformationRequest[];
  checklist: ChecklistRow[];
  busy: boolean;
  canAsk: boolean;
  canGenerate: boolean;
  onAsk: () => void;
  onGenerate: () => void;
}) {
  const outstanding = requests.filter((r) => ["open", "sent"].includes(r.status));
  return (
    <div className="text-sm text-stone-600">
      {outstanding.length > 0 ? (
        <p>
          {outstanding.length} request{outstanding.length === 1 ? "" : "s"}{" "}
          {copy.agent.requestOutstanding.toLowerCase()}.
        </p>
      ) : checklist.length > 0 ? (
        <p>{checklist.length} item(s) still outstanding from the client.</p>
      ) : (
        <p>{copy.agent.checklistEmpty}</p>
      )}
      <div className="mt-3 flex flex-wrap gap-2">
        {canAsk && checklist.length > 0 && (
          <button
            type="button"
            disabled={busy}
            onClick={onAsk}
            className="rounded-xl bg-blue-600 px-3 py-1.5 text-sm font-semibold text-white disabled:opacity-50"
          >
            {copy.agent.checklistAsk}
          </button>
        )}
        {canGenerate && (
          <button
            type="button"
            disabled={busy}
            onClick={onGenerate}
            className="rounded-xl border border-stone-300 px-3 py-1.5 text-sm font-medium text-stone-700 hover:bg-stone-50 disabled:opacity-50"
          >
            {copy.agent.uploadGenerate}
          </button>
        )}
      </div>
    </div>
  );
}

/**
 * Every source column, and what the header mapper made of it.
 *
 * The decisions panel above shows only the columns the mapper could NOT
 * settle. Everything it settled on its own — the majority, and the part nobody
 * has checked — had no screen at all: `run.mapping_report` reached the API in
 * the readiness and review packages, was typed in the frontend, and was
 * rendered nowhere. On a hundred-column tape that is the difference between
 * answering the questions asked and being able to see all the answers.
 *
 * Every row's `state` comes from the server. It turns on the same trusted-tier
 * and confidence test the engine applies when deciding whether to use a mapping
 * without asking, and a copy of that test here could disagree with the engine
 * about which mappings a human checked.
 */
function MappingPanel({
  mapping,
  requests,
  live,
  busy,
  onApprove,
  onStage,
  onResolveUnmapped,
  onDeclareUnit,
  loadFields,
}: {
  mapping: MappingOverview;
  requests: FieldRequest[];
  live: boolean;
  busy: boolean;
  onApprove: () => void;
  onStage: (input: StageInput) => void;
  onResolveUnmapped: (input: UnmappedInput) => void;
  onDeclareUnit: (field: string, unit: string) => void;
  loadFields: () => Promise<RegistryField[]>;
}) {
  const [filter, setFilter] = useState("");
  // Which unmapped column the operator is answering, if any. One at a time:
  // the question is about THIS column, and a form that could be about three
  // of them is a form somebody answers for the wrong one.
  const [answering, setAnswering] = useState<MappingRow | null>(null);
  const counts = mapping.counts ?? {};
  const rows =
    filter === CONTESTED
      ? mapping.rows.filter((r) => r.also_claimed_by.length > 0)
      : filter
        ? mapping.rows.filter((r) => r.state === filter)
        : mapping.rows;
  // WHAT THE COMMIT WOULD DO. `staged` is what the operator has been through
  // by hand; `proposed` is what they have left as Trakt read it, which commits
  // with the rest; `mustAnswerFirst` has no answer at all and is why the
  // button is refused rather than being allowed to invent one.
  const staged = mapping.staged ?? 0;
  const proposed = mapping.proposed ?? 0;
  const toConfirm = mapping.to_confirm ?? staged + proposed;
  const mustAnswerFirst = mapping.unanswered_questions ?? 0;
  const contested = mapping.contested ?? 0;

  // Only the states actually present are offered. A chip reading "Not used 0"
  // is a question nobody asked.
  const chips = MAPPING_STATES.filter((state) => (counts[state] ?? 0) > 0);

  return (
    <Panel title={copy.agent.mappingHeading}>
      {mapping.rows.length === 0 ? (
        <p className="text-sm text-stone-500">{copy.agent.mappingEmpty(live)}</p>
      ) : (
        <>
          <p className="text-sm text-stone-600">
            {copy.agent.mappingCount(counts.mapped ?? 0, counts.columns ?? 0)}
          </p>
          <p className="mt-1 text-xs text-stone-500">{copy.agent.mappingHelp}</p>

          {toConfirm > 0 && (
            <div className="mt-3 rounded-xl border border-amber-200 bg-amber-50 p-3">
              <p className="max-w-3xl text-xs text-amber-900">
                {copy.agent.mappingApproveHelp}
              </p>
              {/* Said out loud, because the whole point of the draft is that
                  it has NOT happened. An operator who has confirmed forty rows
                  needs to know the forty are still theirs to change. */}
              <p className="mt-1 max-w-3xl text-xs text-amber-800">
                {copy.agent.mappingDraftHelp}
              </p>
              <div className="mt-2 flex flex-wrap items-center gap-3">
                <button
                  type="button"
                  disabled={busy || mustAnswerFirst > 0}
                  onClick={onApprove}
                  className="rounded-lg bg-stone-900 px-3 py-1.5 text-sm font-semibold text-white disabled:opacity-50"
                >
                  {copy.agent.mappingCommit(toConfirm)}
                </button>
                {/* The button names its own consequence: how many the operator
                    went through, and how many commit exactly as Trakt read
                    them. "Confirm 26" without that is 26 of what. */}
                <span className="text-xs text-amber-900">
                  {copy.agent.mappingCommitBreakdown(staged, proposed)}
                </span>
                {mustAnswerFirst > 0 && (
                  // Offering to settle the set while a real question waits
                  // would promise a run that cannot move.
                  <span className="text-xs font-medium text-amber-900">
                    {copy.agent.mappingApproveBlocked(mustAnswerFirst)}
                  </span>
                )}
              </div>
            </div>
          )}

          <div className="mt-3 flex flex-wrap gap-2">
            <FilterChip
              label={`${copy.agent.mappingFilterAll} ${counts.columns ?? 0}`}
              active={filter === ""}
              onClick={() => setFilter("")}
            />
            {chips.map((state) => (
              <FilterChip
                key={state}
                label={`${MAPPING_STATE_LABELS[state]} ${counts[state]}`}
                active={filter === state}
                onClick={() => setFilter(state)}
              />
            ))}
            {/* Not a state — a row is contested AND proposed at once — so it
                filters on its own. An operator asked to be able to find them
                before approving the set, and a count they cannot filter to is
                just a number. */}
            {contested > 0 && (
              <FilterChip
                label={`${copy.agent.mappingContestedFilter} ${contested}`}
                active={filter === CONTESTED}
                onClick={() => setFilter(CONTESTED)}
              />
            )}
          </div>

          {/* Grouped by file. A pack is three or four tapes and a flat list
              of every column across all of them reads as one enormous table
              with no way to tell which file a column came from — and which
              file it came from is what decides whether a weak match is a
              question waiting for you or a note. */}
          {mapping.files
            .filter((file) => rows.some((row) => row.source_file === file.name))
            .map((file) => (
              <section key={file.name} className="mt-6 first:mt-4">
                <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
                  <h4 className="text-sm font-semibold text-stone-900">{file.name}</h4>
                  <span
                    className={clsx(
                      "rounded-full px-2 py-0.5 text-xs font-medium",
                      file.primary
                        ? "bg-stone-900 text-white"
                        : "bg-stone-100 text-stone-600",
                    )}
                  >
                    {file.primary
                      ? copy.agent.mappingPrimaryFile
                      : copy.agent.mappingSecondaryFile}
                  </span>
                  <span className="text-xs tabular-nums text-stone-500">
                    {copy.agent.mappingFileColumns(file.columns)}
                  </span>
                </div>
                {/* ONE ROW, ONE LINE. `table-fixed` plus an explicit width per
                    column is what makes that true: without it the browser
                    sizes columns from their content, so one long header or one
                    long evidence sentence re-flows the whole table and every
                    row wraps. Each cell then truncates and carries its full
                    text in `title`, so nothing is lost — it is one hover away
                    rather than one line down.

                    `min-w` keeps the promise on a narrow viewport: the table
                    scrolls sideways inside its own box instead of wrapping,
                    because a row split across two lines is the defect being
                    fixed and a scrollbar is not. */}
                <div className="mt-2 overflow-x-auto rounded-xl border border-stone-200">
                  {/* 72rem, not 60: the table carries an action column now,
                      and three controls need about 190px. Below that the
                      buttons themselves were clipped — which is worse than
                      the scrollbar, because a clipped row still LOOKS
                      complete. `overflow-hidden` on the cells stops anything
                      printing over its neighbour; this stops there being
                      anything to clip. */}
                  <table className="w-full min-w-[72rem] table-fixed text-left text-sm">
                    {/* Proportions measured against the rendered table, not
                        guessed: at 16% the status chip truncated to "Matched
                        automa…", at 19% to "Weak match, nothing a…", and at 9%
                        the confidence heading itself was clipped to
                        "CONFIDENC".

                        What gets the room is what an operator triages on — the
                        column's own name, what Trakt read it as, and the
                        status. The evidence is the justification for those
                        three and is the one that may truncate, because it is
                        a hover away and they are not. */}
                    {/* Six columns now: the operator ACTS on this table, and
                        an action crammed into the field cell overran it and
                        printed on top of the next column. What gets the room
                        is still what they triage on — the column's own name,
                        what Trakt read it as, and the status; the evidence is
                        the justification for those and is the one that
                        truncates, because it is a hover away and they are
                        not. */}
                    <colgroup>
                      <col className="w-[16%]" />
                      <col className="w-[14%]" />
                      <col className="w-[28%]" />
                      <col className="w-[13%]" />
                      <col className="w-[10%]" />
                      <col className="w-[19%]" />
                    </colgroup>
                    <thead>
                      {/* Sticky against the PAGE scroll: a real tape is
                          seventy to a hundred columns, and by row forty an
                          operator is reading five unlabelled values. */}
                      <tr className="sticky top-0 z-10 border-b border-stone-200 bg-stone-50 text-xs uppercase tracking-wide text-stone-500">
                        <th className="truncate px-3 py-2 font-medium">{copy.agent.mappingColumn}</th>
                        <th className="truncate px-3 py-2 font-medium">{copy.agent.mappingState}</th>
                        <th className="truncate px-3 py-2 font-medium">{copy.agent.mappingField}</th>
                        <th className="truncate px-3 py-2 font-medium">{copy.agent.mappingBasis}</th>
                        <th className="truncate px-3 py-2 text-right font-medium">
                          {copy.agent.mappingConfidence}
                        </th>
                        <th className="truncate px-3 py-2 text-right font-medium">
                          <span className="sr-only">{copy.agent.mappingRowConfirm}</span>
                        </th>
                      </tr>
                    </thead>
                    <tbody>
                      {rows
                        .filter((row) => row.source_file === file.name)
                        .map((row) => (
                          <tr
                            key={`${row.source_file}:${row.source_column}`}
                            className="border-b border-stone-100 last:border-0 hover:bg-stone-50"
                          >
                            <td
                              className="truncate px-3 py-2.5 font-medium text-stone-900"
                              title={row.source_column || undefined}
                            >
                              {row.source_column || copy.agent.mappingNothing}
                            </td>
                            <td className="px-3 py-2.5">
                              <span
                                className={clsx(
                                  "inline-block max-w-full truncate whitespace-nowrap rounded-full px-2 py-0.5 text-xs font-medium",
                                  MAPPING_STATE_TONES[row.state],
                                )}
                                title={row.state_label}
                              >
                                {row.state_label}
                              </span>
                            </td>
                            {/* `overflow-hidden` is the backstop: a fixed
                                layout does not clip on its own, so anything
                                that will not shrink prints OVER the next
                                column rather than being cut off by it. */}
                            <td className="overflow-hidden px-3 py-2.5 text-stone-700">
                              <MappingFieldCell
                                row={row}
                                busy={busy}
                                onDeclareUnit={(unit) =>
                                  onDeclareUnit(row.staged_field || row.canonical_field, unit)
                                }
                              />
                            </td>
                            {/* WHAT KIND OF READING THIS IS, first and always
                                in the same place. The tier sentence explains
                                the evidence and is the right length to read;
                                it is the wrong length to scan a hundred and
                                fifty rows by, and on a model-suggested row it
                                read "Nothing Trakt reports on resembles this
                                column" beside a chip saying a model had
                                proposed one. */}
                            <td
                              className="overflow-hidden px-3 py-2.5 text-xs text-stone-500"
                              title={[row.match_kind_label, row.tier_label,
                                      row.note, row.decision_detail]
                                .filter(Boolean)
                                .join(" — ")}
                            >
                              <span
                                className={clsx(
                                  "inline-block max-w-full truncate whitespace-nowrap rounded px-1.5 py-0.5 font-medium",
                                  MATCH_KIND_TONES[row.match_kind] ??
                                    "bg-stone-100 text-stone-600",
                                )}
                              >
                                {row.match_kind_label}
                              </span>
                            </td>
                            <td className="px-3 py-2.5 text-right tabular-nums text-stone-600">
                              {row.confidence === null
                                ? copy.agent.mappingNothing
                                : `${Math.round(row.confidence * 100)}%`}
                            </td>
                            <td className="overflow-hidden px-3 py-2.5 text-right">
                              <MappingRowActions
                                row={row}
                                busy={busy}
                                onStage={onStage}
                                onPickField={() => setAnswering(row)}
                                onWithdrawRequest={() =>
                                  onResolveUnmapped({
                                    source_file: row.source_file,
                                    source_column: row.source_column,
                                    action: "withdraw_request",
                                  })
                                }
                              />
                            </td>
                          </tr>
                        ))}
                    </tbody>
                  </table>
                </div>
              </section>
            ))}

          {/* An ask is a governed record, so it is on the screen rather than
              only in the audit trail: an operator who asked for a field last
              week should not have to remember that they did. */}
          {requests.filter((r) => r.status === "requested").length > 0 && (
            <section className="mt-6 rounded-xl border border-amber-200 bg-amber-50 p-4">
              <h4 className="text-sm font-semibold text-amber-900">
                {copy.agent.mappingRequestsHeading}
              </h4>
              <ul className="mt-2 space-y-2">
                {requests
                  .filter((r) => r.status === "requested")
                  .map((request) => (
                    <li key={request.request_id} className="text-xs text-amber-900">
                      <span className="font-semibold">{request.field_name}</span>
                      {" — "}
                      {request.source_column} in {request.source_file}
                      {request.description && <> · {request.description}</>}
                      <button
                        type="button"
                        disabled={busy}
                        onClick={() =>
                          onResolveUnmapped({
                            source_file: request.source_file,
                            source_column: request.source_column,
                            action: "withdraw_request",
                          })
                        }
                        className="ml-2 font-medium underline disabled:opacity-50"
                      >
                        {copy.agent.mappingWithdrawRequest}
                      </button>
                    </li>
                  ))}
              </ul>
              <p className="mt-2 max-w-3xl text-xs text-amber-800">
                {copy.agent.mappingRequestNewHelp}
              </p>
            </section>
          )}

          {answering && (
            <UnmappedColumnDialog
              row={answering}
              busy={busy}
              loadFields={loadFields}
              onCancel={() => setAnswering(null)}
              onSubmit={(input) => {
                setAnswering(null);
                onResolveUnmapped(input);
              }}
            />
          )}
        </>
      )}
    </Panel>
  );
}

/** The filter that is not a state: a row is contested AND proposed at once. */
const CONTESTED = "__contested__";

/** Every question about a source column. All of them are answered on the
 *  mapping table, which is the one surface with the draft-then-commit rules. */
const MAPPING_DECISION_TYPES = new Set([
  "mapping_proposal",
  "mapping_confirmation",
  "mapping_ambiguity",
]);

/** What the screen asks the server to record about one column. A draft. */
type StageInput = {
  source_file: string;
  source_column: string;
  action: "confirm" | "amend" | "not_used" | "clear";
  target_field?: string;
  reason?: string;
};

/**
 * What an operator does to one row.
 *
 * THE WHOLE TABLE IS A DRAFT UNTIL IT IS COMMITTED, so these words have to
 * keep two things apart that used to be one. "Confirm" here records that this
 * operator has read this column and agrees — and applies nothing. The button
 * at the top of the panel is what applies the lot.
 *
 * A row that has been answered shows the answer and an undo, not the choices
 * again: the question on it has been asked and answered, and re-offering
 * "Confirm / Change / Do not use" invites it to be answered twice.
 */
function MappingRowActions({
  row,
  busy,
  onStage,
  onPickField,
  onWithdrawRequest,
}: {
  row: MappingRow;
  busy: boolean;
  onStage: (input: StageInput) => void;
  onPickField: () => void;
  onWithdrawRequest: () => void;
}) {
  const where = { source_file: row.source_file, source_column: row.source_column };
  if (row.state === "staged") {
    // A SET-ASIDE A REQUEST WROTE UNDOES AS THE REQUEST, not as a set-aside.
    // The ask is what put the column out of the delivery, so "Undo" here would
    // put it back while the ask still stood — mapped and requested at once,
    // which is the state this whole path exists to prevent. One control, and
    // it says what it actually does.
    const byRequest = row.staged_origin === "field_request";
    // NO ANSWER PHRASE ON A REQUEST ROW. On every other staged row the phrase
    // IS the answer — "You said: valuation date". On this one the answer is
    // "Requested: <field>", which the field cell already carries, so a phrase
    // beside it is the fourth thing on the row saying the same thing and the
    // only one paying for it in width: "Set aside for the new field" truncated
    // to "Set asi…" at 1280 and still to "Set aside for …" at 1680.
    const said = byRequest
      ? ""
      : row.staged_action === "not_used"
        ? copy.agent.mappingStagedNotUsed
        : row.staged_action === "amend"
          ? copy.agent.mappingStagedAmend(row.staged_field)
          : copy.agent.mappingStagedConfirm;
    return (
      <div className="flex min-w-0 items-baseline justify-end gap-2">
        {said && (
          <span className="truncate text-xs text-emerald-700" title={said}>
            {said}
          </span>
        )}
        <button
          type="button"
          disabled={busy}
          title={byRequest ? copy.agent.mappingStagedRequested : undefined}
          onClick={() =>
            byRequest ? onWithdrawRequest() : onStage({ ...where, action: "clear" })
          }
          className="shrink-0 text-xs font-medium text-blue-700 underline disabled:opacity-50"
        >
          {byRequest
            ? copy.agent.mappingWithdrawRequest
            : copy.agent.mappingRowUndo}
        </button>
      </div>
    );
  }
  // A COLUMN ALREADY COMMITTED IS STILL NOT A PERMANENT ONE. It becomes
  // permanent at activation, when promotion writes it into the client's
  // governed rules; until then the rehearsal is provisional. This used to
  // render nothing at all, which told an operator reading a settled row that
  // their mistake was final when it was not.
  //
  // Set apart from the ordinary acts, and warned about in the title, because
  // it is not the ordinary act: it sends the case back to this step and
  // withdraws anything approved on the old reading.
  if (row.state === "confirmed") {
    return (
      <div className="flex min-w-0 items-baseline justify-end gap-2">
        <button
          type="button"
          disabled={busy}
          title={copy.agent.mappingReopenWarning}
          onClick={onPickField}
          className="shrink-0 text-xs font-medium text-amber-700 underline disabled:opacity-50"
        >
          {copy.agent.mappingRowReopen}
        </button>
        {row.canonical_field && (
          <button
            type="button"
            disabled={busy}
            title={copy.agent.mappingReopenWarning}
            onClick={() => onStage({ ...where, action: "not_used" })}
            className="shrink-0 text-xs font-medium text-stone-500 underline hover:text-stone-700 disabled:opacity-50"
          >
            {copy.agent.mappingRowNotUsed}
          </button>
        )}
      </div>
    );
  }
  // A file that could not be read has no column to answer about.
  if (row.state === "unreadable") return null;
  return (
    <div className="flex min-w-0 items-baseline justify-end gap-2">
      {row.canonical_field && (
        <button
          type="button"
          disabled={busy}
          onClick={() => onStage({ ...where, action: "confirm" })}
          className="shrink-0 rounded border border-stone-300 px-1.5 py-0.5 text-xs font-medium text-stone-700 hover:bg-stone-100 disabled:opacity-50"
        >
          {copy.agent.mappingRowConfirm}
        </button>
      )}
      <button
        type="button"
        disabled={busy}
        onClick={onPickField}
        className="shrink-0 text-xs font-medium text-blue-700 underline disabled:opacity-50"
      >
        {row.canonical_field
          ? copy.agent.mappingRowChange
          : copy.agent.mappingUnmappedAction}
      </button>
      {row.canonical_field && (
        <button
          type="button"
          disabled={busy}
          onClick={() => onStage({ ...where, action: "not_used" })}
          className="shrink-0 text-xs font-medium text-stone-500 underline hover:text-stone-700 disabled:opacity-50"
        >
          {copy.agent.mappingRowNotUsed}
        </button>
      )}
    </div>
  );
}

/** How firm each kind of reading is, at a glance. A known alias and a model's
 *  guess are not the same claim and must not share a colour. */
const MATCH_KIND_TONES: Record<string, string> = {
  operator: "bg-emerald-50 text-emerald-700",
  alias: "bg-blue-50 text-blue-700",
  name: "bg-blue-50 text-blue-700",
  similar: "bg-amber-50 text-amber-800",
  model: "bg-violet-50 text-violet-700",
  none: "bg-stone-100 text-stone-500",
  unreadable: "bg-rose-50 text-rose-700",
};

/** What the screen asks the server to do about one unmapped column. *//** What the screen asks the server to do about one unmapped column. */
type UnmappedInput = {
  source_file: string;
  source_column: string;
  action: "use_existing" | "request_field" | "withdraw_request";
  target_field?: string;
  field_name?: string;
  label?: string;
  description?: string;
  data_type?: string;
  reason?: string;
};

/**
 * What to do about a column nothing in the registry resembled.
 *
 * THE TWO CHOICES ARE NOT THE SAME ACT, and the dialog is built so they cannot
 * be mistaken for each other. Naming a field Trakt already has is settled on
 * the spot and promotes into this client's governed rules. Asking for a field
 * Trakt does NOT have changes the vocabulary every client's report is written
 * in — so it is recorded as a request, the column stays unmapped, and the
 * panel says so in the form rather than after the fact.
 *
 * The field list is the server's (`OccAgentService.field_catalogue`, the
 * mapper's own selection), fetched when the dialog opens rather than held on
 * every status response: it is five hundred fields, it does not change during
 * a case, and most operators never open this.
 */
function UnmappedColumnDialog({
  row,
  busy,
  loadFields,
  onCancel,
  onSubmit,
}: {
  row: MappingRow;
  busy: boolean;
  loadFields: () => Promise<RegistryField[]>;
  onCancel: () => void;
  onSubmit: (input: UnmappedInput) => void;
}) {
  const [mode, setMode] = useState<"use_existing" | "request_field">("use_existing");
  const [fields, setFields] = useState<RegistryField[] | null>(null);
  const [chosen, setChosen] = useState("");
  const [name, setName] = useState("");
  const [what, setWhat] = useState("");
  const [type, setType] = useState("");

  useEffect(() => {
    let live = true;
    void loadFields().then((list) => {
      if (live) setFields(list);
    });
    return () => {
      live = false;
    };
  }, [loadFields]);

  const known = fields ?? [];
  const valid =
    mode === "use_existing"
      ? known.some((f) => f.name === chosen)
      : name.trim().length > 0;

  return (
    <Modal labelledBy="unmapped-column-title">
      <h3 id="unmapped-column-title" className="text-lg font-semibold text-stone-900">
        {copy.agent.mappingUnmappedHeading(row.source_column)}
      </h3>
      <p className="mt-1 text-xs text-stone-500">{row.source_file}</p>
      {/* A settled column is a different conversation from one that matched
          nothing: it already HAS a field, and changing it costs the approvals
          that rested on it. Saying "nothing resembled this column" over a row
          reading `current interest rate` would be the screen contradicting
          itself at the moment it asks for a decision. */}
      <p className="mt-3 text-sm text-stone-600">
        {row.state === "confirmed"
          ? copy.agent.mappingReopenWarning
          : copy.agent.mappingUnmappedIntro}
      </p>

      <fieldset className="mt-4 space-y-3">
        <label className="flex gap-3">
          <input
            type="radio"
            name="unmapped-mode"
            className="mt-1"
            // Named explicitly: the visible label carries the explanation as
            // well as the choice, and a control whose name is a paragraph is
            // one a screen reader reads as a paragraph.
            aria-label={copy.agent.mappingUseExisting}
            checked={mode === "use_existing"}
            onChange={() => setMode("use_existing")}
          />
          <span>
            <span className="text-sm font-medium text-stone-900">
              {copy.agent.mappingUseExisting}
            </span>
            <span className="mt-0.5 block text-xs text-stone-500">
              {copy.agent.mappingUseExistingHelp}
            </span>
          </span>
        </label>
        {mode === "use_existing" && (
          <div className="pl-7">
            <label
              htmlFor="unmapped-field"
              className="block text-xs font-medium text-stone-700"
            >
              {copy.agent.mappingPickField}
            </label>
            {/* A list, not a free-text box: a field that is not in the
                registry is not a mapping, and the server refuses one. The
                browser's own filtering keeps five hundred fields usable. */}
            <input
              id="unmapped-field"
              list="unmapped-field-options"
              value={chosen}
              disabled={busy || fields === null}
              placeholder={copy.agent.mappingPickFieldPlaceholder}
              onChange={(e) => setChosen(e.target.value)}
              className="mt-1 w-full rounded-lg border border-stone-300 px-3 py-2 text-sm"
            />
            <datalist id="unmapped-field-options">
              {known.map((field) => (
                <option key={field.name} value={field.name}>
                  {field.regimes.length > 0
                    ? `${field.label} · ${field.regimes.join(", ")}`
                    : field.label}
                </option>
              ))}
            </datalist>
          </div>
        )}

        <label className="flex gap-3">
          <input
            type="radio"
            name="unmapped-mode"
            className="mt-1"
            aria-label={copy.agent.mappingRequestNew}
            checked={mode === "request_field"}
            onChange={() => setMode("request_field")}
          />
          <span>
            <span className="text-sm font-medium text-stone-900">
              {copy.agent.mappingRequestNew}
            </span>
            <span className="mt-0.5 block text-xs text-stone-500">
              {copy.agent.mappingRequestNewHelp}
            </span>
          </span>
        </label>
        {mode === "request_field" && (
          <div className="space-y-3 pl-7">
            <div>
              <label
                htmlFor="new-field-name"
                className="block text-xs font-medium text-stone-700"
              >
                {copy.agent.mappingNewFieldName}
              </label>
              <input
                id="new-field-name"
                value={name}
                disabled={busy}
                placeholder={copy.agent.mappingNewFieldNamePlaceholder}
                onChange={(e) => setName(e.target.value)}
                className="mt-1 w-full rounded-lg border border-stone-300 px-3 py-2 text-sm"
              />
            </div>
            <div>
              <label
                htmlFor="new-field-what"
                className="block text-xs font-medium text-stone-700"
              >
                {copy.agent.mappingNewFieldWhat}
              </label>
              <textarea
                id="new-field-what"
                value={what}
                rows={2}
                disabled={busy}
                placeholder={copy.agent.mappingNewFieldWhatPlaceholder}
                onChange={(e) => setWhat(e.target.value)}
                className="mt-1 w-full rounded-lg border border-stone-300 px-3 py-2 text-sm"
              />
            </div>
            <div>
              <label
                htmlFor="new-field-type"
                className="block text-xs font-medium text-stone-700"
              >
                {copy.agent.mappingNewFieldType}
              </label>
              <select
                id="new-field-type"
                value={type}
                disabled={busy}
                onChange={(e) => setType(e.target.value)}
                className="mt-1 w-full rounded-lg border border-stone-300 px-3 py-2 text-sm"
              >
                <option value="">{copy.agent.mappingNothing}</option>
                <option value="string">Text</option>
                <option value="decimal">A number</option>
                <option value="date">A date</option>
                <option value="list">One of a fixed set of values</option>
                <option value="Y/N">Yes or no</option>
              </select>
            </div>
          </div>
        )}
      </fieldset>

      <DialogButtons
        onCancel={onCancel}
        busy={busy}
        disabled={!valid}
        confirmLabel={
          mode === "use_existing"
            ? copy.agent.mappingUseExistingConfirm
            : copy.agent.mappingRequestConfirm
        }
        onConfirm={() =>
          onSubmit(
            mode === "use_existing"
              ? {
                  source_file: row.source_file,
                  source_column: row.source_column,
                  action: "use_existing",
                  target_field: chosen,
                }
              : {
                  source_file: row.source_file,
                  source_column: row.source_column,
                  action: "request_field",
                  field_name: name.trim(),
                  description: what.trim(),
                  data_type: type,
                },
          )
        }
      />
    </Modal>
  );
}

/**
 * What Trakt reads a column as — and, where it could not, what a model
 * proposed instead.
 *
 * A column the deterministic tiers could not place has no canonical field, so
 * this cell used to render "—" for it. Now that the model is wired into the
 * mapping stage that is the one row an operator most needs to see: the
 * proposal is on the run and was visible nowhere. It is shown as a PROPOSAL —
 * in the muted voice, with the basis beside it — because a suggestion set in
 * the same type as a contract-backed match is the model writing mappings by
 * another route.
 *
 * One line: the name truncates, the link does not, so "Answer this" is never
 * pushed off the row by a long field name.
 */
function MappingFieldCell({
  row,
  busy,
  onDeclareUnit,
}: {
  row: MappingRow;
  busy: boolean;
  onDeclareUnit: (unit: string) => void;
}) {
  // A COLUMN CARRIES ITS REQUEST WHATEVER TRAKT MADE OF IT. This was gated on
  // `row.state === "unused"`, written on the assumption that an ask only ever
  // comes from a column that matched nothing. It does not: "Change" opens the
  // same dialog on any row, so a column with a live proposal can be requested
  // as a new field — and on that row the chip was thrown away before it
  // reached the cell. The operator saw their ask recorded in the panel below
  // and the row above it still reading as though nothing had happened.
  const requested = row.requested_field;
  // AN OPEN REQUEST SETTLES WHAT THIS CELL IS ABOUT. A model's guess and an
  // operator's ask are both answers to "what is this column?", and the
  // operator's is the later and the deciding one — they have said Trakt has no
  // field for it. Showing both put four things on one row, which overran the
  // cell and printed the action on top of the next column.
  const proposed = !requested && !row.canonical_field
    && Boolean(row.suggested_label);
  const label = requested ? "" : proposed ? row.suggested_label : row.field_label;
  // HOW THE LENDER WRITES THIS PERCENTAGE. Only on a field Trakt holds as
  // percentage POINTS, where 35 and 0.35 can mean the same thing and the
  // platform cannot tell from the number alone. Left unset, Trakt reconciles
  // the scale against the balance and the valuation — right whenever it has
  // both — so this is the answer for the books where it has not.
  const unitControl = row.percentage_scaled ? (
    <span className="ml-auto flex shrink-0 items-center gap-1">
      <select
        aria-label={copy.agent.mappingUnitLabel}
        title={copy.agent.mappingUnitHelp}
        disabled={busy}
        value={row.source_unit}
        onChange={(e) => onDeclareUnit(e.target.value)}
        className="rounded border border-stone-300 bg-white px-1 py-0.5 text-xs text-stone-700 disabled:opacity-50"
      >
        <option value="">{copy.agent.mappingUnitAuto}</option>
        <option value="percentage_points">{copy.agent.mappingUnitPoints}</option>
        <option value="fraction">{copy.agent.mappingUnitFraction}</option>
      </select>
    </span>
  ) : null;
  return (
    <div className="flex min-w-0 items-baseline gap-2">
      {/* The em-dash stands for "no field", so it is dropped where something
          else in the cell already says what became of the column: "— Requested:
          broker_code" reads as a field called nothing AND a field asked for. */}
      {(label || !requested) && (
        <span
          className={clsx("truncate", proposed && "italic text-stone-500")}
          title={
            [label, proposed ? row.basis_label : "", row.suggested_reason]
              .filter(Boolean)
              .join(" — ") || undefined
          }
        >
          {label || copy.agent.mappingNothing}
        </span>
      )}
      {proposed && (
        <span className="shrink-0 rounded bg-violet-50 px-1.5 py-0.5 text-xs font-medium text-violet-700">
          {copy.agent.mappingProposed}
        </span>
      )}
      {requested && (
        <span
          className="min-w-0 truncate rounded bg-amber-50 px-1.5 py-0.5 text-xs font-medium text-amber-800"
          // WHAT THE ASK DISPLACED, where it displaced something. On a column
          // that matched nothing there is nothing to say; on one Trakt had
          // read as a field, the reading is the very thing the operator
          // overruled, and an operator revisiting the row a week later needs
          // to see what they overruled it with.
          title={[copy.agent.mappingRequestNewHelp,
                  row.field_label
                    ? copy.agent.mappingRequestDisplaced(row.field_label)
                    : ""]
            .filter(Boolean)
            .join(" ")}
        >
          {copy.agent.mappingRequestedChip(requested)}
        </span>
      )}
      {row.also_claimed_by.length > 0 && (() => {
        // A field more than one column reads as — and the two cases are not
        // the same thing, so they must not wear the same warning.
        //
        // SAME FILE is an ambiguity the engine cannot resolve: which of these
        // two columns is the balance? The run blocks on it and the operator
        // has to pick one.
        //
        // DIFFERENT FILES is the ordinary shape of a delivery, and more than
        // that — it is REQUIRED. Every extract carries a loan identifier and
        // the assembler needs each of them to join on; a file without one
        // contributes nothing. Reported in orange, an operator who had just
        // correctly mapped the identifier in a second file was told they had
        // a problem, on the row they had just got right.
        const ambiguous = row.also_claimed_by.some((c) => c.same_file);
        const elsewhere = row.also_claimed_by.filter((c) => !c.same_file);
        return (
          <span
            className={
              "shrink-0 rounded px-1.5 py-0.5 text-xs font-medium " +
              (ambiguous
                ? "bg-orange-50 text-orange-800"
                : "bg-stone-100 text-stone-600")
            }
            title={[ambiguous
                      ? copy.agent.mappingAmbiguousHelp
                      : copy.agent.mappingAlsoInHelp,
                    ...row.also_claimed_by.map(
                      (c) => `${c.source_column} in ${c.source_file}`),
                    // WHAT THE CHOICE TURNS ON. How many records each competing
                    // column actually carries used to live on a decision card;
                    // deleting the card would have deleted the one thing that
                    // settles the question.
                    row.decision_detail].filter(Boolean).join(" ")}
          >
            {ambiguous
              ? copy.agent.mappingContested(row.also_claimed_by.length + 1)
              : copy.agent.mappingAlsoIn(elsewhere.length)}
          </span>
        );
      })()}
      {unitControl}
    </div>
  );
}

const MAPPING_STATES = [
  "needs_you",
  "proposed",
  "unreadable",
  "unchecked",
  "unused",
  "staged",
  "confirmed",
  "automatic",
] as const;

/** The server sends a label per row; these are for the filter chips, which
 *  exist whether or not a row of that kind is on screen. */
const MAPPING_STATE_LABELS: Record<string, string> = {
  needs_you: "Needs you",
  proposed: "Proposed",
  unreadable: "Could not be read",
  unchecked: "Weak match, nothing asked",
  unused: "Not used",
  staged: "Ready to confirm",
  confirmed: "You confirmed",
  automatic: "Matched automatically",
};

const MAPPING_STATE_TONES: Record<string, string> = {
  needs_you: "bg-amber-100 text-amber-800",
  // Distinct from "needs you": a proposal is read and approved, not answered.
  proposed: "bg-sky-100 text-sky-800",
  unreadable: "bg-rose-100 text-rose-800",
  unchecked: "bg-orange-50 text-orange-700",
  unused: "bg-stone-100 text-stone-600",
  // Read and answered, and not yet committed. Distinct from "You confirmed
  // it", which is the same reading after the set was applied.
  staged: "bg-emerald-50 text-emerald-700",
  confirmed: "bg-emerald-100 text-emerald-800",
  automatic: "bg-blue-50 text-blue-700",
};

function FilterChip({
  label,
  active,
  onClick,
}: {
  label: string;
  active: boolean;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      aria-pressed={active}
      onClick={onClick}
      className={clsx(
        "rounded-full border px-3 py-1 text-xs font-medium",
        active
          ? "border-stone-900 bg-stone-900 text-white"
          : "border-stone-300 bg-white text-stone-600 hover:bg-stone-50",
      )}
    >
      {label}
    </button>
  );
}

/**
 * The concentration-test decision, recorded in one act.
 *
 * Approval is blocked while this sits at "waiting on the client", and only an
 * operator moves it. The service method has always taken the whole decision —
 * the status, the client's own wording and the reason — with its own controls:
 * a blank answer can never be recorded as supplied, and only the four declared
 * statuses are accepted. Nothing called it.
 *
 * So the status went in through the conversation and the limits through the
 * client form: two acts with nothing tying them together, and a conversation
 * that truncates prose at the first clause. An operator could set "supplied"
 * with no answer behind it, or paste an answer nobody had recorded a decision
 * about.
 */
function ConcentrationPanel({
  answers,
  busy,
  onRecord,
}: {
  answers: Record<string, unknown>;
  busy: boolean;
  onRecord: (input: { status: string; response_text: string; reason: string }) => void;
}) {
  const held = (answers.risk_limits ?? {}) as Record<string, unknown>;
  const [status, setStatus] = useState(
    String(held.concentration_tests_status ?? "pending_client_response"),
  );
  const [text, setText] = useState(String(held.concentration_tests ?? ""));
  const [reason, setReason] = useState(String(held.concentration_tests_status_reason ?? ""));

  useEffect(() => {
    setStatus(String(held.concentration_tests_status ?? "pending_client_response"));
    setText(String(held.concentration_tests ?? ""));
    setReason(String(held.concentration_tests_status_reason ?? ""));
  }, [held.concentration_tests_status, held.concentration_tests, held.concentration_tests_status_reason]);

  const needsText = status === "supplied" && !text.trim();
  const needsReason =
    ["not_applicable", "deferred_with_reason"].includes(status) && !reason.trim();

  return (
    <Panel title={copy.agent.concentrationHeading}>
      <p className="text-xs text-stone-500">{copy.agent.concentrationHelp}</p>

      <label className="mt-3 block text-xs font-medium text-stone-600" htmlFor="conc-status">
        {copy.agent.concentrationStatus}
      </label>
      <select
        id="conc-status"
        value={status}
        onChange={(event) => setStatus(event.target.value)}
        className="mt-1 w-full rounded-lg border border-stone-300 px-2 py-1.5 text-sm"
      >
        <option value="pending_client_response">{copy.agent.concentrationPending}</option>
        <option value="supplied">{copy.agent.concentrationSupplied}</option>
        <option value="not_applicable">{copy.agent.concentrationNotApplicable}</option>
        <option value="deferred_with_reason">{copy.agent.concentrationDeferred}</option>
      </select>

      <label className="mt-3 block text-xs font-medium text-stone-600" htmlFor="conc-text">
        {copy.agent.concentrationText}
      </label>
      <textarea
        id="conc-text"
        rows={6}
        value={text}
        onChange={(event) => setText(event.target.value)}
        className="mt-1 w-full rounded-lg border border-stone-300 px-2 py-1.5 text-sm"
      />
      <p className="mt-1 text-xs text-stone-500">{copy.agent.concentrationTextHelp}</p>

      <label className="mt-3 block text-xs font-medium text-stone-600" htmlFor="conc-reason">
        {copy.agent.concentrationReason}
      </label>
      <textarea
        id="conc-reason"
        rows={2}
        value={reason}
        onChange={(event) => setReason(event.target.value)}
        className="mt-1 w-full rounded-lg border border-stone-300 px-2 py-1.5 text-sm"
      />

      {/* Said before the button is pressed, not after it is refused. The
          server enforces both; this only explains why it would. */}
      {needsText && (
        <p className="mt-2 text-xs text-amber-700">{copy.agent.concentrationNeedsText}</p>
      )}
      {needsReason && (
        <p className="mt-2 text-xs text-amber-700">{copy.agent.concentrationNeedsReason}</p>
      )}

      <button
        type="button"
        disabled={busy || needsText || needsReason}
        onClick={() =>
          onRecord({ status, response_text: text.trim(), reason: reason.trim() })
        }
        className="mt-3 rounded-xl bg-stone-900 px-3 py-1.5 text-sm font-medium text-white hover:bg-stone-800 disabled:opacity-40"
      >
        {copy.agent.concentrationSave}
      </button>
    </Panel>
  );
}

/**
 * Which delivery this run is for: the book, and the period it reports.
 *
 * `POST /cases/{ref}/target` and `setAgentRunTarget` both existed already; no
 * screen called either, so the reporting period could not be set from anywhere
 * and the panel above could only DISPLAY one, hidden entirely while empty. The
 * visible symptom was a file card reading "Where this would be filed: —",
 * because the intended URI needs client, portfolio and period, and the period
 * was the one of the three with no way in.
 *
 * The period is free text rather than `<input type="month">` deliberately: a
 * month picker cannot express a pipeline snapshot's own date (2026-09-14) or
 * an ISO week, and both are periods Trakt files under. The server canonicalises
 * what is typed and refuses what it cannot read, so the tolerance costs nothing.
 */
function RunTargetPanel({
  run,
  busy,
  onSave,
}: {
  run: AgentStatus["run"];
  busy: boolean;
  onSave: (input: { dataset: string; reporting_period: string }) => void;
}) {
  const [period, setPeriod] = useState(run.reporting_period);
  const [dataset, setDataset] = useState(run.dataset || "funded");

  // The run is the source of truth: a save, or another operator's change
  // arriving on a reload, replaces what is in the boxes.
  useEffect(() => {
    setPeriod(run.reporting_period);
    setDataset(run.dataset || "funded");
  }, [run.reporting_period, run.dataset]);

  const dirty = period !== run.reporting_period || dataset !== (run.dataset || "funded");
  return (
    <Panel title={copy.agent.targetHeading}>
      <label className="block text-xs font-medium text-stone-600" htmlFor="run-period">
        {copy.agent.targetPeriodLabel}
      </label>
      <input
        id="run-period"
        value={period}
        placeholder={copy.agent.targetPeriodPlaceholder}
        onChange={(event) => setPeriod(event.target.value)}
        className="mt-1 w-full rounded-lg border border-stone-300 px-2 py-1.5 text-sm"
      />
      <p className="mt-1 text-xs text-stone-500">{copy.agent.targetHelp}</p>

      <label className="mt-3 block text-xs font-medium text-stone-600" htmlFor="run-dataset">
        {copy.agent.targetDatasetLabel}
      </label>
      <select
        id="run-dataset"
        value={dataset}
        onChange={(event) => setDataset(event.target.value)}
        className="mt-1 w-full rounded-lg border border-stone-300 px-2 py-1.5 text-sm"
      >
        <option value="funded">{copy.agent.targetDatasetFunded}</option>
        <option value="pipeline">{copy.agent.targetDatasetPipeline}</option>
      </select>

      <button
        type="button"
        disabled={busy || !dirty || !period.trim()}
        onClick={() => onSave({ dataset, reporting_period: period.trim() })}
        className="mt-3 rounded-xl bg-stone-900 px-3 py-1.5 text-sm font-medium text-white hover:bg-stone-800 disabled:opacity-40"
      >
        {copy.agent.targetSave}
      </button>
    </Panel>
  );
}

/**
 * One file's row, with the way to take it back out.
 *
 * The confirmation is inline rather than a dialog because the thing being
 * confirmed is the row itself: an operator removing the third of four files
 * needs to see WHICH one while they decide. It also states what removal does
 * and does not do — the record goes, the uploaded bytes stay in the case
 * sandbox — because "Remove" on its own reads as "delete", and that would be
 * a promise the platform cannot keep.
 */
function ArtefactRow({
  artefact,
  busy,
  onRemove,
}: {
  artefact: AgentStatus["run"]["received_artefacts"][number];
  busy: boolean;
  onRemove: () => void;
}) {
  const [confirming, setConfirming] = useState(false);
  return (
    <li className="rounded-xl border border-stone-200 px-3 py-2">
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <p className="text-sm font-medium text-stone-900">{artefact.source_file}</p>
          <p className="text-xs text-stone-500">
            {artefact.artefact_type ? humanize(artefact.artefact_type) : copy.workflow.fileKind}
            {artefact.row_count > 0 && ` · ${artefact.row_count} records`}
          </p>
        </div>
        {!confirming && (
          <button
            type="button"
            disabled={busy}
            onClick={() => setConfirming(true)}
            className="shrink-0 rounded-lg border border-stone-300 px-2 py-1 text-xs font-medium text-stone-600 hover:bg-stone-50 disabled:opacity-50"
          >
            {copy.agent.artefactRemove}
          </button>
        )}
      </div>
      <p className="mt-1 break-all text-xs text-stone-400">
        {copy.agent.artefactIntended}: {artefact.intended_live_uri || "—"}
      </p>
      <p className="text-xs font-medium text-violet-700">{copy.agent.artefactNotWritten}</p>
      {confirming && (
        <div className="mt-2 rounded-lg bg-stone-50 p-2">
          <p className="text-xs text-stone-600">{copy.agent.artefactRemoveExplain}</p>
          <div className="mt-2 flex gap-2">
            <button
              type="button"
              disabled={busy}
              onClick={() => {
                setConfirming(false);
                onRemove();
              }}
              className="rounded-lg bg-rose-600 px-2 py-1 text-xs font-medium text-white hover:bg-rose-700 disabled:opacity-50"
            >
              {copy.agent.artefactRemoveConfirm}
            </button>
            <button
              type="button"
              onClick={() => setConfirming(false)}
              className="rounded-lg border border-stone-300 px-2 py-1 text-xs font-medium text-stone-600 hover:bg-white"
            >
              {copy.agent.artefactRemoveKeep}
            </button>
          </div>
        </div>
      )}
    </li>
  );
}

/** The practice files: what has arrived, and the ways to provide it. */
function ArtefactsPanel({
  status,
  busy,
  canGenerate,
  onUpload,
  onRemove,
  onGenerate,
  onFixture,
}: {
  status: AgentStatus;
  busy: boolean;
  canGenerate: boolean;
  onUpload: (files: File[]) => void;
  onRemove: (artefactId: string) => void;
  onGenerate: () => void;
  onFixture: () => void;
}) {
  const run = status.run;
  const noDestination =
    run.received_artefacts.length > 0 &&
    run.received_artefacts.every((artefact) => !artefact.intended_live_uri);
  return (
    <Panel title={copy.agent.artefactsHeading}>
      {run.received_artefacts.length === 0 ? (
        <>
          <Empty />
          <p className="mt-2 text-sm text-stone-500">{copy.agent.uploadHelp}</p>
        </>
      ) : (
        <ul className="space-y-3">
          {run.received_artefacts.map((artefact) => (
            <ArtefactRow
              key={artefact.artefact_id}
              artefact={artefact}
              busy={busy}
              onRemove={() => onRemove(artefact.artefact_id)}
            />
          ))}
        </ul>
      )}
      {noDestination && (
        <p className="mt-2 text-sm text-stone-500">{copy.agent.artefactNoDestination}</p>
      )}
      <div className="mt-3 flex flex-wrap gap-2">
        <label className="cursor-pointer rounded-xl border border-stone-300 px-3 py-1.5 text-sm font-medium text-stone-700 hover:bg-stone-50">
          {copy.agent.uploadButton}
          <input
            type="file"
            multiple
            className="hidden"
            onChange={(event) => {
              const files = Array.from(event.target.files ?? []);
              if (files.length > 0) onUpload(files);
            }}
          />
        </label>
        {canGenerate && (
          <button
            type="button"
            disabled={busy}
            onClick={onGenerate}
            className="rounded-xl border border-stone-300 px-3 py-1.5 text-sm font-medium text-stone-700 hover:bg-stone-50 disabled:opacity-50"
          >
            {copy.agent.uploadGenerate}
          </button>
        )}
        {run.fixture_id && (
          <button
            type="button"
            disabled={busy}
            onClick={onFixture}
            className="rounded-xl border border-stone-300 px-3 py-1.5 text-sm font-medium text-stone-700 hover:bg-stone-50 disabled:opacity-50"
          >
            {copy.agent.uploadFixture}
          </button>
        )}
      </div>
    </Panel>
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
  // Tolerant reads: a pack projected by an older backend must degrade to an
  // empty list, never crash the case page.
  const sections = pack.sections ?? [];
  const confirmations = pack.confirmations ?? [];
  const notAsked = pack.not_asked ?? [];
  const allowed = new Set(status.state.allowed_human_actions ?? []);
  const document = useLoad(() => client.getAgentPack(caseId), [caseId, status.run.version]);
  const recipients = pack.email?.to ?? [];

  return (
    <Panel
      title={copy.agent.packHeading}
      action={
        sections.length > 0 ? (
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

      {sections.length === 0 ? (
        <p className="mt-3 text-sm text-stone-400">{copy.agent.packNone}</p>
      ) : (
        <>
          {/* What the client is asked, grouped as they will meet it. */}
          <p className="mt-3 text-sm text-stone-700">
            <span className="font-medium">{pack.questions}</span>{" "}
            {copy.agent.classificationClientFacing}
            {pack.summary?.total ? ` (of ${pack.summary.total} fields)` : ""}
          </p>
          <ul className="mt-2 space-y-1 text-sm text-stone-600">
            {steps(sections).map((step) => (
              <li key={step.key} className="flex justify-between gap-2">
                <span>{step.label}</span>
                <span className="text-xs text-stone-500">
                  {step.questions} · {step.required} {copy.agent.packRequired}
                </span>
              </li>
            ))}
          </ul>

          {confirmations.length > 0 && (
            <details className="mt-3">
              <summary className="cursor-pointer text-sm font-medium text-stone-700">
                {copy.agent.packConfirmHeading} ({confirmations.length})
              </summary>
              <p className="mt-1 text-xs text-stone-500">{copy.agent.packConfirmNote}</p>
              <ul className="mt-1 space-y-0.5 text-sm text-stone-600">
                {confirmations.map((question) => (
                  <li key={`${question.section}.${question.field}-${question.index}`}>
                    {question.label}: <span className="font-medium">{render(question.value)}</span>
                  </li>
                ))}
              </ul>
            </details>
          )}

          {notAsked.length > 0 && (
            <details className="mt-2">
              <summary className="cursor-pointer text-sm font-medium text-stone-700">
                {copy.agent.packNotAskedHeading} ({notAsked.length})
              </summary>
              <ul className="mt-1 space-y-0.5 text-sm text-stone-600">
                {notAsked.map((row) => (
                  <li key={row.key}>
                    {row.label} — <span className="text-xs text-stone-500">{row.reason}</span>
                  </li>
                ))}
              </ul>
            </details>
          )}
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
            className={clsx(
              "rounded-xl px-3 py-1.5 text-sm",
              sections.length > 0
                ? "border border-stone-300 font-medium text-stone-700 hover:bg-stone-50"
                : "bg-blue-600 font-semibold text-white",
              "disabled:opacity-50",
            )}
          >
            {sections.length > 0 ? copy.agent.packRedraft : copy.agent.packDraft}
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
            {/* A DISABLED BUTTON MUST SAY WHY. Without this the control goes
                grey and nothing else changes, which reads as a broken button —
                and the operator's next move is to press it again rather than to
                fill in the box beside it. */}
            {recipients.length === 0 && !recipient.trim() && (
              <span role="note" className="text-xs text-stone-500">
                {copy.agent.packNeedsAddress}
              </span>
            )}
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

/** The pack's sections regrouped into the client-facing steps. */
function steps(sections: AgentStatus["pack"]["sections"]) {
  const order: string[] = [];
  for (const section of sections) {
    if (!order.includes(section.step)) order.push(section.step);
  }
  return order.map((key) => {
    const own = sections.filter((s) => s.step === key);
    return {
      key,
      label: own[0]?.step_label || key,
      questions: own.reduce((n, s) => n + s.questions.length, 0),
      required: own.reduce(
        (n, s) => n + s.questions.filter((q) => q.required).length,
        0,
      ),
    };
  });
}

function render(value: unknown): string {
  if (value === null || value === undefined || value === "") return "—";
  if (typeof value === "boolean") return value ? "yes" : "no";
  if (Array.isArray(value)) return value.join(", ") || "—";
  return String(value);
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
            className="rounded-xl bg-blue-600 px-3 py-1.5 text-sm font-semibold text-white disabled:opacity-50"
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
 * The readiness table, folded away until it is the work.
 *
 * Nine rows, always expanded, six of them "Blocked" purely because the case
 * has not reached them — the same wall that made the agent's own answer to
 * "what is pending" useless. The count is the part worth seeing at every
 * stage; the table is one click from it, and open by default once the case is
 * actually at readiness.
 */
function CriteriaPanel({
  criteria,
  open,
  live,
}: {
  criteria: ReadinessCriterion[];
  open: boolean;
  live: boolean;
}) {
  const passed = criteria.filter((criterion) => criterion.passed).length;
  return (
    <details open={open} className="rounded-2xl border border-stone-200 bg-white p-5">
      <summary className="flex cursor-pointer items-center gap-2 text-sm font-semibold text-stone-900">
        {copy.agent.criteriaHeading}
        <span className="ml-auto shrink-0 text-xs font-medium text-stone-500">
          {copy.agent.criteriaSummary(passed, criteria.length)}
        </span>
      </summary>
      <div className="mt-3">
        <CriteriaList criteria={criteria} live={live} />
      </div>
    </details>
  );
}

/** Readiness criteria, grouped by which half of the process they belong to. */
function CriteriaList({
  criteria,
  live,
}: {
  criteria: ReadinessCriterion[];
  live: boolean;
}) {
  const groups: { key: ReadinessCriterion["stage"]; label: string }[] = [
    { key: "onboarding", label: copy.agent.criteriaOnboarding },
    { key: "execution", label: copy.agent.criteriaExecution(live) },
    { key: "boundary", label: copy.agent.criteriaBoundary(live) },
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
  // Cancelling has its own control at the foot of the page. Left in here it
  // appeared as a bullet reading "cancel run" under "what you can do next" —
  // which is both the wrong claim (abandoning is not a way forward) and the
  // wrong words, and was the only mention of cancelling anywhere on the screen.
  const conversational = [...available].filter(
    (action) =>
      action !== CANCEL_ACTION && !STEPS.some((entry) => entry.action === action),
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
    <li
      // The mapping table links here, so a row reading "Needs you" is the
      // thing you click rather than a label sending you to hunt for the
      // matching question in another list.
      id={`decision-${decision.decision_id}`}
      className="scroll-mt-6 rounded-xl border border-amber-200 bg-amber-50 p-4"
    >
      {/* The card runs the full width of a wide screen; its PROSE does not.
          A question set across 1100px is a question nobody finishes reading. */}
      <p className="max-w-3xl text-sm font-semibold text-stone-900">{decision.title}</p>
      <p className="mt-1 max-w-3xl text-sm text-stone-700">{decision.question}</p>

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
