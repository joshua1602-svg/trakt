import type { AgentStatus } from "@/api/agentTypes";
import { copy } from "@/lib/copy";

/**
 * The operator journey, derived from the case's own states.
 *
 * Nothing here is a new state model: every predicate reads a decision the
 * backend already made — the onboarding case's status, the practice run's
 * lifecycle, the pack's own workflow — and folds them into the fifteen
 * chronological stages an operator works through. Delete this file and the
 * states all still exist; only the timeline goes.
 */

export type StageKey =
  | "define"
  | "scope"
  | "pack_prepare"
  | "pack_review"
  | "pack_issue"
  | "responses"
  | "artefacts"
  | "configure"
  | "config_review"
  | "rehearsal"
  | "exceptions"
  | "readiness"
  | "approve_activation"
  | "confirm_activation"
  | "ingestion";

export const STAGE_ORDER: StageKey[] = [
  "define",
  "scope",
  "pack_prepare",
  "pack_review",
  "pack_issue",
  "responses",
  "artefacts",
  "configure",
  "config_review",
  "rehearsal",
  "exceptions",
  "readiness",
  "approve_activation",
  "confirm_activation",
  "ingestion",
];

export interface StageInfo {
  key: StageKey;
  label: string;
  status: "done" | "current" | "future";
  /** One line shown on a collapsed or future stage. */
  note?: string;
  /** True when the case is blocked while this stage is current. */
  blocked?: boolean;
  /** A stage practice mode deliberately never reaches. */
  liveOnly?: boolean;
}

function reachedStates(status: AgentStatus): Set<string> {
  return new Set(
    status.lifecycle.filter((entry) => entry.reached).map((entry) => entry.state),
  );
}

/** Whether each stage is complete, given only what the backend decided. */
function completions(status: AgentStatus): Record<StageKey, boolean> {
  const onboarding = status.onboarding;
  const run = status.run;
  const pack = status.pack;
  const reached = reachedStates(status);
  const packStatus = pack.status ?? "";
  const packDrafted = (pack.sections ?? []).length > 0 || Boolean(packStatus);
  const artefactsIn = (run.received_artefacts ?? []).length > 0;
  const outstandingRequests = (onboarding.information_requests ?? []).filter(
    (request) => ["open", "sent"].includes(request.status),
  );
  const checklist = onboarding.client_checklist ?? [];
  const submitted = ["ready_for_approval", "approved", "activated"].includes(
    onboarding.status,
  );
  const approved = ["approved", "activated"].includes(onboarding.status);
  const rehearsalPassed =
    reached.has("SYNTHETIC_ONBOARDING_PASSED") ||
    run.readiness_status === "READY_FOR_EXECUTION";
  const openDecisions = (run.open_decisions ?? []).filter(
    (decision) => decision.status === "open",
  );
  const ready =
    run.readiness_status === "READY_FOR_EXECUTION" || reached.has("READY_FOR_EXECUTION");
  const activationApproved =
    reached.has("APPROVED_FOR_ACTIVATION") ||
    ["ACTIVATION_CONFIRMATION_REQUIRED", "ACTIVATING", "INGESTION_STARTED"].includes(
      run.state,
    );

  return {
    define: Boolean(status.facts.client_id || onboarding.client_name),
    scope: Boolean(status.facts.portfolio_id) && status.streams.length > 0,
    // An operator who already holds the client's files may skip the pack; the
    // lifecycle allows it, so the timeline does too.
    pack_prepare: packDrafted || artefactsIn,
    pack_review:
      ["APPROVED_TO_SEND", "SENT"].includes(packStatus) || (!packDrafted && artefactsIn),
    pack_issue: packStatus === "SENT" || (!packDrafted && artefactsIn),
    responses:
      onboarding.ready || (outstandingRequests.length === 0 && checklist.length === 0),
    artefacts: artefactsIn,
    configure: submitted,
    config_review: approved,
    rehearsal: rehearsalPassed,
    exceptions: rehearsalPassed && openDecisions.length === 0,
    readiness: ready,
    approve_activation: activationApproved,
    confirm_activation: run.state === "INGESTION_STARTED",
    ingestion: run.state === "INGESTION_STARTED",
  };
}

/** Notes worth showing on a stage row, beyond done/current/future. */
function stageNote(key: StageKey, status: AgentStatus): string {
  const run = status.run;
  const pack = status.pack;
  switch (key) {
    case "define":
      return status.onboarding.client_name || "";
    case "scope":
      return status.streams.length > 0
        ? status.streams.map((stream) => stream.label).join(" + ")
        : "";
    case "pack_prepare":
      return (pack.sections ?? []).length > 0 ? `${pack.questions} questions` : "";
    case "pack_issue":
      return pack.status === "SENT"
        ? pack.sent
          ? copy.agent.packIssued
          : copy.agent.packNotSent
        : "";
    case "artefacts": {
      const count = (run.received_artefacts ?? []).length;
      return count > 0 ? `${count} file${count === 1 ? "" : "s"} received` : "";
    }
    case "exceptions": {
      const open = (run.open_decisions ?? []).filter((d) => d.status === "open").length;
      const total = (run.open_decisions ?? []).length;
      if (open > 0) return `${open} open`;
      return total === 0 ? "No exceptions were raised" : "All resolved";
    }
    case "confirm_activation":
      // The activation panel itself states the refusal; the row stays short.
      return "";
    case "ingestion":
      return copy.agent.stageLiveOnly;
    default:
      return "";
  }
}

/**
 * The whole timeline. Exactly one stage is current unless the case is
 * finished; a blocked run marks the current stage blocked rather than adding
 * a sixteenth stage.
 */
export function deriveStages(status: AgentStatus): StageInfo[] {
  const done = completions(status);
  const blocked = status.run.state === "BLOCKED";
  const finished = status.run.state === "INGESTION_STARTED";
  const currentIndex = finished
    ? STAGE_ORDER.length
    : STAGE_ORDER.findIndex((key) => !done[key]);

  return STAGE_ORDER.map((key, index) => ({
    key,
    label: copy.agent.stages[key] ?? key,
    status: index < currentIndex ? "done" : index === currentIndex ? "current" : "future",
    note: stageNote(key, status),
    blocked: blocked && index === currentIndex,
    liveOnly: key === "ingestion",
  }));
}

export function currentStage(status: AgentStatus): StageInfo | undefined {
  return deriveStages(status).find((stage) => stage.status === "current");
}
