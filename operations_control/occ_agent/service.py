"""operations_control.occ_agent.service — the OCC Agent's typed tool surface.

This is the bounded set of operations the agent can perform. There is no
general-purpose "do what the text says" entry point, and the interpreter cannot
reach the store, the filesystem or the pipeline except through these methods.

The service is deliberately thin over two things it does not own:

* **the onboarding** is :class:`operations_control.onboarding.service.
  OnboardingService` — the platform's own governed capability, driven here
  rather than reimplemented. Opening a case, answering a step, asking the client
  for what is missing, recording what came back, submitting for approval and
  approving are all *its* operations, with its validation, its inference, its
  transition table and its event history. The one call it offers that this
  feature must never make is ``activate()``, which its own docstring calls "the
  only place active configuration is created" — so
  :meth:`OccAgentService.activate` names the capability and always refuses;
* **the pipeline** is the existing orchestration conductor, run over the
  synthetic execution adapter.

What this module adds is the part neither has: a natural-language door onto the
first, and a practice execution that carries the second from an *approved but
never activated* case to ``READY_FOR_EXECUTION``.

The order of every state-changing execution method is the same, and it matters:

1. check the current run state permits the action (:mod:`.states`);
2. do the deterministic work (real components, real configuration);
3. derive the resulting state from what the controls actually returned;
4. assert the transition is legal;
5. persist, then audit.

So a control decides the state, and the interpreter only ever chooses which tool
to call. :meth:`OccAgentService.instruct` is the natural-language door: it
interprets, and for anything material it returns a *proposal* the human must
confirm before step 2 runs at all.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from apps.blob_trigger_app.storage import Storage

from ..contracts import new_id, now_iso
from ..engine import OpsError
from ..onboarding.case import (
    ACTIVATED,
    APPROVED,
    CHANGES_REQUIRED,
    DRAFT,
    IN_REVIEW,
    INFORMATION_REQUESTED,
    KIND_NEW_CLIENT,
    NO_ACTIVE_CONFIGURATION,
    READY_FOR_APPROVAL,
    STATUS_LABELS,
    TERMINAL,
    OnboardingCase,
)
from ..onboarding.service import STEP_LABELS, STEPS, OnboardingService
from ..stores import OpsLayout, OpsStore
from . import adapters as _adapters
from . import classification as _classification
from . import client_form as _client_form
from . import communication as _comms
from . import derive as _derive
from . import execution as _execution
from . import field_registry as _field_registry
from . import mapping_promotion as _mapping_promotion
from . import mapping_view as _mapping_view
from . import pack as _pack
from . import planning as _planning
from . import promotion as _promotion
from . import readiness as _readiness
from . import review as _review
from . import staging as _staging
from . import states as _states
from . import workbook as _workbook
from .adapters import (
    ActivationPreconditions,
    ExecutionAdapter,
    LiveExecutionAdapter,
    SyntheticExecutionAdapter,
)
from .artefacts import ArtefactService, RoleReadiness, sample_manifest
from .derive import ExecutionFacts
from .execution import (
    DECISION_MAPPING_PROPOSAL,
    MATERIALITY_MARK,
    SyntheticOnboardingAdapters,
    run_synthetic_orchestration,
)
from .execution import mapping_key as _mapping_key
from .input_roles import artefact_vocabulary
from .interpretation import (
    PROV_AGENT,
    PROV_APPROVED,
    PROV_ARTEFACT,
    PROV_CLIENT,
    PROV_HUMAN,
    PROV_INHERITED,
    DeterministicInterpreter,
    Interpretation,
    InterpretationError,
    Interpreter,
    ProposedChange,
)
from .planning import ApplicationPlan
from .policy import (
    CAP_ACTIVATE_CONFIGURATION,
    CAP_EXTERNAL_EMAIL,
    SyntheticPolicy,
    synthetic_policy,
    validate_segment,
)
from .run import (
    ACTOR_AGENT,
    ACTOR_HUMAN,
    ACTOR_SYSTEM,
    EXEC_BLOCKED,
    EXEC_DETERMINISTIC,
    EXEC_HUMAN_CONFIRMED,
    EXEC_MODEL_PROPOSED,
    EXEC_SIMULATED,
    EXEC_SYNTHETICALLY_EXECUTED,
    STAGE_DETERMINISTIC_COMPLETED,
    STAGE_HARD_BLOCKED,
    Message,
    SyntheticRun,
)
from .store import SyntheticRunStore, synthetic_ops_store

logger = logging.getLogger("trakt.operations_control.occ_agent")

#: Each lifecycle action in the words an operator would use for it. The state
#: table names actions for the machine (``run_synthetic_onboarding``); an
#: operator asking what they can do next should be told in their own language,
#: and an action with no phrase here is rendered from its key rather than
#: silently dropped.
ACTION_PHRASES: Dict[str, str] = {
    _states.ACTION_ANSWER: "answer an onboarding question",
    _states.ACTION_REQUEST_INFORMATION: "ask the client for what is missing",
    _states.ACTION_RECORD_RESPONSE: "record what the client sent back",
    _states.ACTION_SUBMIT_FOR_APPROVAL: "submit the onboarding for approval",
    _states.ACTION_APPROVE_ONBOARDING: "approve the onboarding",
    _states.ACTION_REQUEST_CHANGES: "send the onboarding back for changes",
    _states.ACTION_WITHDRAW: "withdraw the onboarding",
    _states.ACTION_DRAFT_PACK: "draft the client pack",
    _states.ACTION_APPROVE_PACK: "approve the pack to send",
    _states.ACTION_SEND_PACK: "issue the pack to the client",
    _states.ACTION_REGISTER_ARTEFACT: "provide a file",
    _states.ACTION_RUN_ONBOARDING: "run the practice onboarding",
    _states.ACTION_RESOLVE_DECISION: "settle an open decision",
    _states.ACTION_ACKNOWLEDGE_EXCEPTION: "acknowledge an exception",
    _states.ACTION_GENERATE_PLAN: "generate the orchestration plan",
    _states.ACTION_APPROVE_EXECUTION: "approve readiness for execution",
    _states.ACTION_REQUEST_ACTIVATION: "request activation",
    _states.ACTION_APPROVE_ACTIVATION: "approve activation",
    _states.ACTION_CONFIRM_ACTIVATION: "confirm activation",
    _states.ACTION_CANCEL: "cancel the practice case",
}

#: Actions that exist but are never the answer to "what should I do next" —
#: they undo or abandon, and offering them as the way forward is noise.
_NOT_A_WAY_FORWARD = (_states.ACTION_CANCEL, _states.ACTION_WITHDRAW,
                      _states.ACTION_REQUEST_CHANGES)

#: A decision a human has ANSWERED, and which a rerun must therefore keep
#: rather than ask again. Written deliberately by the three paths that settle
#: one: ``resolve_decision`` sets approved or rejected, ``acknowledge_exception``
#: sets acknowledged, and the mapping commit sets approved.
#:
#: Everything else is re-raisable. That is deliberate and it is the safe
#: direction — the worst case is a question asked twice, against a decision
#: nobody can answer and no deploy can dislodge, because the value that froze
#: it lives in the case's own document rather than in the code.
_SETTLED_STATUSES = ("approved", "rejected", "acknowledged")


def action_phrase(action: str) -> str:
    return ACTION_PHRASES.get(action, str(action).replace("_", " "))


def _join(parts: List[str]) -> str:
    """"a", "a or b", "a, b or c" — a list a person would read aloud."""
    parts = [p for p in parts if p]
    if len(parts) <= 1:
        return parts[0] if parts else ""
    return ", ".join(parts[:-1]) + " or " + parts[-1]


class ActionNotAllowed(OpsError):
    """The action is not available from the run's current state."""

    def __init__(self, action: str, state: str):
        super().__init__(
            "OCC_AGENT_ACTION_NOT_ALLOWED",
            f"'{action.replace('_', ' ')}' is not something you can do while "
            f"this practice case is at {_states.spec_label(state)}.",
            http_status=409)


class PartiallyUnderstood(OpsError):
    """Part of an instruction could not be read, so none of it was applied.

    The whole point of the exception is what it carries: the plan's disclosure,
    so the caller can show what WAS understood, what was not, and what the agent
    would like to ask — and then confirm the disclosed remainder if they choose
    to. Nothing is written in the meantime.
    """

    def __init__(self, plan: ApplicationPlan):
        self.plan = plan
        self.disclosure = plan.disclosure()
        detail: List[str] = []
        if plan.unrecognised:
            detail.append("I could not read: "
                          + "; ".join(f'"{u}"' for u in plan.unrecognised[:3]))
        if plan.questions:
            detail.append(" ".join(q.question for q in plan.questions[:2]))
        super().__init__(
            "OCC_AGENT_PARTIALLY_UNDERSTOOD",
            "Nothing was applied. " + " ".join(detail),
            http_status=422)


@dataclass
class AgentCase:
    """One practice case: the onboarding case and the run that sits beside it.

    Two records, never merged. The case is the platform's; the run is this
    feature's. Presenting them together is a view, not a third model.
    """

    case: OnboardingCase
    run: SyntheticRun

    @property
    def case_ref(self) -> str:
        return self.case.case_id


@dataclass
class TurnResult:
    """What one natural-language turn produced."""

    case: AgentCase
    reply: str = ""
    proposal: Optional[Dict[str, Any]] = None
    applied: bool = False
    decisions: List[Dict[str, Any]] = field(default_factory=list)


class OccAgentService:
    """Every OCC Agent operation, in one injectable service."""

    def __init__(self, storage: Storage, *,
                 container: Optional[str] = None,
                 sandbox: Optional[Path] = None,
                 policy: Optional[SyntheticPolicy] = None,
                 interpreter: Optional[Interpreter] = None,
                 store: Optional[SyntheticRunStore] = None,
                 onboarding: Optional[OnboardingService] = None,
                 adapter: Optional[ExecutionAdapter] = None,
                 communication: Optional[_comms.CommunicationAdapter] = None,
                 engine: Optional[Any] = None,
                 live_onboarding: Optional[OnboardingService] = None):
        self.store = store or SyntheticRunStore(storage, container=container,
                                                sandbox=sandbox)
        # The onboarding service, pinned to the synthetic container. Everything
        # it writes — cases, versions, artefacts — lands there and nowhere near
        # the live operations container.
        self.onboarding = onboarding or OnboardingService(
            synthetic_ops_store(storage, self.store.container))
        # The governed side of the doorway: the SAME onboarding service against
        # the real operations container. Nothing reaches it until an operator
        # confirms activation, and then only through
        # `operations_control.occ_agent.promotion`. Built only where live
        # execution is switched on, so an ordinary rehearsal deployment never
        # even holds a handle to the governed store.
        self.live_onboarding = live_onboarding
        if self.live_onboarding is None and _adapters.live_enabled():
            self.live_onboarding = OnboardingService(
                OpsStore(storage, OpsLayout.from_env()))
        self.policy = policy or synthetic_policy(audit_sink=self._audit_refusal)
        self.interpreter = interpreter or DeterministicInterpreter(
            cat=self.onboarding.catalogue)
        self.artefacts = ArtefactService(self.store, self.policy)
        # How the pack reaches a client, and what happens at activation. Both
        # are seams; the workflow above them does not change with either.
        self.communication = communication or _comms.default_adapter(
            self.policy)
        self.adapter: ExecutionAdapter = adapter or self._default_adapter(
            engine)
        # Practice runs execute on their own thread, as live workflows already
        # do (operations_control.engine.Engine.start), so a pipeline pass never
        # occupies an API request.
        self._jobs: Dict[str, threading.Thread] = {}
        self._jobs_lock = threading.Lock()

    def _default_adapter(self, engine: Optional[Any]) -> ExecutionAdapter:
        """Rehearsal unless this environment explicitly enables live mode.

        The live adapter is still constructed when the flag is on, so the
        refusal an operator sees comes from the one activation gate rather than
        from a missing object.

        It is given the GOVERNED onboarding service, not the practice one. That
        is the point of the doorway: the case is authored in the practice
        container and promoted across at activation, so the configuration the
        regulatory return reads is built by production, in production.
        """
        if (_adapters.live_enabled() and engine is not None
                and self.live_onboarding is not None):
            return LiveExecutionAdapter(self.live_onboarding, engine)
        return SyntheticExecutionAdapter(self.policy)

    # ------------------------------------------------------------------ #
    # Audit plumbing
    # ------------------------------------------------------------------ #
    def _audit_refusal(self, event: Dict[str, Any]) -> None:
        """Sink for policy refusals.

        A refusal can happen with no case in hand (a misconfigured call), so it
        is logged when there is no case to file it against — never silently
        discarded. The capability is named; nothing about the case is.
        """
        case_ref = str(event.get("case_id") or "")
        tenant = str(event.get("tenant") or "")
        if not case_ref or not tenant:
            logger.warning("occ_agent: refused %s with no case to file it "
                           "against", event.get("capability"))
            return
        self.store.append_audit(
            tenant, case_ref, action=str(event.get("action") or "refused"),
            actor_type=ACTOR_SYSTEM,
            actor_identity=str(event.get("actor_identity") or ""),
            decision_basis=str(event.get("decision_basis") or ""),
            execution_classification=EXEC_BLOCKED,
            detail={"capability": event.get("capability"),
                    "detail": event.get("detail")})

    def _audit(self, run: SyntheticRun, action: str, *,
               actor_type: str = ACTOR_SYSTEM, actor: str = "",
               prior_state: str = "", decision_basis: str = "",
               classification: str = EXEC_DETERMINISTIC,
               input_reference: str = "", output_reference: str = "",
               detail: Optional[Dict[str, Any]] = None) -> None:
        self.store.append_audit(
            run.tenant, run.case_ref, action=action, actor_type=actor_type,
            actor_identity=actor, prior_state=prior_state or run.state,
            resulting_state=run.state, decision_basis=decision_basis,
            execution_classification=classification,
            input_reference=input_reference, output_reference=output_reference,
            detail=detail or {})

    def _move(self, run: SyntheticRun, to_state: str) -> str:
        """Assert and apply a lifecycle transition. Returns the prior state."""
        prior = run.state
        _states.assert_transition(prior, to_state)
        run.state = to_state
        return prior

    @staticmethod
    def _require_action(run: SyntheticRun, action: str) -> None:
        if not _states.action_allowed(run.state, action):
            raise ActionNotAllowed(action, run.state)

    # ------------------------------------------------------------------ #
    # Opening and loading
    # ------------------------------------------------------------------ #
    def create_case(self, *, tenant: str, initiating_user: str,
                    instruction: str = "",
                    fixture_id: str = "", live: bool = False,
                    amend_client: str = "") -> AgentCase:
        """Open a case — a rehearsal unless ``live`` is asked for explicitly.

        ``amend_client`` opens an AMENDMENT to that client's active
        configuration instead of a new onboarding. Everything downstream is
        identical: the same rehearsal, the same controls, the same two
        approvals, the same one doorway. Only the starting answers differ, and
        where they came from is recorded on the case.

        The onboarding case is opened by Client Onboarding itself — same
        reference series, same blank start, same event history — and the run
        record is created beside it.

        A live case is authored in exactly the same isolated container as a
        rehearsal; ``live`` marks where it is ALLOWED to end, not where it is
        written. The single difference arrives at activation, where a live case
        crosses into the governed store through
        :mod:`~operations_control.occ_agent.promotion`. It is a parameter and
        never a default because a case that turns out to be real by accident is
        the failure this whole boundary exists to prevent.
        """
        validate_segment(tenant, "tenant")
        if live and not _adapters.live_enabled():
            raise OpsError(
                "OPS_LIVE_NOT_ENABLED",
                "Live execution is not switched on in this environment, so a "
                "live case cannot be opened here.", 409)
        if amend_client:
            # A CHANGE TO A CLIENT ALREADY LIVE, not a second onboarding of
            # them. Client Onboarding opens it pre-populated from the version
            # in force and records which version it started from, so the
            # change is reviewable as a difference rather than as a fresh set
            # of answers that happen to mostly match.
            #
            # This is the supported way to add a reporting product after
            # activation. Editing a live case in conversation does not reach
            # the source registry, where `regime_required` is written at
            # activation — so the book stays registered as it was, and the
            # engine refuses the delivery rather than splitting it across two
            # incomplete ones. An amendment re-activates, and the registry is
            # rewritten with it.
            case = self.onboarding.start_amendment(client_id=amend_client,
                                                   by=initiating_user)
        else:
            case = self.onboarding.start_new_client(by=initiating_user)
        run = SyntheticRun(case_ref=case.case_id, tenant=tenant,
                           initiating_user=initiating_user,
                           fixture_id=fixture_id,
                           mode=_adapters.MODE_LIVE if live
                           else _adapters.MODE_SYNTHETIC)
        self.store.save(run)
        self._audit(run, "practice_case_opened", actor_type=ACTOR_HUMAN,
                    actor=initiating_user,
                    decision_basis="an operator opened a practice case",
                    output_reference=case.case_id)
        agent_case = AgentCase(case=case, run=run)
        if instruction:
            agent_case = self.answer_from_instruction(
                agent_case, instruction=instruction, actor=initiating_user)
        return agent_case

    def load(self, tenant: str, case_ref: str) -> AgentCase:
        run = self.store.load(tenant, case_ref)
        return AgentCase(case=self.onboarding.load_case(run.case_ref), run=run)

    def list_cases(self, tenant: str, *,
                   state: Optional[str] = None) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for row in self.store.list_runs(tenant):
            if state and row.get("state") != state:
                continue
            try:
                case = self.onboarding.load_case(str(row.get("case_ref") or ""))
            except OpsError:
                # A run whose onboarding case has gone is shown as what it is
                # rather than hidden: the operator needs to see the orphan.
                rows.append({**row, "onboarding_status": "",
                             "client_name": "", "onboarding_missing": True})
                continue
            rows.append({
                **row,
                "onboarding_status": case.status,
                "onboarding_status_label": STATUS_LABELS.get(case.status,
                                                             case.status),
                "client_id": case.client_id,
                "client_name": case.client_name or case.client_id
                or "Not yet named",
                "onboarding_missing": False,
            })
        return rows

    def facts(self, agent_case: AgentCase) -> ExecutionFacts:
        return _derive.facts(agent_case.case,
                             portfolio_id=agent_case.run.portfolio_id,
                             dataset=agent_case.run.dataset,
                             cat=self.onboarding.catalogue)

    # ------------------------------------------------------------------ #
    # The onboarding half — every one of these delegates
    # ------------------------------------------------------------------ #
    def plan_instruction(self, agent_case: AgentCase, *,
                         instruction: str) -> ApplicationPlan:
        """What an instruction WOULD do. Writes nothing.

        This is the honest half of the conversation: it reports what was
        understood, what it proposes, what it could not read, and what it
        cannot place without being told — before anything is applied.
        """
        interpretation = self.interpreter.interpret_instruction(instruction)
        interpretation.validate(self.onboarding.catalogue)
        return self.plan_interpretation(agent_case, interpretation)

    def plan_interpretation(self, agent_case: AgentCase,
                            interpretation: Interpretation) -> ApplicationPlan:
        plan = _planning.plan_changes(
            agent_case.case, self.onboarding.catalogue,
            steps=interpretation.steps,
            provenance=interpretation.provenance,
            confidence=interpretation.confidence,
            set_changes=interpretation.set_changes)
        plan.unrecognised.extend(interpretation.unrecognised)
        plan.reporting_period = interpretation.reporting_period
        plan.streams = list(interpretation.streams)
        plan.expected_artefacts = list(interpretation.expected_artefacts)
        if interpretation.delivery:
            plan.cadence = str(interpretation.delivery.get("cadence") or "")
            plan.steps["_delivery"] = dict(interpretation.delivery)
        if interpretation.stream_delivery:
            plan.stream_cadence = {
                stream: str(payload.get("cadence") or "")
                for stream, payload in interpretation.stream_delivery.items()
                if payload.get("cadence")}
            plan.steps["_stream_delivery"] = {
                stream: dict(payload) for stream, payload
                in interpretation.stream_delivery.items()}
        return plan

    def answer_from_instruction(self, agent_case: AgentCase, *,
                                instruction: str, actor: str,
                                confirm: bool = True) -> AgentCase:
        """Turn one instruction into answers on the onboarding case."""
        plan = self.plan_instruction(agent_case, instruction=instruction)
        agent_case.run.messages.append(
            Message(role="operator", text=instruction[:4000]).to_dict())
        return self.apply_plan(agent_case, plan=plan, actor=actor,
                               confirm=confirm)

    def apply_plan(self, agent_case: AgentCase, *, plan: ApplicationPlan,
                   actor: str, confirm: bool = False) -> AgentCase:
        """Write a plan onto the case, step by step, through ``save_step``.

        A plan that could not be fully read is refused unless the human has
        explicitly confirmed the disclosed remainder. Nothing is ever applied
        in part without that being said first.
        """
        if not plan.complete and not confirm:
            raise PartiallyUnderstood(plan)

        run = agent_case.run
        case = agent_case.case
        delivery = dict((plan.steps or {}).pop("_delivery", {}) or {})
        per_stream = dict((plan.steps or {}).pop("_stream_delivery", {}) or {})
        written: List[str] = []
        for step in STEPS:
            payload = (plan.steps or {}).get(step)
            if not payload:
                continue
            case = self.onboarding.save_step(case_id=case.case_id, step=step,
                                             payload=payload, by=actor)
            written.append(step)
        if "pipeline" in plan.streams:
            # A declared pipeline stream becomes its own source registration
            # beside the funded one Client Onboarding derives for every book —
            # through the onboarding service's own writer, never a second path.
            pid = next((str(p.get("portfolio_id") or "")
                        for p in case.items("portfolios")
                        if p.get("portfolio_id")), "")
            if pid:
                case = self.onboarding.add_pipeline_source(
                    case_id=case.case_id, portfolio_id=pid, by=actor)
                written.append("pipeline_book")
        if delivery or per_stream:
            case = self._apply_delivery(case, delivery, actor,
                                        per_stream=per_stream)
            written.append("sources")
        if plan.reporting_period:
            run.reporting_period = plan.reporting_period

        agent_case.case = self._record_provenance(case, plan, actor,
                                                  confirmed=confirm)
        self._reserve_identifiers(agent_case)
        run.facts = self.facts(agent_case).to_dict()
        self.store.save(run)
        self._audit(run, "onboarding_answered", actor_type=ACTOR_AGENT,
                    actor=actor, classification=EXEC_MODEL_PROPOSED,
                    decision_basis="structured answers read from the "
                                   "instruction and written through Client "
                                   "Onboarding",
                    detail={"steps": written,
                            "changes": plan.change_count,
                            "unrecognised": len(plan.unrecognised),
                            "reporting_period": run.reporting_period})
        run.messages.append(Message(
            role="agent",
            text=self.describe_plan(agent_case, plan)).to_dict())
        self.store.save(run)
        return agent_case

    def _record_provenance(self, case: OnboardingCase, plan: ApplicationPlan,
                           actor: str, *,
                           confirmed: bool = False) -> OnboardingCase:
        """Keep where each answer came from, on the case that holds it.

        Client Onboarding already carries a per-field provenance map and shows
        it in its own screens, so the agent's answers appear there beside a
        migrated client's — rather than living only in a transient
        interpretation the operator never sees again.

        A value the agent could only *propose* — it was not certain which item
        the operator meant — is recorded as human-approved once they confirm
        it, so the record distinguishes "the client said so" from "the agent
        guessed and a person agreed".

        Two maps are written, because the case already has two questions to
        answer: ``provenance`` is the sentence its own screens show, and
        ``provenance_class`` is the category a control can act on.
        """
        changed = [c for c in plan.understood
                   if c.action != _planning.UNCHANGED]
        if not changed:
            return case
        for change in changed:
            path = f"{change.section}.{change.field}"
            if change.index is not None:
                path = f"{change.section}[{change.index}].{change.field}"
            uncertain = (change.confidence is not None
                         and change.confidence < 1.0)
            if confirmed and uncertain:
                category = PROV_APPROVED
            else:
                category = change.provenance or PROV_CLIENT
            case.provenance_class[path] = category
            case.provenance[path] = _PROVENANCE_SENTENCE.get(category, category)
        case.record("provenance_recorded", actor=actor,
                    detail={"fields": sorted(
                        f"{c.section}.{c.field}" for c in changed)})
        self.onboarding.cases.save_case(case)
        return case

    def _reserve_identifiers(self, agent_case: AgentCase) -> None:
        """Claim the client identifier this case intends to use.

        Nothing is activated in a rehearsal, so Client Onboarding's own
        collision check cannot see one practice case from another. The
        reservation closes that gap inside the practice container.
        """
        client_id = agent_case.case.client_id
        if not client_id:
            return
        clash = self.store.reserve_identifier(
            "client_id", client_id, case_ref=agent_case.case_ref,
            tenant=agent_case.run.tenant)
        if clash:
            note = (f"'{client_id}' is already claimed by practice case "
                    f"{clash.get('case_ref')}.")
            if note not in agent_case.run.observations:
                agent_case.run.observations.append(note)

    def describe_plan(self, agent_case: AgentCase,
                      plan: ApplicationPlan) -> str:
        """What the agent did, in the catalogue's own labels."""
        lines: List[str] = []
        applied = [c.sentence() for c in plan.understood
                   if c.action != _planning.UNCHANGED]
        if applied:
            lines.append("Recorded:")
            lines += [f"- {line}" for line in applied]
        if plan.streams:
            lines.append("Streams registered separately:")
            lines += [f"- {_derive.stream_sentence(stream)}"
                      for stream in plan.streams]
        if plan.reporting_period:
            lines.append(f"- Reporting period: {plan.reporting_period}")
        uncertain = plan.uncertain
        if uncertain:
            # Named separately from what was read with certainty. An operator
            # scanning a wall of "Recorded:" lines has no way to tell which one
            # Trakt guessed at, and the guess is the line worth reading.
            lines.append("Read, but not certain — please check:")
            lines += [f"- {change.sentence()}" for change in uncertain]
        if plan.unrecognised:
            lines.append("I could not read:")
            lines += [f"- \"{fragment}\"" for fragment in plan.unrecognised]
        for question in plan.questions:
            lines.append(f"- {question.question}")
        outstanding = self.onboarding.client_checklist(agent_case.case)
        if outstanding:
            lines.append("Still needed from the client:")
            lines += [f"- {row['label']}" for row in outstanding[:8]]
        return "\n".join(lines) or "Nothing changed."

    def _apply_delivery(self, case: OnboardingCase, values: Dict[str, Any],
                        actor: str, *,
                        per_stream: Optional[Dict[str, Any]] = None
                        ) -> OnboardingCase:
        """Apply delivery answers to the deliveries Trakt has derived.

        Deliveries are derived from the portfolios, so "they send monthly by
        SFTP" is a statement about all of them, not about one — and that is
        what ``values`` carries.

        ``per_stream`` is what a single registration said about ITSELF: "a
        weekly pipeline" answers for the pipeline and for nothing else. It is
        applied after the blanket and overrides it, because a statement about
        one book is more specific than a statement about every book. Before
        this existed, one cadence field held both answers and every
        registration got whichever was read last.
        """
        sources = [dict(s) for s in case.items("sources")]
        if not sources:
            return case
        # A source's identity — which book, which dataset — is never a blanket
        # answer. "They deliver monthly" applies to every registration;
        # "a funded book" names ONE stream and is handled as one.
        blanket = {k: v for k, v in (values or {}).items()
                   if k not in ("dataset", "portfolio_id", "source_key")}
        for source in sources:
            source.update({k: v for k, v in blanket.items() if v not in
                           (None, "", [])})
            own = (per_stream or {}).get(str(source.get("dataset") or ""))
            if own:
                source.update({k: v for k, v in own.items()
                               if k not in ("dataset", "portfolio_id",
                                            "source_key")
                               and v not in (None, "", [])})
        return self.onboarding.save_step(case_id=case.case_id, step="sources",
                                         payload={"sources": sources}, by=actor)

    def request_client_information(self, agent_case: AgentCase, *, actor: str,
                                   items: Optional[List[Dict[str, Any]]] = None,
                                   due_date: str = "",
                                   note: str = "") -> AgentCase:
        """Ask the client for what the catalogue says is still outstanding.

        The checklist is Client Onboarding's — restricted to client-supplied
        fields, so it is something that could actually be sent — and the request
        is its own record. Nothing is emailed: sending is a prohibited
        capability, and :meth:`send_request_by_email` names it.
        """
        case = agent_case.case
        chosen = items if items is not None else \
            self.onboarding.client_checklist(case)
        if not chosen:
            raise OpsError(
                "OCC_AGENT_NOTHING_OUTSTANDING",
                "There is nothing outstanding to ask the client for.",
                http_status=409)
        agent_case.case = self.onboarding.create_request(
            case_id=case.case_id, items=chosen, by=actor,
            responsible_party="client", due_date=due_date, note=note)
        self._audit(agent_case.run, "client_information_requested",
                    actor_type=ACTOR_HUMAN, actor=actor,
                    classification=EXEC_HUMAN_CONFIRMED,
                    decision_basis="the outstanding client checklist was "
                                   "turned into an information request",
                    detail={"items": len(chosen)})
        return agent_case

    def clarifications_available(self, agent_case: AgentCase
                                 ) -> Dict[str, Any]:
        """Phase two: what a delivered file has now made askable.

        Read-only. Returns the deferred questions that are still outstanding,
        with the evidence to put them against, plus a short account of what was
        analysed. Empty until a file has arrived.
        """
        from . import clarification as _clarification
        artefacts = agent_case.run.received_artefacts
        items = _clarification.outstanding(agent_case.case, artefacts,
                                           cat=self.onboarding.catalogue)
        return {"items": items,
                "summary": _clarification.summary(items, artefacts)}

    def request_clarifications(self, agent_case: AgentCase, *, actor: str,
                               note: str = "", due_date: str = "") -> AgentCase:
        """Ask the deferred questions, now that there is a file to ask about.

        Deliberately routed through :meth:`request_client_information` rather
        than creating a request of its own: the deferred questions are ordinary
        information requests, and they get the same record, the same status
        transitions and the same audit as an operator's. There is no second
        channel to a client, and no gate is skipped.
        """
        available = self.clarifications_available(agent_case)
        items = available["items"]
        if not items:
            raise OpsError(
                "OCC_AGENT_NOTHING_TO_CLARIFY",
                "There is nothing to clarify yet. Trakt asks these once it has "
                "a file to ask about, and either no file has arrived or it "
                "answered everything itself.", http_status=409)
        agent_case = self.request_client_information(
            agent_case, actor=actor, items=items, due_date=due_date,
            note=note or "Clarification after the first delivery.")
        self._audit(agent_case.run, "clarifications_requested",
                    actor_type=ACTOR_HUMAN, actor=actor,
                    classification=EXEC_HUMAN_CONFIRMED,
                    decision_basis="the questions the catalogue defers until a "
                                   "representative file has been analysed",
                    detail=available["summary"])
        return agent_case

    def send_request_by_email(self, agent_case: AgentCase, *,
                              actor: str) -> None:
        """The external-email seam. Always refused in synthetic mode."""
        self.policy.require(CAP_EXTERNAL_EMAIL, detail="information request",
                            case_id=agent_case.case_ref,
                            tenant=agent_case.run.tenant, actor=actor)

    def record_client_response(self, agent_case: AgentCase, *, request_id: str,
                               actor: str,
                               answers: Optional[Dict[str, Any]] = None,
                               note: str = "",
                               accept: bool = True) -> AgentCase:
        """Record what the client sent back, and accept or reject it."""
        case_id = agent_case.case_ref
        case = agent_case.case
        request = case.request(request_id)
        if request is None:
            raise OpsError("OCC_AGENT_REQUEST_NOT_FOUND",
                           "That information request could not be found.",
                           http_status=404)
        if request.status == "open":
            case = self.onboarding.mark_request_sent(
                case_id=case_id, request_id=request_id, by=actor)
        case = self.onboarding.record_response(
            case_id=case_id, request_id=request_id, by=actor, note=note,
            answers=answers or {})
        agent_case.case = self.onboarding.review_response(
            case_id=case_id, request_id=request_id, accept=accept, by=actor,
            note=note)
        if accept and answers:
            # The fields the request asked for, not every key in the payload:
            # a repeatable answer carries the whole row back.
            asked = [
                (f"{item.get('section')}[{item.get('index')}]."
                 f"{item.get('field')}" if item.get("index") is not None
                 else f"{item.get('section')}.{item.get('field')}")
                for item in (request.items or [])]
            agent_case.case = self._mark_client_supplied(agent_case.case,
                                                          asked, actor)
        self._audit(agent_case.run, "client_response_recorded",
                    actor_type=ACTOR_HUMAN, actor=actor,
                    classification=EXEC_HUMAN_CONFIRMED,
                    input_reference=request_id,
                    decision_basis=("the operator recorded and accepted the "
                                    "client's response" if accept else
                                    "the operator rejected the client's "
                                    "response"),
                    detail={"sections": sorted(answers or {})})
        return agent_case

    def _mark_client_supplied(self, case: OnboardingCase,
                              paths: List[str],
                              actor: str) -> OnboardingCase:
        """Record that these SPECIFIC values came from the client.

        Client Onboarding accepts the response; what it has no way to know is
        that the answers arrived from the client rather than from an operator.
        That distinction is exactly what an approver needs, so it is written to
        the case's own provenance map rather than kept here.

        Takes explicit paths rather than a payload. Writing a repeatable section
        means sending the whole row back, derived values included, so inferring
        the answered fields from the payload marked everything in the row as the
        client's — including a currency Trakt worked out and an entity link it
        minted. An approver reading "the client told Trakt" against a value the
        client never saw is worse than no provenance at all.
        """
        recorded: List[str] = []
        for path in paths:
            section_key, _, rest = path.partition(".")
            section_key = section_key.split("[")[0]
            section = self.onboarding.catalogue.section(section_key)
            if section is None or section.field(rest) is None:
                continue
            case.provenance_class[path] = PROV_CLIENT
            case.provenance[path] = _PROVENANCE_SENTENCE[PROV_CLIENT]
            recorded.append(path)
        if recorded:
            case.record("provenance_recorded", actor=actor,
                        detail={"fields": sorted(recorded),
                                "provenance": PROV_CLIENT})
            self.onboarding.cases.save_case(case)
        return case

    def submit_for_approval(self, agent_case: AgentCase, *,
                            actor: str) -> AgentCase:
        agent_case.case = self.onboarding.submit_for_approval(
            case_id=agent_case.case_ref, by=actor)
        self._audit(agent_case.run, "onboarding_submitted_for_approval",
                    actor_type=ACTOR_HUMAN, actor=actor,
                    classification=EXEC_HUMAN_CONFIRMED,
                    decision_basis="the onboarding reported itself ready")
        return agent_case

    def request_changes(self, agent_case: AgentCase, *, actor: str,
                        reason: str) -> AgentCase:
        agent_case.case = self.onboarding.request_changes(
            case_id=agent_case.case_ref, by=actor, reason=reason)
        self._audit(agent_case.run, "onboarding_changes_requested",
                    actor_type=ACTOR_HUMAN, actor=actor,
                    classification=EXEC_HUMAN_CONFIRMED,
                    decision_basis=reason)
        return agent_case

    def approve_onboarding(self, agent_case: AgentCase, *, actor: str,
                           reason: str = "") -> AgentCase:
        """Approve the onboarding. Records the decision; writes nothing."""
        agent_case.case = self.onboarding.approve(
            case_id=agent_case.case_ref, by=actor,
            reason=reason or "Approved in a practice case.")
        run = agent_case.run
        # Approving the onboarding releases the execution half, wherever the
        # pack has got to. A case whose pack is still in draft is not held
        # back: issuing one is an option, not a precondition.
        if run.state in (_states.AWAITING_ONBOARDING,) + _states.PACK_STATES \
                and _states.is_transition_allowed(run.state,
                                                  _states.READY_TO_RUN):
            self._move(run, _states.READY_TO_RUN)
        run.facts = self.facts(agent_case).to_dict()
        self.store.save(run)
        self._audit(run, "onboarding_approved", actor_type=ACTOR_HUMAN,
                    actor=actor, classification=EXEC_HUMAN_CONFIRMED,
                    decision_basis="the operator approved the onboarding; no "
                                   "configuration was created")
        return agent_case

    def activate(self, agent_case: AgentCase, *, actor: str) -> None:
        """The configuration-write seam. Always refused in synthetic mode.

        ``OnboardingService.activate()`` is, by its own docstring, "the only
        place active configuration is created". Naming it as a capability makes
        that the exact line a practice case never crosses — and exercising the
        refusal here means the guarantee is tested rather than asserted.
        """
        self.policy.require(CAP_ACTIVATE_CONFIGURATION,
                            detail=f"onboarding case {agent_case.case_ref}",
                            case_id=agent_case.case_ref,
                            tenant=agent_case.run.tenant, actor=actor)

    def preview(self, agent_case: AgentCase) -> Dict[str, Any]:
        """Exactly what activation would create. Writes nothing."""
        return self.onboarding.preview(agent_case.case)

    def onboarding_readiness(self, agent_case: AgentCase) -> Dict[str, Any]:
        return self.onboarding.readiness(agent_case.case)

    # ------------------------------------------------------------------ #
    # The client pack — DRAFTED → HUMAN_REVIEW_REQUIRED → APPROVED_TO_SEND
    #                                                   → SENT
    # ------------------------------------------------------------------ #
    def build_pack(self, agent_case: AgentCase) -> _pack.OnboardingPack:
        """The pack this case would issue. Writes nothing."""
        return _pack.build(agent_case.case, cat=self.onboarding.catalogue,
                           outcome=self.facts(agent_case).outcome,
                           reporting_period=agent_case.run.reporting_period)

    def client_form(self, agent_case: AgentCase) -> _client_form.ClientForm:
        """The structured form this client should see now. Writes nothing."""
        return _client_form.build(agent_case.case,
                                  cat=self.onboarding.catalogue)

    def classify_case(self, agent_case: AgentCase) -> Dict[str, Any]:
        """Every catalogue field, in one of the five categories.

        The operator's answer to "why is the client not being asked that?" —
        and the evidence that only category 2 ever reaches one.
        """
        rows = _classification.classify_all(
            agent_case.case.answers, cat=self.onboarding.catalogue,
            provenance=agent_case.case.provenance_class)
        return {"summary": _classification.summarise(rows),
                "fields": [r.to_dict() for r in rows]}

    def submit_client_response(self, agent_case: AgentCase, *, actor: str,
                               response: Dict[str, Any],
                               request_id: str = "",
                               strict: bool = True) -> AgentCase:
        """Persist a structured client response. Deterministic, end to end.

        Every key is an authoritative catalogue key, checked against the form
        the client was actually served. Values are written through
        ``OnboardingService.save_step`` exactly as submitted — no interpretation
        happens between the control and the case, and a test asserts this path
        never reaches the interpreter.
        """
        case = agent_case.case
        form = self.client_form(agent_case)
        plan = _client_form.plan_response(
            case, response, cat=self.onboarding.catalogue, form=form,
            strict=strict)
        problems = _client_form.validate_response(
            case, plan, cat=self.onboarding.catalogue)
        if problems:
            raise OpsError(
                "OCC_AGENT_RESPONSE_INVALID",
                "Some answers could not be accepted: "
                + "; ".join(f"{p['label']}: {p['message']}"
                            for p in problems[:3]),
                http_status=400)

        for step in STEPS:
            payload = plan.steps.get(step)
            if not payload:
                continue
            case = self.onboarding.save_step(case_id=case.case_id, step=step,
                                             payload=payload, by=actor)
        agent_case.case = self._mark_client_supplied(
            case, sorted(plan.accepted), actor)
        if request_id:
            agent_case = self._close_request(agent_case, request_id, actor)
        else:
            agent_case = self._close_answered_requests(agent_case, actor)
        self._audit(agent_case.run, "client_response_submitted",
                    actor_type=ACTOR_HUMAN, actor=actor,
                    classification=EXEC_DETERMINISTIC,
                    input_reference=request_id,
                    decision_basis="structured answers written straight "
                                   "through Client Onboarding; nothing was "
                                   "interpreted",
                    detail={"answers": len(plan.accepted),
                            "ignored": plan.ignored,
                            "form_hash": form.content_hash})
        return agent_case

    def record_concentration_outcome(self, agent_case: AgentCase, *,
                                     actor: str, status: str,
                                     response_text: str = "",
                                     reason: str = "") -> AgentCase:
        """Record the operator's decision on the concentration-test request.

        The request is mandatory at the onboarding-control level: approval is
        blocked while it sits at ``pending_client_response``, and only an
        operator moves it — to ``supplied`` (with the client's actual
        response), ``not_applicable`` or ``deferred_with_reason`` (each with a
        reason). A blank answer can never be recorded as supplied; the
        catalogue's ``required_when`` and the structural validation rule both
        refuse it.
        """
        allowed = ("supplied", "not_applicable", "deferred_with_reason",
                   "pending_client_response")
        if status not in allowed:
            raise OpsError("OCC_AGENT_INVALID_STATUS",
                           f"'{status}' is not a concentration-test status.",
                           http_status=400)
        if status == "supplied" and not response_text.strip():
            raise OpsError("OCC_AGENT_BLANK_SUPPLIED",
                           "A blank answer cannot be recorded as supplied.",
                           http_status=400)
        payload: Dict[str, Any] = {"concentration_tests_status": status}
        if response_text.strip():
            payload["concentration_tests"] = response_text
        if reason.strip():
            payload["concentration_tests_status_reason"] = reason
        agent_case.case = self.onboarding.save_step(
            case_id=agent_case.case_ref, step="risk_limits", payload=payload,
            by=actor)
        if response_text.strip():
            agent_case.case = self._mark_client_supplied(
                agent_case.case, ["risk_limits.concentration_tests"], actor)
        self._audit(agent_case.run, "concentration_outcome_recorded",
                    actor_type=ACTOR_HUMAN, actor=actor,
                    classification=EXEC_DETERMINISTIC,
                    decision_basis=f"operator recorded the request as "
                                   f"{status}",
                    detail={"status": status, "reason": reason,
                            "response_chars": len(response_text)})
        return agent_case

    def _close_answered_requests(self, agent_case: AgentCase,
                                 actor: str) -> AgentCase:
        """Close every open request whose items are now all answered.

        An information request is what an operator sends when they ask a client
        for something; ``readiness()`` treats an open one as outstanding, and
        refuses to submit the onboarding while any remains. So a request that is
        never closed is a case that can never be approved — which is what
        happened: the questions panel wrote the answers and left the request
        that asked for them open, so a case moved from "waiting on the client"
        to "waiting on nobody" and stopped.

        Closing is derived, not asserted. A request is closed only when EVERY
        item it covers has left the catalogue's own outstanding list, so a
        partial response closes nothing and cannot make a case look complete
        that is not. The close itself goes through the same governed path an
        operator's own "record the response" takes — sent, responded, reviewed
        — so the case history reads identically whichever route answered it.

        Deliberately not conditional on WHO answered. A client's emailed reply
        typed in by an operator, an operator answering from a phone call, and a
        client's own submission are the same act as far as the request is
        concerned: the thing that was asked for is now known.
        """
        case = agent_case.case
        outstanding = {(row["section"], row["field"], row["index"])
                       for row in self.onboarding.catalogue
                       .outstanding_for_client(case.answers)}
        for request in list(case.outstanding_requests):
            items = [(i.get("section"), i.get("field"), i.get("index"))
                     for i in request.items]
            if not items or any(key in outstanding for key in items):
                continue
            agent_case = self._close_request(
                agent_case, request.request_id, actor)
            self._audit(agent_case.run, "client_request_answered",
                        actor_type=ACTOR_HUMAN, actor=actor,
                        classification=EXEC_DETERMINISTIC,
                        input_reference=request.request_id,
                        decision_basis="every item this request asked for is "
                                       "now answered on the case",
                        detail={"items": len(items)})
        return agent_case

    def _close_request(self, agent_case: AgentCase, request_id: str,
                       actor: str) -> AgentCase:
        """Mark the information request this response answers as reviewed."""
        request = agent_case.case.request(request_id)
        if request is None:
            return agent_case
        if request.status == "open":
            self.onboarding.mark_request_sent(case_id=agent_case.case_ref,
                                              request_id=request_id, by=actor)
        self.onboarding.record_response(
            case_id=agent_case.case_ref, request_id=request_id, by=actor,
            note="Structured client form.", answers={})
        agent_case.case = self.onboarding.review_response(
            case_id=agent_case.case_ref, request_id=request_id, accept=True,
            by=actor, note="Structured client form.")
        return agent_case

    def draft_pack(self, agent_case: AgentCase, *, actor: str) -> AgentCase:
        """Draft the client pack, and put it straight in front of a human.

        A draft is never issuable: drafting moves the pack to
        ``HUMAN_REVIEW_REQUIRED`` in the same call, so there is no state in
        which something could be sent without having been read.
        """
        run = agent_case.run
        self._require_action(run, _states.ACTION_DRAFT_PACK)
        built = self.build_pack(agent_case)
        run.pack = built.to_dict()
        document = self.store.write_package(
            run.tenant, run.case_ref, "onboarding_pack.md", built.document())
        email = self.store.write_package(
            run.tenant, run.case_ref, "covering_email.txt",
            f"To: {', '.join(built.email.to) or '(no contact recorded)'}\n"
            f"Subject: {built.email.subject}\n\n{built.email.body}\n")
        run.pack["artefacts"] = [{"name": "onboarding_pack.md",
                                  "ref": document},
                                 {"name": "covering_email.txt", "ref": email}]

        self._set_pack_status(run, _comms.DRAFTED, actor=actor,
                              note="the agent drafted the pack from the "
                                   "governed catalogue")
        self._set_pack_status(run, _comms.HUMAN_REVIEW_REQUIRED, actor=actor,
                              note="a human must read it before it goes out")
        if _states.is_transition_allowed(run.state, _states.PACK_DRAFTED):
            self._move(run, _states.PACK_DRAFTED)
        if _states.is_transition_allowed(run.state,
                                         _states.PACK_REVIEW_REQUIRED):
            self._move(run, _states.PACK_REVIEW_REQUIRED)
        self.store.save(run)
        self._audit(run, "onboarding_pack_drafted", actor_type=ACTOR_AGENT,
                    actor=actor, classification=EXEC_MODEL_PROPOSED,
                    output_reference=document,
                    decision_basis="every question is a field the governed "
                                   "catalogue declares; nothing was invented",
                    detail={"questions": built.question_count,
                            "outstanding": built.outstanding,
                            "content_hash": built.content_hash})
        return agent_case

    def approve_pack(self, agent_case: AgentCase, *, actor: str,
                     reason: str = "") -> AgentCase:
        """A human approves the pack for issue. The agent cannot do this."""
        run = agent_case.run
        self._require_action(run, _states.ACTION_APPROVE_PACK)
        if not run.pack:
            raise OpsError("OCC_AGENT_NO_PACK",
                           "There is no pack to approve. Draft one first.",
                           http_status=409)
        run.approvals.append({
            "approval_id": new_id("appr"), "subject": "client_pack",
            "decision": "approved", "actor": actor, "at": now_iso(),
            "reason": reason,
            "content_hash": str(run.pack.get("content_hash") or "")})
        self._set_pack_status(run, _comms.APPROVED_TO_SEND, actor=actor,
                              note=reason or "approved by a human")
        prior = self._move(run, _states.PACK_APPROVED_TO_SEND)
        self.store.save(run)
        self._audit(run, "onboarding_pack_approved", actor_type=ACTOR_HUMAN,
                    actor=actor, prior_state=prior,
                    classification=EXEC_HUMAN_CONFIRMED,
                    decision_basis=reason or "a human approved the pack for "
                                             "issue",
                    detail={"content_hash": run.pack.get("content_hash")})
        return agent_case

    def send_pack(self, agent_case: AgentCase, *, actor: str,
                  to: Optional[List[str]] = None) -> AgentCase:
        """Issue the approved pack through the communication adapter.

        The receipt says whether it actually left Trakt. In this environment it
        does not, and the receipt says exactly that rather than "sent".
        """
        run = agent_case.run
        self._require_action(run, _states.ACTION_SEND_PACK)
        email = (run.pack or {}).get("email") or {}
        recipients = list(to or email.get("to") or [])
        if not recipients:
            raise OpsError(
                "OCC_AGENT_NO_RECIPIENT",
                "There is no contact address on this case to issue the pack "
                "to. Record one first.", http_status=409)
        receipt = self.communication.deliver(
            to=recipients, subject=str(email.get("subject") or ""),
            body=str(email.get("body") or ""),
            artefacts=list((run.pack or {}).get("artefacts") or []),
            content_hash=str((run.pack or {}).get("content_hash") or ""),
            actor=actor)
        run.pack_receipt = receipt.to_dict()
        self._set_pack_status(run, _comms.SENT, actor=actor,
                              note=receipt.statement)
        prior = self._move(run, _states.PACK_SENT)
        self.store.save(run)
        self._audit(run, "onboarding_pack_issued", actor_type=ACTOR_HUMAN,
                    actor=actor, prior_state=prior,
                    classification=(EXEC_HUMAN_CONFIRMED if receipt.sent
                                    else EXEC_SIMULATED),
                    output_reference=receipt.receipt_id,
                    decision_basis=receipt.statement,
                    detail={"adapter": receipt.adapter, "sent": receipt.sent,
                            "recipients": len(recipients)})
        return agent_case

    def _set_pack_status(self, run: SyntheticRun, to_status: str, *,
                         actor: str, note: str = "") -> None:
        _comms.assert_pack_transition(run.pack_status, to_status)
        run.pack_status = to_status
        run.pack_history.append({"status": to_status, "actor": actor,
                                 "at": now_iso(), "note": note})

    # ------------------------------------------------------------------ #
    # The execution half
    # ------------------------------------------------------------------ #
    def register_synthetic_artefact(self, agent_case: AgentCase, *,
                                    filename: str, data: bytes, actor: str,
                                    fixture_id: str = "",
                                    declared_type: str = "") -> AgentCase:
        run = agent_case.run
        self._require_action(run, _states.ACTION_REGISTER_ARTEFACT)
        artefact = self.artefacts.register(
            run, self.facts(agent_case), filename=filename, data=data,
            provided_by=actor, fixture_id=fixture_id,
            declared_type=declared_type)
        run.received_artefacts.append(artefact.to_dict())
        self.store.save(run)
        self._audit(run, "synthetic_artefact_registered",
                    actor_type=ACTOR_HUMAN, actor=actor,
                    classification=EXEC_SYNTHETICALLY_EXECUTED,
                    input_reference=artefact.source_file,
                    output_reference=artefact.synthetic_location,
                    decision_basis="stored in the practice sandbox; the "
                                   "intended live location was derived but not "
                                   "written",
                    detail={"intended_live_uri": artefact.intended_live_uri,
                            "execution_status": "simulated_only",
                            "sha256": artefact.sha256})
        return agent_case

    def set_run_target(self, agent_case: AgentCase, *, actor: str,
                       portfolio_id: str = "", dataset: str = "",
                       reporting_period: str = "") -> AgentCase:
        """Name which delivery this run is for, and re-derive what follows.

        ``intended_live_uri`` is computed once, when a file is registered, from
        the client, the portfolio and the reporting period. The period is
        normally the last of the three to be known — and nothing recomputed the
        URI when it arrived, so a file uploaded first kept the empty string it
        was given and the card read "Where this would be filed: —" for the life
        of the case, however many times the target was set afterwards.

        Upload-then-name is the ordinary order of work, not a mistake, so the
        destinations are re-derived here rather than the operator being
        expected to remove every file and attach it again.

        Refused once the run is finished. Nothing had ever reached this from a
        screen, so it was ungated; now that it can be typed into, renaming the
        period of a delivery that has already started would leave the record
        describing a period the files did not go to.
        """
        run = agent_case.run
        if run.state in _states.TERMINAL_STATES:
            raise OpsError(
                "OCC_AGENT_RUN_FINISHED",
                "This case is finished, so the delivery it was for cannot be "
                "changed.", 409)
        if portfolio_id:
            run.portfolio_id = portfolio_id
        if dataset:
            run.dataset = dataset
        if reporting_period:
            run.reporting_period = reporting_period
        facts = self.facts(agent_case)
        for doc in run.received_artefacts:
            doc["intended_live_uri"] = self.artefacts.intended_uri(
                run, facts, str(doc.get("source_file") or ""))
        run.facts = facts.to_dict()
        self.store.save(run)
        self._audit(run, "run_target_set", actor_type=ACTOR_HUMAN, actor=actor,
                    classification=EXEC_DETERMINISTIC,
                    decision_basis="an operator named the delivery this run "
                                   "is for",
                    detail={"portfolio_id": run.portfolio_id,
                            "dataset": run.dataset,
                            "reporting_period": run.reporting_period,
                            "artefacts_retargeted":
                                len(run.received_artefacts)})
        return agent_case

    def remove_synthetic_artefact(self, agent_case: AgentCase, *,
                                  artefact_id: str, actor: str) -> AgentCase:
        """Take a file back out of the case's pack.

        WHY THIS EXISTS. Uploading was a one-way door: ``received_artefacts``
        was appended to and never read back out, and the wrong file — an
        encrypted workbook, a draft, last month's tape — stayed in the pack for
        the life of the case. Re-uploading under the same name overwrites the
        BYTES (:meth:`SyntheticRunStore.write_artefact_bytes` writes a
        sanitised leaf with no uniquifier) but appends a second ROW, so the
        only way to correct a mistake was to make the record say two files had
        arrived when one had. Cancelling the case was the alternative.

        WHAT IT REMOVES, EXACTLY. The record, not the bytes. Everything that
        decides what activation does reads ``run.artefacts()`` — the intent's
        file list, :meth:`_payloads`, classification, and role readiness — so
        dropping the row removes the file from all of them. The uploaded bytes
        stay where they were written, in this case's own sandbox under
        ``{tenant}/agent-runs/{case_ref}/artefacts/``, and are not written
        anywhere else. ``Storage`` has no delete at all, and this is not the
        change that should introduce one: a governed store that can be told to
        forget is a much larger decision than an operator undoing an upload.

        Allowed wherever uploading is allowed, and for the same reason — if an
        operator may add a file at this state, they may take one back.
        """
        run = agent_case.run
        self._require_action(run, _states.ACTION_REGISTER_ARTEFACT)
        removed = next((a for a in run.received_artefacts
                        if a.get("artefact_id") == artefact_id), None)
        if removed is None:
            raise OpsError("OCC_AGENT_ARTEFACT_NOT_FOUND",
                           "That file is not in this case's pack.", 404)
        run.received_artefacts = [a for a in run.received_artefacts
                                  if a.get("artefact_id") != artefact_id]
        self.store.save(run)
        self._audit(run, "synthetic_artefact_removed",
                    actor_type=ACTOR_HUMAN, actor=actor,
                    classification=EXEC_SYNTHETICALLY_EXECUTED,
                    input_reference=str(removed.get("source_file") or ""),
                    output_reference=str(removed.get("synthetic_location")
                                         or ""),
                    decision_basis="an operator removed the file from the "
                                   "pack; its bytes remain in the case "
                                   "sandbox and were never written elsewhere",
                    detail={"artefact_id": artefact_id,
                            "sha256": str(removed.get("sha256") or ""),
                            "bytes_deleted": False})
        # Readiness, recognition and the onboarding case's sample all follow
        # from the pack, so they are recomputed rather than left describing a
        # file the case no longer holds.
        return self.classify_artefacts(agent_case, actor=actor)

    def generate_synthetic_answers(self, agent_case: AgentCase, *,
                                   actor: str) -> AgentCase:
        """Answer THIS case's own outstanding client questions, synthetically.

        The counterpart to :meth:`generate_synthetic_response`, which makes up
        files. At "receive client responses" an operator is looking at a
        checklist of unanswered questions, and generating a loan tape does
        nothing for it — the two are different acts and this is the second one.

        The answers are derived from the served form's own declarations and
        then submitted through :meth:`submit_client_response`, so they take the
        identical validated path a real client's answers take: authoritative
        keys, checked against the form actually served, written by
        ``OnboardingService.save_step`` exactly as submitted. Nothing here
        writes to a case directly, and no validation is skipped.
        """
        from . import fixtures as _fixtures

        # Deliberately NOT gated on the run's state. A client may answer at any
        # point, so `submit_client_response` is ungated too — and this must not
        # be harder to reach than the path it stands in for. `ACTION_ANSWER` is
        # a different thing: answering an EXCEPTION raised during a run.
        form = self.client_form(agent_case)
        answers = _fixtures.generate_answers(
            form, client_name=self.facts(agent_case).client_name)
        if not answers:
            raise OpsError(
                "OCC_AGENT_NOTHING_TO_ANSWER",
                "There is nothing outstanding for Trakt to answer on this "
                "case.", http_status=409)
        agent_case = self.submit_client_response(
            agent_case, actor=actor, response=answers)
        self._audit(agent_case.run, "synthetic_answers_generated",
                    actor_type=ACTOR_AGENT, actor=actor,
                    classification=EXEC_SYNTHETICALLY_EXECUTED,
                    decision_basis="answers derived from the served form's own "
                                   "declared types and options; submitted "
                                   "through the ordinary client-response path",
                    detail={"answers": len(answers)})
        return agent_case

    def generate_synthetic_response(self, agent_case: AgentCase, *,
                                    actor: str) -> AgentCase:
        """Generate a client response for THIS case's own requirements.

        The alternative to uploading files or replaying a fixture: the same
        generators the fixtures use, driven by the roles the delivery outcome
        requires and the case's own client and portfolio. It registers what it
        produces through the ordinary artefact path, so a generated response is
        subject to exactly the same sanitisation, classification and controls.
        """
        from . import fixtures as _fixtures

        run = agent_case.run
        self._require_action(run, _states.ACTION_REGISTER_ARTEFACT)
        facts = self.facts(agent_case)
        roles = artefact_vocabulary().required_roles(facts.outcome)
        files = _fixtures.generate_response(
            roles=roles,
            client_name=facts.client_name or facts.client_id or "Practice",
            portfolio_id=facts.portfolio_id)
        if not files:
            raise OpsError(
                "OCC_AGENT_NOTHING_TO_GENERATE",
                "Trakt cannot make up files for this kind of delivery. Upload "
                "them, or start from a prepared example.", http_status=400)
        for spec in files:
            agent_case = self.register_synthetic_artefact(
                agent_case, filename=spec.filename,
                data=spec.content.encode("utf-8"), actor=actor,
                fixture_id="generated", declared_type=spec.declared_type)
        self._audit(agent_case.run, "synthetic_response_generated",
                    actor_type=ACTOR_AGENT, actor=actor,
                    classification=EXEC_SYNTHETICALLY_EXECUTED,
                    decision_basis="generated from the delivery outcome the "
                                   "case's own products imply",
                    detail={"roles": roles, "files": len(files)})
        return self.classify_artefacts(agent_case, actor=actor)

    def classify_artefacts(self, agent_case: AgentCase, *,
                           actor: str) -> AgentCase:
        """Recognise what each file is, and tell the onboarding case about it.

        The pack is registered with Client Onboarding as a sample too, so the
        file format, the expected file names and often the asset class are
        answered by *its* inference rather than by this feature.

        Recognition never blocks. Whether the pack is *complete* is a
        precondition of the practice run, and is checked there — an incomplete
        pack must not stop an operator finishing the onboarding, which is the
        half that produces the configuration.
        """
        run = agent_case.run
        classified, findings = self.artefacts.classify(run.artefacts())
        run.received_artefacts = [a.to_dict() for a in classified]
        for finding in findings:
            if finding not in run.observations:
                run.observations.append(finding)
        readiness = self.artefacts.readiness(run, self.facts(agent_case).outcome)
        self._record_control(run, "artefact_readiness", readiness.to_dict())
        self.store.save(run)
        self._audit(run, "artefacts_classified", actor_type=ACTOR_AGENT,
                    actor=actor, classification=EXEC_DETERMINISTIC,
                    decision_basis="apps.blob_trigger_app.file_roles",
                    detail=readiness.to_dict())

        # AN APPROVED CASE IS THE ONLY KIND THAT CAN REHEARSE, so excluding one
        # here excluded every pack the rehearsal exists to read.
        # `run_synthetic_onboarding` refuses to start unless the onboarding is
        # APPROVED; this refused to record the sample once it was. The two
        # conditions never overlap, so the files uploaded FOR the practice run
        # could not reach the sample, and the expected delivery stayed at
        # whatever was registered before approval — one file, where a client
        # sends three, with no operator action able to correct it.
        if agent_case.case.status not in TERMINAL:
            agent_case.case = self.onboarding.register_sample(
                case_id=agent_case.case_ref,
                files=sample_manifest(classified), by=actor)
        run.facts = self.facts(agent_case).to_dict()
        self.store.save(run)
        return agent_case

    def start_synthetic_onboarding(self, agent_case: AgentCase, *,
                                   actor: str) -> Dict[str, Any]:
        """Start the practice run on its own thread and return immediately.

        The same pattern the live engine already uses
        (:meth:`operations_control.engine.Engine.start`): a pipeline pass takes
        minutes, and an HTTP request must not hold one open. The run's own state
        is the job record — an operator polls the case, not a separate job.
        """
        run = agent_case.run
        self._require_action(run, _states.ACTION_RUN_ONBOARDING)
        key = f"{run.tenant}/{run.case_ref}"
        with self._jobs_lock:
            existing = self._jobs.get(key)
            if existing is not None and existing.is_alive():
                raise OpsError(
                    "OCC_AGENT_RUN_IN_PROGRESS",
                    "This practice case is already running.", http_status=409)

            def _job() -> None:
                try:
                    self.run_synthetic_onboarding(
                        self.load(run.tenant, run.case_ref), actor=actor)
                except Exception:               # noqa: BLE001 — see below
                    # The failure is recorded on the run by _block/_audit where
                    # it is recoverable; anything else is logged rather than
                    # lost with the thread.
                    logger.exception("occ_agent: practice run %s failed",
                                     run.case_ref)

            thread = threading.Thread(target=_job, name=f"occ-agent-{key}",
                                      daemon=True)
            self._jobs[key] = thread
            thread.start()
        return {"case_ref": run.case_ref, "started": True,
                "state": run.state,
                "poll": f"/api/occ-agent/cases/{run.case_ref}"}

    def job_running(self, tenant: str, case_ref: str) -> bool:
        with self._jobs_lock:
            thread = self._jobs.get(f"{tenant}/{case_ref}")
        return bool(thread is not None and thread.is_alive())

    def run_synthetic_onboarding(self, agent_case: AgentCase, *,
                                 actor: str) -> AgentCase:
        run = agent_case.run
        self._require_action(run, _states.ACTION_RUN_ONBOARDING)
        if agent_case.case.status != APPROVED:
            raise OpsError(
                "OCC_AGENT_ONBOARDING_NOT_APPROVED",
                "The onboarding has to be approved before a practice run can "
                "use its configuration.", http_status=409)
        facts = self.facts(agent_case)
        if not facts.complete:
            return self._block(
                agent_case,
                ["The practice run needs a client identifier, a portfolio "
                 "identifier and an asset class before it can start."],
                actor=actor, reason="the onboarding does not identify a book")
        # Is the pack complete? The configured input requirements decide, not
        # this feature — the same declaration the live intake route uses.
        roles = self.artefacts.readiness(run, facts.outcome)
        if not roles.ready:
            return self._block(agent_case, _missing_role_messages(roles),
                               actor=actor,
                               reason="required input roles not satisfied")

        # THE PACK ABOUT TO BE READ IS THE PACK TO BE EXPECTED. Registering the
        # sample otherwise happens only on a FILE action — an upload, a
        # removal, a fixture — so an operator whose expected delivery was
        # recorded wrongly had nothing they could press to correct it. Starting
        # the run is the act that says "use what I have given you", and it
        # reads every artefact on the case, so it states the same thing to the
        # onboarding record. Recorded, never fatal: a sample that cannot be
        # registered must not take the practice run down with it.
        self._record_sample_from_pack(agent_case, actor=actor)
        # THIS run's obstacles, not the last one's. run.blockers was written by
        # _block and cleared by nothing, so a case that recovered kept showing
        # what used to be wrong.
        run.blockers = []

        prior = self._move(run, _states.SYNTHETIC_ONBOARDING_RUNNING)
        run.facts = facts.to_dict()
        self.store.save(run)
        self._audit(run, "synthetic_onboarding_started", actor_type=ACTOR_AGENT,
                    actor=actor, prior_state=prior,
                    classification=EXEC_SYNTHETICALLY_EXECUTED,
                    decision_basis="the existing orchestration conductor was "
                                   "run over the synthetic adapter")

        # Rebuild the working files from durable storage first: the instance
        # that runs the case need not be the one that received the upload.
        self.store.materialise(run.tenant, run.case_ref)
        adapters = SyntheticOnboardingAdapters(
            artefact_paths=self._artefact_paths(run), policy=self.policy,
            sandbox=self.store.case_dir(run.tenant, run.case_ref),
            asset_type=facts.asset_class,
            confirmed_product_profile=run.confirmed_product_profile,
            regime=facts.regime,
            approved_mappings=self._approved_mappings(run),
            # A new client's columns have never been read by anyone. An
            # amendment's have, and are already governed — see
            # `SyntheticOnboardingAdapters.confirm_every_mapping`.
            confirm_every_mapping=(agent_case.case.kind == KIND_NEW_CLIENT),
            # THE CLIENT'S OWN STANDING ANSWERS. The originator's name, LEI and
            # country of establishment are captured once on the entity holding
            # the originator role and reach the regime defaults from there —
            # `onboarding.artefacts` already derives exactly this, and an
            # operator is never asked for the same legal name twice. Read here
            # so a field the client configuration already holds is not reported
            # as something the regulator is still waiting for.
            client_defaults=self._standing_client_defaults(agent_case),
            source_units=dict(run.source_units or {}),
            case_id=run.case_ref, tenant=run.tenant)
        run_root = self.store.run_dir(run.tenant, run.case_ref)
        self._purge_stale_decisions(run_root)
        state = run_synthetic_orchestration(
            adapters, client_id=facts.client_id,
            portfolio_id=facts.portfolio_id, out_root=run_root,
            created_at=run.created_at,
            target=("regime" if facts.regime else "mi"),
            regime=facts.regime or None,
            run_id=f"syn_{run.case_ref.lower().replace('-', '_')}")

        for record in adapters.records:
            run.stage_outcomes[record.stage] = record.outcome
            self._record_control(run, "stage", record.to_dict())
        if adapters.validation_report:
            self._record_control(run, "validation",
                                 {"findings": adapters.validation_report})
        run.mapping_report = adapters.mapping_report
        run.cross_file = adapters.cross_file
        run.excused_findings = adapters.excused_findings
        # A finding that did not block still belongs on the operator's screen.
        # It used to be a count on a stage that had gone green, which was
        # survivable only while REVIEW was the quiet outcome — and it is not
        # any more, now that volume alone cannot promote a warning-severity
        # check into a refusal.
        #
        # THIS RUN'S FINDINGS REPLACE THE LAST RUN'S. Observations otherwise
        # only ever accumulate, which is right for the artefact notes they were
        # built for — a file was recognised, and it stays recognised — and
        # exactly wrong for a count that a rerun is meant to change. An
        # operator who fixes a mapping and reruns would be shown the figure
        # they just fixed sitting underneath the figure they fixed it to.
        run.observations = [o for o in run.observations
                            if MATERIALITY_MARK not in o]
        run.observations.extend(adapters.review_findings)
        run.llm = adapters.llm

        # Resolved decisions are kept (they are the record of what the human
        # settled); a decision the rerun raises again replaces its open twin
        # rather than being appended beside it.
        decisions = self._decisions_from_run(run, facts, run_root)
        # The product question, raised beside the blockers it would clear.
        # Answering it is what lets the profile excuse anything, so it belongs
        # in the same list an operator works through rather than in a panel of
        # its own.
        if adapters.product_profile_decision is not None:
            decisions = [adapters.product_profile_decision, *decisions]
        # WHAT A HUMAN ACTUALLY SETTLED, rather than everything that is not the
        # word "open". The distinction was invisible while every card carried
        # an explicit `"status": "open"` and every settled one carried
        # "approved" — and then one decision was raised carrying "pending",
        # which is neither. It counted as settled, `setdefault` below refused to
        # replace it, and it was stuck on the run for good: a rerun could not
        # dislodge it and a deploy could not fix it, because the broken value
        # was in the case's own document rather than in the code.
        #
        # A status nobody wrote deliberately is not an answer. These three are
        # the ones the writers use — `resolve_decision` sets approved/rejected,
        # `acknowledge_exception` sets acknowledged — and anything else is
        # re-raisable, which is the safe direction: the worst case is a
        # question asked twice, not an answer that cannot be given.
        settled = {d["decision_id"]: d for d in run.open_decisions
                   if str(d.get("status") or "").lower() in _SETTLED_STATUSES}
        for decision in decisions:
            settled.setdefault(decision["decision_id"], decision)
        run.open_decisions = list(settled.values())
        open_now = [d for d in run.open_decisions
                    if d.get("status", "open") == "open"]
        run.planned_pipeline_actions = _planned_actions(state)

        if open_now:
            self._move(run, _states.EXCEPTIONS_REQUIRE_INPUT)
            self.store.save(run)
            self._audit(run, "synthetic_onboarding_needs_input",
                        actor_type=ACTOR_AGENT, actor=actor,
                        classification=EXEC_SYNTHETICALLY_EXECUTED,
                        decision_basis="a governed control needs a human",
                        detail={"open_decisions": len(open_now)})
            return agent_case

        blocked = [r for r in adapters.records
                   if r.outcome == STAGE_HARD_BLOCKED]
        if blocked or state.status not in ("done",):
            blockers = [b for r in blocked for b in r.blockers] or \
                list(state.blockers)
            self._move(run, _states.EXCEPTIONS_REQUIRE_INPUT)
            self.store.save(run)
            return self._block(agent_case, blockers, actor=actor,
                               reason="a deterministic control blocked the run")

        self._move(run, _states.SYNTHETIC_ONBOARDING_PASSED)
        self.store.save(run)
        self._audit(run, "synthetic_onboarding_passed", actor_type=ACTOR_AGENT,
                    actor=actor, classification=EXEC_SYNTHETICALLY_EXECUTED,
                    decision_basis="every deterministic control passed",
                    detail={"stages": dict(run.stage_outcomes)})
        return agent_case

    # ------------------------------------------------------------------ #
    # Decisions
    # ------------------------------------------------------------------ #
    def resolve_decision(self, agent_case: AgentCase, *, decision_id: str,
                         action: str, value: str = "", reason: str = "",
                         actor: str) -> AgentCase:
        """Record a human decision, then rerun the affected controls."""
        run = agent_case.run
        self._require_action(run, _states.ACTION_RESOLVE_DECISION)
        target = next((d for d in run.open_decisions
                       if d.get("decision_id") == decision_id), None)
        if target is None:
            raise OpsError("OCC_AGENT_DECISION_NOT_FOUND",
                           "That decision could not be found on this case.",
                           http_status=404)
        if action not in ("approve", "amend", "reject"):
            raise OpsError("OCC_AGENT_INVALID_DECISION",
                           "That is not an answer Trakt understands.",
                           http_status=400)
        target["status"] = "approved" if action != "reject" else "rejected"
        target["resolution"] = action
        target["resolved_value"] = value or target.get("recommendation", "")
        target["resolved_by"] = actor
        target["resolved_at"] = now_iso()
        target["reason"] = reason
        # Confirming the product is not just an answered question: it is what
        # lets the product profile excuse a field, so the confirmation is
        # recorded on the run and every later run reads it. Approving it takes
        # the proposed profile; amending it takes the one the operator named.
        if str(target.get("decision_type")) == "product_confirmation" \
                and action != "reject":
            run.confirmed_product_profile = str(
                value or target.get("profile_id") or "")
        self.store.save(run)
        self._audit(run, "human_decision_recorded", actor_type=ACTOR_HUMAN,
                    actor=actor, classification=EXEC_HUMAN_CONFIRMED,
                    input_reference=decision_id,
                    decision_basis=reason or f"operator chose '{action}'",
                    detail={"resolved_value": target["resolved_value"]})

        if not run.blocking_decisions():
            if _states.is_transition_allowed(
                    run.state, _states.SYNTHETIC_ONBOARDING_RUNNING):
                return self.run_synthetic_onboarding(agent_case, actor=actor)
        return agent_case

    # ------------------------------------------------------------------ #
    # The mapping table: read it, then commit it
    # ------------------------------------------------------------------ #
    @staticmethod
    def _standing_client_defaults(agent_case: AgentCase) -> Dict[str, Any]:
        """The regime's standing CLIENT fields, from the entity that holds the
        role rather than from the loan tape.

        ``config/regime/onboarding_standing_fields.yaml`` declares the
        originator's name (RREL82), LEI (RREL83) and country of establishment
        (RREL84) as ``standing_client``, ``source: derived``, ``derives_from:
        the entity holding the originator role``. They are not columns in a
        monthly extract, and a delivery stopped for want of them is asking the
        client to restate per loan what they told us once at onboarding.
        """
        try:
            originators = agent_case.case.entities_with_role("originator")
        except Exception:                    # noqa: BLE001 — a case guard
            return {}
        if not originators:
            return {}
        entity = originators[0]
        out: Dict[str, Any] = {}
        for key, field_name in (
                ("legal_name", "originator_name"),
                ("lei", "originator_legal_entity_identifier"),
                ("country_of_establishment",
                 "originator_establishment_country")):
            value = str(entity.get(key) or "").strip()
            if value:
                out[field_name] = value
        return out

    def _require_mapping_change(self, run: SyntheticRun) -> None:
        """A mapping may be answered while the run is at the mapping stage,
        and RE-answered at any point until the case activates.

        Two permissions rather than one, because they are two different acts.
        ``resolve_decision`` answers a question that is still open.
        ``reopen_mapping`` takes back a reading the run has already settled,
        which sends the run backwards and withdraws what rested on it — see
        :meth:`_reopen_for_mapping_change`.
        """
        if _states.action_allowed(run.state, _states.ACTION_RESOLVE_DECISION):
            return
        if _states.action_allowed(run.state, _states.ACTION_REOPEN_MAPPING):
            return
        raise ActionNotAllowed(_states.ACTION_REOPEN_MAPPING, run.state)

    def _reopen_for_mapping_change(self, run: SyntheticRun, *, actor: str,
                                   columns: List[str]) -> bool:
        """Put the run back at the mapping stage, and withdraw what rested on
        the reading being changed.

        A COMMITTED MAPPING IS NOT A PERMANENT ONE. It becomes permanent at
        activation, when promotion writes it into the client's governed rules;
        until then the whole rehearsal is provisional, and an operator who
        reads a settled row and sees it is wrong must be able to say so. A
        screen that says "You confirmed it" with nothing to click is telling
        them their mistake is final when it is not.

        WHAT IT COSTS, SAID OUT LOUD. Readiness was evaluated against the old
        reading and activation may have been approved against it. Those
        approvals were about a delivery that no longer exists, so they are
        withdrawn here rather than left standing over a changed mapping — the
        one outcome worse than not being able to go back is going back
        silently and activating on an approval nobody would give again.
        """
        if run.state == _states.EXCEPTIONS_REQUIRE_INPUT:
            return False
        prior = run.state
        withdrawn = [a["subject"] for a in run.approvals
                     if a.get("decision") == "approved"
                     and a.get("subject") in ("execution_readiness",
                                              "configuration")]
        for subject in withdrawn:
            run.approvals.append({
                "approval_id": new_id("appr"), "subject": subject,
                "decision": "withdrawn", "actor": actor, "at": now_iso(),
                "reason": "a mapping this approval rested on was re-opened"})
        run.readiness_status = "not_evaluated"
        run.readiness = {}
        run.review_package_ref = ""
        run.readiness_package_ref = ""
        run.activation_intent = {}
        self._move(run, _states.EXCEPTIONS_REQUIRE_INPUT)
        self._audit(run, "mapping_reopened", actor_type=ACTOR_HUMAN,
                    actor=actor, prior_state=prior,
                    classification=EXEC_HUMAN_CONFIRMED,
                    decision_basis="an operator took back a mapping the "
                                   "rehearsal had already settled",
                    detail={"columns": columns,
                            "approvals_withdrawn": withdrawn})
        return True

    def stage_mapping(self, agent_case: AgentCase, *, source_file: str,
                      source_column: str, action: str, actor: str,
                      target_field: str = "", reason: str = "",
                      origin: str = _staging.ORIGIN_OPERATOR) -> AgentCase:
        """What this operator says one column is. A draft, not an act.

        Nothing is resolved, nothing is promoted and nothing reruns: the answer
        is held on the run so it survives a refresh, and it can be replaced or
        withdrawn until the set is committed. See :mod:`.staging`.
        """
        run = agent_case.run
        self._require_mapping_change(run)
        column = str(source_column or "").strip()
        file_name = str(source_file or "").strip()
        what = str(action or "").strip()
        if what not in _staging.ACTIONS:
            raise OpsError("OCC_AGENT_UNKNOWN_MAPPING_ACTION",
                           "That is not something Trakt can record about a "
                           "column.", http_status=400)
        row = self._mapping_row(run, file_name, column)
        if row is None:
            raise OpsError("OCC_AGENT_COLUMN_NOT_FOUND",
                           f"'{column}' is not a column Trakt read in "
                           f"{file_name}.", http_status=404)
        previous = _staging.find(run.staged_mappings, file_name, column)
        run.staged_mappings = [
            e for e in run.staged_mappings
            if _staging.key(e.get("source_file"), e.get("source_column"))
            != _staging.key(file_name, column)]
        if what == _staging.ACTION_CLEAR:
            # Undoing a set-aside a field request wrote takes the REQUEST back
            # too. Leaving the ask standing over a column that has gone back to
            # being proposed is the mapped-and-requested state this path exists
            # to prevent, reached from the other end.
            if _staging.is_request_driven(previous):
                self._withdraw_field_request(
                    run, file_name, column, actor=actor, at=now_iso(),
                    why="the operator undid the set-aside the request made")
            self.store.save(run)
            return agent_case

        field_name = str(target_field or "").strip()
        if what == _staging.ACTION_CONFIRM:
            # Confirming means "what is on the row is right", so the field is
            # the row's own — taking one from the request would let a caller
            # confirm a column onto a field the operator never saw.
            field_name = str(row.get("canonical_field") or "")
            if not field_name:
                raise OpsError(
                    "OCC_AGENT_NOTHING_TO_CONFIRM",
                    f"Trakt has not read '{column}' as anything, so there is "
                    "nothing to confirm. Name the field instead.",
                    http_status=409)
        elif what == _staging.ACTION_AMEND:
            self._require_registry_field(agent_case, field_name)
        else:
            field_name = ""

        # Answering a column the run has already SETTLED is a different act
        # from answering one that is still open: it sends the run back to the
        # mapping stage and withdraws the approvals that rested on the old
        # reading. Done here, at the moment the operator says so, rather than
        # at the commit — so the screen tells them immediately what their
        # change costs instead of after they press the button.
        if str(row.get("tier") or "") == "operator_approved":
            self._reopen_for_mapping_change(run, actor=actor,
                                            columns=[column])
        decision_id, _, _ = _mapping_view.decision_for_column(
            run.open_decisions, file_name, column)
        run.staged_mappings.append(_staging.entry(
            source_file=file_name, source_column=column, action=what,
            target_field=field_name, decision_id=decision_id, actor=actor,
            at=now_iso(), reason=reason, origin=origin))
        # An ask for a new field and a decision about the column are two
        # answers to one question; recording either withdraws the other.
        if field_name:
            self._withdraw_field_request(
                run, file_name, column, actor=actor, at=now_iso(),
                why="the column was given a field instead")
        self.store.save(run)
        return agent_case

    def confirm_mappings(self, agent_case: AgentCase, *, actor: str,
                         reason: str = "") -> AgentCase:
        """THE act. Everything the operator staged, plus every untouched
        proposal, applied in one go.

        One act for them; one resolved decision per column on the record,
        because that is what promotion turns into governed rules and what an
        auditor asking "who said 'Month Run' was the cut-off date?" reads back.
        Each column keeps the approver who STAGED it — the reading is the
        judgement, and a second operator pressing the button has not done it.

        Refused while a genuine question is unanswered. A proposal carries a
        field and "confirm" means accepting it; a weak match or an ambiguity
        carries no answer at all, and committing the set around one would be
        the button inventing an answer nobody gave.
        """
        run = agent_case.run
        self._require_mapping_change(run)
        overview = _mapping_view.overview(run)
        unanswered = overview["unanswered_questions"]
        if unanswered:
            raise OpsError(
                "OCC_AGENT_QUESTIONS_UNANSWERED",
                f"{unanswered} column{'s' if unanswered != 1 else ''} still "
                f"need{'' if unanswered != 1 else 's'} an answer before the "
                "mappings can be confirmed.", http_status=409)
        staged = _staging.by_column(run.staged_mappings)
        untouched = [d for d in run.open_decisions
                     if _decision_type_of(d) == DECISION_MAPPING_PROPOSAL
                     and str(d.get("status", "open")) in ("open", "pending")
                     and _staging.key(
                         (d.get("subject") or {}).get("source_file"),
                         (d.get("subject") or {}).get("source_column"))
                     not in staged]
        if not staged and not untouched:
            raise OpsError(
                "OCC_AGENT_NOTHING_TO_CONFIRM",
                "There are no mappings waiting to be confirmed.",
                http_status=409)

        at = now_iso()
        applied: Dict[str, str] = {}
        # ONE RESOLVED DECISION PER COLUMN, always.
        #
        # An ambiguity is one decision about SEVERAL columns, and the operator
        # answers per column. Writing each of their answers onto that shared
        # decision made the last one overwrite the rest: setting the losing
        # column aside stamped "not used" on the decision, and
        # `_approved_mappings` reads that as BOTH columns unused — including
        # the one they had just confirmed. So each answer gets a record of its
        # own, and the shared decision is settled separately, below, from the
        # whole set of answers rather than from whichever came last.
        for item in run.staged_mappings:
            file_name = str(item.get("source_file") or "")
            column = str(item.get("source_column") or "")
            value = _staging.resolved_value(item)
            decision = self._own_decision_for(run, file_name, column)
            if decision is None:
                decision = _field_registry.alias_decision(
                    source_file=file_name, source_column=column,
                    target_field=item.get("target_field") or "",
                    actor=str(item.get("staged_by") or actor),
                    reason=str(item.get("reason") or ""),
                    at=str(item.get("staged_at") or at))
                run.open_decisions.append(decision)
            decision["status"] = "approved"
            decision["resolution"] = _staging.RESOLUTION[item["action"]]
            decision["resolved_value"] = value
            decision["resolved_by"] = str(item.get("staged_by") or actor)
            decision["resolved_at"] = str(item.get("staged_at") or at)
            decision["reason"] = (str(item.get("reason") or "")
                                  or reason or "confirmed with the mapping set")
            applied[_mapping_key(file_name, column)] = value
        self._settle_shared_decisions(run, actor=actor, at=at, reason=reason)
        for decision in untouched:
            subject = decision.get("subject") or {}
            value = str(subject.get("target_field")
                        or decision.get("target_field") or "")
            decision["status"] = "approved"
            decision["resolution"] = "approve"
            decision["resolved_value"] = value
            decision["resolved_by"] = actor
            decision["resolved_at"] = at
            decision["reason"] = reason or "confirmed as proposed"
            applied[_mapping_key(str(subject.get("source_file") or ""),
                                 str(subject.get("source_column") or ""))] = value

        counts = _staging.summary(run.staged_mappings)
        run.staged_mappings = []
        self.store.save(run)
        self._audit(run, "mappings_confirmed", actor_type=ACTOR_HUMAN,
                    actor=actor, classification=EXEC_HUMAN_CONFIRMED,
                    decision_basis=(reason or "the operator confirmed the "
                                    "mappings for this delivery"),
                    # Keyed by FILE and column, because every file in a pack
                    # carries a loan identifier: keyed on the name alone, one
                    # confirmation of three columns left one line in the record
                    # and an auditor asking which file could not be answered.
                    detail={"columns": len(applied),
                            "as_proposed": len(untouched),
                            "changed": counts[_staging.ACTION_AMEND],
                            "set_aside": counts[_staging.ACTION_NOT_USED],
                            "mappings": applied})
        if not run.blocking_decisions():
            if _states.is_transition_allowed(
                    run.state, _states.SYNTHETIC_ONBOARDING_RUNNING):
                return self.run_synthetic_onboarding(agent_case, actor=actor)
        return agent_case

    @staticmethod
    def _own_decision_for(run: SyntheticRun, source_file: str,
                          source_column: str) -> Optional[Dict[str, Any]]:
        """The open decision about THIS column and no other, if there is one.

        A decision naming several columns is deliberately not returned: it is
        one question about the set and is settled from the whole set, not from
        whichever of its columns the operator answered last.

        Matched on the pair, falling back to the bare column name for a
        decision recorded before decisions carried their file.
        """
        fallback = None
        for decision in run.open_decisions:
            # An ALREADY-RESOLVED decision about this column counts: the
            # operator is re-answering it. Skipping it created a SECOND
            # decision for the same column, and the rerun collapses decisions
            # by id — so one of the two answers was silently dropped.
            if str(decision.get("status", "open")) not in ("open", "approved",
                                                           "pending"):
                continue
            subject = decision.get("subject") or {}
            if len(subject.get("source_columns") or []) > 1:
                continue
            if str(subject.get("source_column") or "") != source_column:
                continue
            if str(subject.get("source_file") or "") == source_file:
                return decision
            if not subject.get("source_file"):
                fallback = decision
        return fallback

    @staticmethod
    def _settle_shared_decisions(run: SyntheticRun, *, actor: str, at: str,
                                 reason: str) -> None:
        """Close an ambiguity from the answers its columns were given.

        "Which of these two is the balance?" is answered by the operator
        keeping one and setting the other aside, and the decision records the
        COLUMN that won — which is the shape ``_approved_mappings`` and
        ``mapping_promotion`` both read for an ambiguity, and the opposite of
        the shape a confirmation uses. If they set all of them aside, that is
        an answer too: none of these is it.
        """
        staged = _staging.by_column(run.staged_mappings)
        for decision in run.open_decisions:
            if str(decision.get("status", "open")) not in ("open", "approved",
                                                           "pending"):
                continue
            subject = decision.get("subject") or {}
            columns = [str(c or "") for c in
                       (subject.get("source_columns") or [])]
            if len(columns) < 2:
                continue
            source_file = str(subject.get("source_file") or "")
            answers = [staged.get(_staging.key(source_file, c))
                       for c in columns]
            if not all(answers):
                continue          # still a question; the commit refused above
            target = str(subject.get("target_field") or "")
            winner = next((a for a in answers
                           if a["action"] != _staging.ACTION_NOT_USED
                           and a["target_field"] == target), None)
            decision["status"] = "approved"
            decision["resolution"] = "amend" if winner else "approve"
            decision["resolved_value"] = (winner["source_column"] if winner
                                          else _staging.NOT_USED_VALUE)
            decision["resolved_by"] = str(
                (winner or answers[0]).get("staged_by") or actor)
            decision["resolved_at"] = str(
                (winner or answers[0]).get("staged_at") or at)
            decision["reason"] = (reason
                                  or "settled with the mapping set")

    def _require_registry_field(self, agent_case: AgentCase,
                                field_name: str) -> None:
        """A field an operator names has to be one this book reports on.

        Said as two different sentences, because the remedies differ: a name
        nothing in the registry has is a request, and a name this book cannot
        use is a configuration question about the book.
        """
        name = str(field_name or "").strip()
        if not name:
            raise OpsError("OCC_AGENT_FIELD_REQUIRED",
                           "Name the field this column feeds.",
                           http_status=400)
        facts = self.facts(agent_case)
        if _field_registry.known_field(name, facts.asset_class):
            return
        if _field_registry.registered_anywhere(name):
            raise OpsError(
                "OCC_AGENT_FIELD_NOT_FOR_THIS_BOOK",
                f"'{name}' is a Trakt field, but not one this asset class "
                "reports on.", http_status=409)
        raise OpsError(
            "OCC_AGENT_FIELD_NOT_REGISTERED",
            f"Trakt has no field called '{name}'. If it genuinely does not "
            "exist, request it as a new field instead.", http_status=409)

    # ------------------------------------------------------------------ #
    # Columns that matched nothing
    # ------------------------------------------------------------------ #
    def field_catalogue(self, agent_case: AgentCase) -> List[Dict[str, Any]]:
        """Every canonical field this book may map an unmapped column to."""
        facts = self.facts(agent_case)
        return _field_registry.catalogue(facts.asset_class,
                                         regime=facts.regime or "")

    def map_unmapped_column(self, agent_case: AgentCase, *, source_file: str,
                            source_column: str, target_field: str, actor: str,
                            reason: str = "") -> AgentCase:
        """An operator naming the existing field a column feeds.

        The column matched nothing, so nothing asked about it and there is no
        decision to answer — the operator is volunteering knowledge the
        platform did not have. It STAGES like every other answer on this table,
        and ``confirm_mappings`` turns it into a resolved decision: that is the
        shape promotion reads, so this client's name for the field becomes a
        governed rule at activation and the next delivery matches it without
        asking.
        """
        return self.stage_mapping(
            agent_case, source_file=source_file, source_column=source_column,
            action=_staging.ACTION_AMEND, target_field=target_field,
            actor=actor,
            reason=reason or "an operator named the field this column feeds")

    def request_registry_field(self, agent_case: AgentCase, *,
                               source_file: str, source_column: str,
                               field_name: str, actor: str, label: str = "",
                               description: str = "", data_type: str = "",
                               reason: str = "") -> AgentCase:
        """An ask for a canonical field the platform does not have.

        Recorded, not created. Adding a field changes the vocabulary every
        client's report is written in, and it has its own governed route — a
        versioned system config package an administrator drafts and activates.
        The column stays unmapped in this delivery, visibly, so nobody reads a
        request as a mapping.

        THE COLUMN IS SET ASIDE HERE, which is what makes the sentence above
        true. It used to record the ask and touch nothing else, and that was
        only harmless while a request could only come from a column that
        matched nothing. It cannot: "Change" opens the same dialog on ANY row,
        so a column with a live proposal could be requested as a new field and
        keep its proposal — and the commit approves every untouched proposal.
        One column would be mapped and requested at once. That is how
        'ERCs Paid in the period' — a money column — came to sit on a standing
        request for ``early_repayment_charge_amount_in_period`` while still
        proposed onto ``early_repayment_charge``, which is a Y/N field: a
        decimal heading into a boolean parser, with a governed rule promoted
        for it at activation.

        Asking for a new field IS saying Trakt has no field for this column, so
        it is staged like any other answer — as a draft, reversible until the
        set is committed, and released again if the ask is withdrawn.
        """
        run = agent_case.run
        # The same permission a mapping change needs, rather than
        # `resolve_decision` alone: this now stages an answer, and answering a
        # column the run has already settled is the act `reopen_mapping`
        # governs.
        self._require_mapping_change(run)
        column = str(source_column or "").strip()
        file_name = str(source_file or "").strip()
        if not column or not file_name:
            raise OpsError("OCC_AGENT_COLUMN_REQUIRED",
                           "Name the file and the column this request is "
                           "about.", http_status=400)
        row = self._mapping_row(run, file_name, column)
        if row is None:
            raise OpsError("OCC_AGENT_COLUMN_NOT_FOUND",
                           f"'{column}' is not a column Trakt read in "
                           f"{file_name}.", http_status=404)
        name = _field_registry.validate_field_name(field_name)
        facts = self.facts(agent_case)
        if _field_registry.known_field(name, facts.asset_class):
            raise OpsError(
                "OCC_AGENT_FIELD_ALREADY_REGISTERED",
                f"Trakt already reports on '{name}'. Map the column to it "
                "instead of requesting it again.", http_status=409)
        at = now_iso()
        request = _field_registry.field_request(
            source_file=file_name, source_column=column, field_name=name,
            label=label, description=description, data_type=data_type,
            actor=actor, at=at,
            samples=self._sample_values(run, file_name, column))
        # SUPERSEDED BY WHAT IT IS ABOUT, not by its id. A request IS its
        # (file, column): asking twice about one column is one ask, restated.
        # Keying on the id made that true only while the id scheme held still,
        # and it has since changed — so a request made under the old scheme
        # would not have been replaced by the same ask under the new one, and
        # the case would carry the column twice.
        run.field_requests = [
            r for r in run.field_requests
            if _staging.key(r.get("source_file"), r.get("source_column"))
            != _staging.key(file_name, column)] + [request]
        # SET THE COLUMN ASIDE, so the commit leaves it out. An answer the
        # operator has already given by hand is left exactly as it is: they
        # have read this column and said what it is, and a request about it
        # does not overrule that — nor should withdrawing the request later
        # release a set-aside they meant.
        existing = _staging.find(run.staged_mappings, file_name, column)
        if existing is None or str(existing.get("action") or "") != \
                _staging.ACTION_NOT_USED:
            self.stage_mapping(
                agent_case, source_file=file_name, source_column=column,
                action=_staging.ACTION_NOT_USED, actor=actor,
                origin=_staging.ORIGIN_REQUEST,
                reason=f"an operator asked for a new field, '{name}', for "
                       f"this column")
        self.store.save(run)
        self._audit(run, "field_registry_requested", actor_type=ACTOR_HUMAN,
                    actor=actor, classification=EXEC_HUMAN_CONFIRMED,
                    input_reference=request["request_id"],
                    decision_basis=(reason or description
                                    or "an operator asked for a canonical "
                                       "field the registry does not have"),
                    detail={"field_name": name, "source_file": file_name,
                            "source_column": column,
                            "route": request["route"]})
        return agent_case

    def declare_source_unit(self, agent_case: AgentCase, *,
                            field: str, unit: str, actor: str) -> AgentCase:
        """Say which scale the lender writes a percentage on.

        THE REMEDY A BLOCKER HAD NO ROUTE TO. A lender stating loan-to-value as
        ``0.35`` where the platform means ``35`` fails every consistency check
        on the field, at an error rate the policy escalates to BLOCKING — and
        nothing on that screen let an operator say which scale was meant. The
        transform reconciles the two where it can see a balance and a valuation
        to reconcile against; where it cannot, this is the answer.

        NEVER INFERRED FROM MAGNITUDE. "Small numbers must be fractions" is
        wrong for a genuinely small ratio and the mistake is invisible
        afterwards, so the platform declines to guess and asks instead.

        An empty ``unit`` withdraws the declaration and puts the field back to
        reconciliation, which is the default and is usually right.
        """
        run = agent_case.run
        # The same permission a mapping change needs: this changes how a
        # confirmed column is READ, which is the act `reopen_mapping` governs.
        self._require_mapping_change(run)
        name = str(field or "").strip()
        said = str(unit or "").strip().lower()
        allowed = _execution.percentage_scaled_fields()
        if name not in allowed:
            raise OpsError(
                "OCC_AGENT_NOT_A_PERCENTAGE_FIELD",
                f"'{name}' is not a field Trakt holds on a percentage scale, "
                "so there is no unit to declare for it.", http_status=400)
        if said and said not in _execution.SOURCE_UNITS:
            raise OpsError(
                "OCC_AGENT_UNKNOWN_SOURCE_UNIT",
                f"'{unit}' is not a scale Trakt reads. Use "
                f"{' or '.join(_execution.SOURCE_UNITS)}.", http_status=400)
        units = dict(run.source_units or {})
        if said:
            units[name] = said
        else:
            units.pop(name, None)
        run.source_units = units
        self.store.save(run)
        self._audit(run, "source_unit_declared", actor_type=ACTOR_HUMAN,
                    actor=actor, classification=EXEC_HUMAN_CONFIRMED,
                    input_reference=name,
                    decision_basis=(f"the lender writes {name} as {said}"
                                    if said else
                                    f"the declared scale for {name} was "
                                    "withdrawn; Trakt reconciles it again"),
                    detail={"field": name, "source_unit": said})
        return agent_case

    def withdraw_registry_field_request(self, agent_case: AgentCase, *,
                                        source_file: str, source_column: str,
                                        actor: str, reason: str = ""
                                        ) -> AgentCase:
        """Take back an ask, and with it the set-aside the ask imposed.

        The request is what put this column out of the delivery, so withdrawing
        it has to put the column back — otherwise taking back a mistaken ask
        would silently leave the column unused, which is the opposite of what
        the operator meant and invisible until the report came back short.

        Only a set-aside THIS request wrote is released. One the operator
        staged by hand stands: they said the column feeds nothing, and an
        unrelated ask being withdrawn is not them changing their mind.
        """
        run = agent_case.run
        self._require_mapping_change(run)
        file_name = str(source_file or "").strip()
        column = str(source_column or "").strip()
        withdrawn = self._withdraw_field_request(
            run, file_name, column, actor=actor, at=now_iso(),
            why=reason or "withdrawn by the operator")
        if not withdrawn:
            raise OpsError("OCC_AGENT_FIELD_REQUEST_NOT_FOUND",
                           "There is no open field request for that column.",
                           http_status=404)
        if _staging.is_request_driven(
                _staging.find(run.staged_mappings, file_name, column)):
            run.staged_mappings = [
                e for e in run.staged_mappings
                if _staging.key(e.get("source_file"), e.get("source_column"))
                != _staging.key(file_name, column)]
        self.store.save(run)
        self._audit(run, "field_registry_request_withdrawn",
                    actor_type=ACTOR_HUMAN, actor=actor,
                    classification=EXEC_HUMAN_CONFIRMED,
                    input_reference=withdrawn["request_id"],
                    decision_basis=reason or "withdrawn by the operator")
        return agent_case

    @staticmethod
    def _withdraw_field_request(run: SyntheticRun, source_file: str,
                                source_column: str, *, actor: str, at: str,
                                why: str) -> Optional[Dict[str, Any]]:
        """Mark an open request withdrawn. Kept, never deleted: an ask that was
        made and taken back is part of the record of what happened."""
        for request in run.field_requests:
            if (request.get("source_file") == source_file
                    and request.get("source_column") == source_column
                    and request.get("status") == _field_registry.REQUEST_OPEN):
                request["status"] = _field_registry.REQUEST_WITHDRAWN
                request["withdrawn_by"] = actor
                request["withdrawn_at"] = at
                request["withdrawn_because"] = why
                return request
        return None

    @staticmethod
    def _mapping_row(run: SyntheticRun, source_file: str,
                     source_column: str) -> Optional[Dict[str, Any]]:
        """The report row for one column of one file, or None if Trakt never
        read it. Stops a mapping being recorded against a column that is not
        in the delivery — a typo would otherwise promote into a governed rule
        that silently matches nothing every month."""
        for row in run.mapping_report or []:
            if (str(row.get("source_file") or "") == source_file
                    and str(row.get("source_column") or "") == source_column):
                return row
        return None

    def _sample_values(self, run: SyntheticRun, source_file: str,
                       source_column: str) -> List[str]:
        """A few real values from the column, for whoever reads the request.

        Best effort: the files are rebuilt from durable storage and may not be
        present on this instance, and a request with no samples is still a
        request worth recording.
        """
        try:
            for path in self._artefact_paths(run):
                if path.name != source_file:
                    continue
                table = _workbook.read_table(path)
                if table.frame is None or source_column not in table.frame:
                    return []
                series = table.frame[source_column].dropna().astype(str)
                seen: List[str] = []
                for value in series:
                    text = value.strip()
                    if text and text not in seen:
                        seen.append(text)
                    if len(seen) >= 5:
                        break
                return seen
        except Exception:                  # noqa: BLE001 — never a crash here
            return []
        return []

    def acknowledge_exception(self, agent_case: AgentCase, *, decision_id: str,
                              actor: str, reason: str = "") -> AgentCase:
        """Acknowledge a NON-blocking exception.

        A blocking exception is refused here: acknowledgement is not a
        resolution, and a deterministic blocker cannot be talked past.
        """
        run = agent_case.run
        self._require_action(run, _states.ACTION_ACKNOWLEDGE_EXCEPTION)
        target = next((d for d in run.open_decisions
                       if d.get("decision_id") == decision_id), None)
        if target is None:
            raise OpsError("OCC_AGENT_DECISION_NOT_FOUND",
                           "That decision could not be found on this case.",
                           http_status=404)
        if target.get("blocking"):
            raise OpsError(
                "OCC_AGENT_BLOCKING_NOT_ACKNOWLEDGEABLE",
                "This is a blocking control. It has to be resolved, not "
                "acknowledged.", http_status=409)
        target["status"] = "acknowledged"
        target["resolved_by"] = actor
        target["resolved_at"] = now_iso()
        self.store.save(run)
        self._audit(run, "exception_acknowledged", actor_type=ACTOR_HUMAN,
                    actor=actor, classification=EXEC_HUMAN_CONFIRMED,
                    input_reference=decision_id, decision_basis=reason)
        return agent_case

    # ------------------------------------------------------------------ #
    # Planning and readiness
    # ------------------------------------------------------------------ #
    def generate_orchestration_plan(self, agent_case: AgentCase, *,
                                    actor: str) -> AgentCase:
        from engine.orchestrator_agent.orchestrator import (
            onboarding_mode_for_target,
            steps_for_target,
        )
        run = agent_case.run
        self._require_action(run, _states.ACTION_GENERATE_PLAN)
        facts = self.facts(agent_case)
        target = "regime" if facts.regime else "mi"
        # The step sequence is the conductor's own, not a list written here.
        step_names = list(steps_for_target(target, full_pipeline=True))
        produces = {
            "onboard": ["18_central_lender_tape.csv",
                        "24_onboarding_handoff_manifest.json"],
            "transform": ["31_transformed_canonical_tape.csv",
                          "30_transformation_manifest.json"],
            "validate": ["40_validation_manifest.json"],
            "stamp": [f"{facts.portfolio_id}_canonical_typed.csv"],
        }
        steps = [{"step": name, "agent": _AGENT_FOR_STEP.get(name, ""),
                  "produces": produces.get(name, []),
                  "observed_outcome": run.stage_outcomes.get(name, "")}
                 for name in step_names]
        steps.append({"step": "assemble", "agent": "Assembler Agent",
                      "produces": ["platform_canonical_typed.csv"],
                      "observed_outcome": run.stage_outcomes.get("assemble",
                                                                 "")})
        if target == "mi":
            steps.append({"step": "route", "agent": "MI route",
                          "produces": [],
                          "observed_outcome": run.stage_outcomes.get("route",
                                                                     "")})
        else:
            steps.append({"step": "project", "agent": "Regime projector",
                          "produces": [f"central_{facts.regime}_projected.csv"],
                          "observed_outcome": run.stage_outcomes.get("project",
                                                                     "")})
        run.orchestration_plan = {
            "target": target, "outcome": facts.outcome, "regime": facts.regime,
            "onboarding_mode": onboarding_mode_for_target(target),
            "steps": steps,
            "valid": all(s["observed_outcome"] for s in steps),
            "source": "engine.orchestrator_agent.orchestrator",
            "execution_status": "not_executed",
        }
        run.assembler_plan = self._assembler_prerequisites(run)
        prior = self._move(run, _states.ORCHESTRATION_PLAN_GENERATED)
        self.store.save(run)
        self._audit(run, "orchestration_plan_generated", actor_type=ACTOR_AGENT,
                    actor=actor, prior_state=prior,
                    classification=EXEC_DETERMINISTIC,
                    decision_basis="sequence taken from the orchestration "
                                   "conductor; nothing was executed",
                    detail={"steps": [s["step"] for s in steps]})
        if not run.assembler_plan.get("satisfied"):
            return self._block(agent_case,
                               run.assembler_plan.get("problems", []),
                               actor=actor,
                               reason="assembler prerequisites not satisfied")
        self._move(run, _states.EXECUTION_APPROVAL_REQUIRED)
        self.store.save(run)
        return agent_case

    def _assembler_prerequisites(self, run: SyntheticRun) -> Dict[str, Any]:
        """Check the Assembler Agent's real prerequisites against the run."""
        from engine.platform_assembler import LOAN_KEY_FIELDS
        problems: List[str] = []
        mapped = {m.get("canonical_field") for m in run.mapping_report
                  if m.get("canonical_field")}
        if not (set(LOAN_KEY_FIELDS) & mapped):
            problems.append(
                "The Assembler needs a loan identifier "
                f"({' or '.join(LOAN_KEY_FIELDS)}) and none was mapped.")
        if run.stage_outcomes.get("assemble") != STAGE_DETERMINISTIC_COMPLETED:
            problems.append("The practice run did not produce an assembled "
                            "canonical.")
        return {
            "prerequisites": ["a stamped per-portfolio canonical",
                              f"a loan identity field "
                              f"({' or '.join(LOAN_KEY_FIELDS)})",
                              "unique (source_portfolio_id + loan_identifier)"],
            "source": "engine.platform_assembler",
            "satisfied": not problems,
            "problems": problems,
            "summary": ("Assembler prerequisites are satisfied."
                        if not problems else "; ".join(problems)),
        }

    def evaluate_readiness(self, agent_case: AgentCase) -> Dict[str, Any]:
        return self._verdict(agent_case).to_dict()

    def _verdict(self, agent_case: AgentCase) -> _readiness.ReadinessVerdict:
        return _readiness.evaluate(
            agent_case.run, agent_case.case, self.facts(agent_case),
            self.policy, onboarding=self.onboarding_readiness(agent_case),
            preview=self._safe_preview(agent_case))

    def _safe_preview(self, agent_case: AgentCase) -> Dict[str, Any]:
        """The preview, or an empty one when the case cannot yet produce it.

        A case with no client identifier cannot be previewed; that is a
        readiness finding, not an error, so it must not become an exception on
        the status route.
        """
        try:
            return self.preview(agent_case)
        except OpsError:
            return {}

    def approve_execution_readiness(self, agent_case: AgentCase, *,
                                    actor: str) -> AgentCase:
        run = agent_case.run
        self._require_action(run, _states.ACTION_APPROVE_EXECUTION)
        run.approvals.append({
            "approval_id": new_id("appr"), "subject": "execution_readiness",
            "decision": "approved", "actor": actor, "at": now_iso()})
        # The approval is one criterion, not the verdict: readiness is
        # re-derived AFTER it is recorded, and only a full pass moves the run.
        verdict = self._verdict(agent_case)
        run.readiness = verdict.to_dict()
        if not verdict.ready:
            run.readiness_status = "NOT_READY"
            self.store.save(run)
            self._audit(run, "readiness_refused", actor_type=ACTOR_SYSTEM,
                        actor=actor, classification=EXEC_DETERMINISTIC,
                        decision_basis="deterministic criteria are not all "
                                       "satisfied",
                        detail={"outstanding": [c.key
                                                for c in verdict.outstanding]})
            return self._block(
                agent_case, [c.remedy or c.detail for c in verdict.outstanding],
                actor=actor, reason="readiness criteria not satisfied")
        prior = self._move(run, _states.READY_FOR_EXECUTION)
        run.readiness_status = _states.READY_FOR_EXECUTION
        package = self.readiness_package(agent_case, verdict=verdict)
        path = self.store.package_dir(run.tenant, run.case_ref) \
            / "readiness_package.json"
        path.write_text(json.dumps(package, indent=2, default=str),
                        encoding="utf-8")
        run.readiness_package_ref = self.store.relative(run.tenant,
                                                        run.case_ref, path)
        self.store.save(run)
        self._audit(run, "ready_for_execution", actor_type=ACTOR_SYSTEM,
                    actor=actor, prior_state=prior,
                    classification=EXEC_DETERMINISTIC,
                    output_reference=run.readiness_package_ref,
                    decision_basis="every readiness criterion passed "
                                   "deterministically",
                    detail={"manifest_hash":
                            package["execution_manifest"]["content_hash"]})
        return agent_case

    def readiness_package(self, agent_case: AgentCase, *,
                          verdict: Optional[_readiness.ReadinessVerdict] = None
                          ) -> Dict[str, Any]:
        run = agent_case.run
        return _readiness.build_package(
            run, agent_case.case, self.facts(agent_case),
            verdict or self._verdict(agent_case),
            self.store.list_audit(run.tenant, run.case_ref), self.policy,
            onboarding=self.onboarding_readiness(agent_case),
            preview=self._safe_preview(agent_case))

    # ------------------------------------------------------------------ #
    # Human review, and the road to production
    # ------------------------------------------------------------------ #
    def build_review_package(self, agent_case: AgentCase
                             ) -> _review.ReviewPackage:
        """Everything the approver needs. Derived; writes nothing."""
        run = agent_case.run
        return _review.build(
            agent_case.case, run, self.facts(agent_case),
            cat=self.onboarding.catalogue,
            readiness=self._verdict(agent_case).to_dict(),
            preview=self._safe_preview(agent_case),
            onboarding=self.onboarding_readiness(agent_case),
            intent=self._intent(agent_case).to_dict(),
            audit=self.store.list_audit(run.tenant, run.case_ref))

    def request_activation(self, agent_case: AgentCase, *,
                           actor: str) -> AgentCase:
        """Put the complete review package in front of a human.

        The rehearsal passing is not permission to do anything. This is where
        the process turns from "would this work" into "shall we", and the thing
        the human is asked to decide on is the package, not a button.
        """
        run = agent_case.run
        self._require_action(run, _states.ACTION_REQUEST_ACTIVATION)
        package = self.build_review_package(agent_case)
        run.review_package_ref = self.store.write_package(
            run.tenant, run.case_ref, "review_package.json",
            json.dumps(package.to_dict(), indent=2, default=str))
        self.store.write_package(run.tenant, run.case_ref, "review_package.md",
                                 package.document())
        prior = self._move(run, _states.READY_FOR_REVIEW)
        self.store.save(run)
        self._audit(run, "review_package_assembled", actor_type=ACTOR_AGENT,
                    actor=actor, prior_state=prior,
                    classification=EXEC_DETERMINISTIC,
                    output_reference=run.review_package_ref,
                    decision_basis="the review package was derived from the "
                                   "case, the catalogue, the run and the audit "
                                   "trail",
                    detail={"content_hash": package.content_hash,
                            "outstanding": len(package.outstanding),
                            "operator_actions": len(package.operator_actions)})
        return agent_case

    def _record_sample_from_pack(self, agent_case: AgentCase, *,
                                 actor: str) -> None:
        """Tell the onboarding case what the pack on this run actually is.

        Recorded, never fatal — a sample that cannot be registered must not
        stop an approval.
        """
        if agent_case.case.status in TERMINAL:
            return
        try:
            agent_case.case = self.onboarding.register_sample(
                case_id=agent_case.case_ref,
                files=sample_manifest(
                    self.artefacts.classify(agent_case.run.artefacts())[0]),
                by=actor)
        except OpsError as exc:
            logger.warning("occ_agent: sample not registered for %s: %s",
                           agent_case.run.case_ref, exc)

    def _clear_settled_blockers(self, agent_case: AgentCase) -> None:
        """``run.blockers`` is the CURRENT obstacle, not a history of them.

        It was written by :meth:`_block` and by an activation failure, and
        cleared by nothing. So a case that recovered went on showing what used
        to be wrong: a readiness panel reading "13 of 13 criteria passed,
        blocking exceptions cleared" beside a "What's in the way" naming a
        check that no longer blocks — and, once volume stopped promoting a
        warning to BLOCKING, naming a materiality the platform can no longer
        produce. The history is in the audit trail, which is where a history
        belongs.
        """
        run = agent_case.run
        if run.blockers and self._verdict(agent_case).ready:
            run.blockers = []

    def approve_activation(self, agent_case: AgentCase, *, actor: str,
                           reason: str = "") -> AgentCase:
        """A human approves the configuration for activation.

        **This does not start anything.** It records a decision and prepares the
        confirmation. Production needs a second, explicit act — which is the
        whole reason the two are separate states.
        """
        run = agent_case.run
        self._require_action(run, _states.ACTION_APPROVE_ACTIVATION)
        if not run.review_package_ref:
            raise OpsError(
                "OCC_AGENT_NO_REVIEW_PACKAGE",
                "There is no review package to approve. Submit the case for "
                "review first.", http_status=409)
        run.approvals.append({
            "approval_id": new_id("appr"), "subject": "configuration",
            "decision": "approved", "actor": actor, "at": now_iso(),
            "reason": reason or "Configuration approved for activation.",
            "review_package_ref": run.review_package_ref})
        prior = self._move(run, _states.APPROVED_FOR_ACTIVATION)
        # THE PACK THIS CONFIGURATION IS FOR, stated once more before the
        # expectation derived from it is written. Every other point that
        # records the sample — an upload, a removal, the start of a run — is
        # unreachable from here: no state past the rehearsal permits
        # ACTION_REGISTER_ARTEFACT or ACTION_RUN_ONBOARDING, so an operator
        # whose expected delivery was recorded from a smaller pack had nothing
        # left to press but `reopen_mapping`, which withdraws the approvals
        # they had just spent the session earning.
        self._record_sample_from_pack(agent_case, actor=actor)
        self._clear_settled_blockers(agent_case)
        intent = self._intent(agent_case)
        run.activation_intent = intent.to_dict()
        self._move(run, _states.ACTIVATION_CONFIRMATION_REQUIRED)
        self.store.save(run)
        self._audit(run, "activation_approved", actor_type=ACTOR_HUMAN,
                    actor=actor, prior_state=prior,
                    classification=EXEC_HUMAN_CONFIRMED,
                    input_reference=run.review_package_ref,
                    decision_basis=reason or "a human approved the "
                                             "configuration; nothing was "
                                             "started",
                    detail={"started": False,
                            "requires": _states.ACTION_CONFIRM_ACTIVATION})
        return agent_case

    def activation_confirmation(self, agent_case: AgentCase) -> Dict[str, Any]:
        """What confirming would do, and whether it may be confirmed at all."""
        pre = self.activation_preconditions(agent_case, confirmed=False)
        return {
            "intent": self._intent(agent_case).to_dict(),
            "preconditions": pre.to_dict(),
            # The confirmation itself is deliberately excluded from the reasons
            # shown here: the operator has not given it yet, and listing it
            # would read as a failure rather than as the next step.
            "refusals": [r for r in _adapters.activation_refusals(pre)
                         if "final confirmation" not in r],
            "mode": self.adapter.mode,
            "live_enabled": _adapters.live_enabled(),
        }

    def confirm_activation(self, agent_case: AgentCase, *, actor: str,
                           confirmation: str = "") -> AgentCase:
        """The one call that can reach production, through the one gate.

        Every precondition is assembled here and checked in
        :func:`~operations_control.occ_agent.adapters.assert_may_activate`. In
        a rehearsal the synthetic adapter refuses through the policy, which is
        audited — so the refusal is exercised rather than assumed.
        """
        run = agent_case.run
        self._require_action(run, _states.ACTION_CONFIRM_ACTIVATION)
        pre = self.activation_preconditions(agent_case, confirmed=True)
        intent = self._intent(agent_case)
        run.activation_intent = intent.to_dict()

        if not self.adapter.may_activate():
            # Rehearsal. Name the capability, take the audited refusal, and
            # report every reason rather than only the first.
            self._audit(run, "activation_refused", actor_type=ACTOR_SYSTEM,
                        actor=actor, classification=EXEC_BLOCKED,
                        decision_basis="live execution is not available in "
                                       "this environment",
                        detail={"reasons":
                                _adapters.activation_refusals(pre)})
            try:
                self.adapter.activate(pre=pre, intent=intent, actor=actor)
            except _adapters.ActivationRefused:
                raise
            except OpsError:
                raise _adapters.ActivationRefused(
                    _adapters.activation_refusals(pre)) from None
            raise _adapters.ActivationRefused(          # pragma: no cover
                _adapters.activation_refusals(pre))

        _adapters.assert_may_activate(pre)
        prior = self._move(run, _states.ACTIVATING)
        self.store.save(run)
        self._audit(run, "activation_started", actor_type=ACTOR_HUMAN,
                    actor=actor, prior_state=prior,
                    classification=EXEC_HUMAN_CONFIRMED,
                    decision_basis=confirmation or intent.statement,
                    detail={"files": len(intent.files),
                            "targets": intent.target_locations})

        # The doorway. The approved answers cross into the governed container
        # here and nowhere else, immediately before the adapter activates them
        # there. Ordering matters: a failure to promote must stop the
        # activation, not leave a delivery running against a configuration the
        # governed side never received.
        if self.adapter.mode == _adapters.MODE_LIVE:
            _promotion.promote(source=self.onboarding,
                               target=self.live_onboarding,
                               case_ref=run.case_ref, actor=actor)
            self._audit(run, "case_promoted_to_live", actor_type=ACTOR_SYSTEM,
                        actor=actor, classification=EXEC_HUMAN_CONFIRMED,
                        decision_basis="the approved answers were copied into "
                                       "the governed store for activation",
                        output_reference=run.case_ref)

        # The run's decisions travel with the activation so the mappings a
        # human settled during the rehearsal become governed rules the ingest
        # can read. The LIVE adapter promotes them; the synthetic one discards
        # them, so an unactivated rehearsal still leaves nothing behind.
        # The run's field requests travel with it too. The LIVE adapter turns
        # them into a draft system configuration version for a configuration
        # owner; the synthetic one discards them, so an unactivated rehearsal
        # leaves no proposal behind either.
        result = self.adapter.activate(
            pre=pre, intent=intent, actor=actor,
            payloads=self._payloads(run),
            decisions=list(run.open_decisions or []),
            field_requests=list(run.field_requests or []))
        run.activation_result = result.to_dict()
        if result.ok:
            self._move(run, _states.INGESTION_STARTED)
            if self.adapter.mode == _adapters.MODE_LIVE:
                # The activation stamp was written on the governed side; carry
                # it back so the practice case — which every precondition reads
                # — no longer reads as merely approved.
                activated = self.live_onboarding.load_case(run.case_ref)
                _promotion.record_activation(source=self.onboarding,
                                             activated=activated)
                agent_case.case = activated
            else:
                agent_case.case = self.onboarding.load_case(run.case_ref)
        else:
            self._move(run, _states.ACTIVATION_FAILED)
            run.blockers = [result.error or "Activation failed."]
        self.store.save(run)
        self._audit(run,
                    "ingestion_started" if result.ok else "activation_failed",
                    actor_type=ACTOR_SYSTEM, actor=actor,
                    classification=(EXEC_DETERMINISTIC if result.ok
                                    else EXEC_BLOCKED),
                    output_reference=result.workflow_id,
                    decision_basis=result.message,
                    detail=result.to_dict())
        return agent_case

    def activation_preconditions(self, agent_case: AgentCase, *,
                                 confirmed: bool) -> ActivationPreconditions:
        """Assemble the gate's inputs from state this service already holds.

        Assembling and checking are deliberately separate: this method reads,
        :func:`assert_may_activate` decides. A caller cannot reach production by
        assembling a friendlier set of facts, because every one of them comes
        from a persisted record.
        """
        run, case = agent_case.run, agent_case.case
        facts = self.facts(agent_case)
        verdict = self._verdict(agent_case)
        preview = self._safe_preview(agent_case)
        problems = [str(p.get("detail") or p)
                    for p in (self.onboarding_readiness(agent_case).get(
                        "blocking") or [])]
        roles = self.artefacts.readiness(run, facts.outcome)
        approval = run.approval("configuration")
        audited = any(
            e.get("action") == "activation_approved"
            for e in self.store.list_audit(run.tenant, run.case_ref))
        return ActivationPreconditions(
            mode=run.mode,
            flag_enabled=_adapters.live_enabled(),
            case_ref=run.case_ref,
            onboarding_status=case.status,
            configuration_approved=(approval.get("decision") == "approved"),
            readiness_passed=verdict.ready,
            tenant=run.tenant,
            client_id=facts.client_id,
            portfolio_id=facts.portfolio_id,
            asset_class=facts.asset_class,
            configuration_valid=bool(preview.get("artefacts")) and not problems,
            configuration_problems=problems,
            artefacts_present=len(run.received_artefacts),
            required_artefacts_satisfied=roles.ready,
            approval_audited=audited,
            already_activated=bool(case.activated_version),
            confirmed=confirmed)

    def _intent(self, agent_case: AgentCase) -> _adapters.ActivationIntent:
        """What activation would cause, built from the case and the files."""
        run = agent_case.run
        facts = self.facts(agent_case)
        files = [{"name": a.source_file, "target": a.intended_live_uri,
                  "sha256": a.sha256} for a in run.artefacts()]
        preview = self._safe_preview(agent_case)
        # WHAT WOULD BE PROMOTED, COUNTED WITHOUT PROMOTING IT.
        # `rules_from` builds the records and persists nothing, for exactly
        # this: showing an operator what activation would add to the client's
        # standing rules before they agree to it.
        promotable = _mapping_promotion.rules_from(
            list(run.open_decisions or []),
            client_id=facts.client_id, portfolio_id=facts.portfolio_id,
            workflow_id="")
        return _adapters.build_intent(
            agent_case.case, facts, reporting_period=run.reporting_period,
            files=files,
            configuration_artefacts=[str(a.get("path") or a.get("name") or a)
                                     for a in (preview.get("artefacts") or [])],
            mappings=len(promotable),
            field_requests=len(run.field_requests or []))

    def _payloads(self, run: SyntheticRun) -> Dict[str, bytes]:
        """The bytes behind the intent's files, rebuilt from durable storage.

        Read at the moment of activation rather than carried through the
        approval, so what is placed is what the case actually holds — and so an
        instance that never saw the upload can still perform it.
        """
        self.store.materialise(run.tenant, run.case_ref)
        out: Dict[str, bytes] = {}
        base = self.store.artefact_dir(run.tenant, run.case_ref)
        for artefact in run.artefacts():
            path = base / artefact.source_file
            if path.exists():
                out[artefact.source_file] = path.read_bytes()
        return out

    # ------------------------------------------------------------------ #
    # Cancelling
    # ------------------------------------------------------------------ #
    def cancel(self, agent_case: AgentCase, *, actor: str,
               reason: str = "") -> AgentCase:
        run = agent_case.run
        self._require_action(run, _states.ACTION_CANCEL)
        prior = self._move(run, _states.CANCELLED)
        self.store.save(run)
        self._audit(run, "practice_case_cancelled", actor_type=ACTOR_HUMAN,
                    actor=actor, prior_state=prior,
                    classification=EXEC_HUMAN_CONFIRMED,
                    decision_basis=reason or "cancelled by the operator")
        return self.withdraw(agent_case, actor=actor,
                             reason=reason or "The practice case was "
                                              "cancelled.")

    def withdraw(self, agent_case: AgentCase, *, actor: str,
                 reason: str) -> AgentCase:
        """End the onboarding case without creating anything."""
        if agent_case.case.status in TERMINAL:
            return agent_case
        agent_case.case = self.onboarding.withdraw(
            case_id=agent_case.case_ref, by=actor, reason=reason)
        return agent_case

    # ------------------------------------------------------------------ #
    # The natural-language door
    # ------------------------------------------------------------------ #
    def instruct(self, agent_case: AgentCase, *, text: str, actor: str,
                 confirm: bool = False) -> TurnResult:
        """One natural-language turn.

        A non-material action is applied straight away. A material one comes
        back as a proposal the human confirms — which is what keeps "natural
        language must not override a governed control" true in practice.
        """
        run = agent_case.run
        run.messages.append(Message(role="operator",
                                    text=text[:4000]).to_dict())
        try:
            change = self.interpreter.interpret_action(text, run,
                                                       agent_case.case)
        except InterpretationError:
            self.store.save(run)
            raise
        change.validate()

        if change.action == _states.ACTION_ASK:
            reply = self.answer(agent_case, change.payload.get("question", ""))
            run.messages.append(Message(role="agent", text=reply).to_dict())
            self.store.save(run)
            return TurnResult(case=agent_case, reply=reply)

        self._require_action(run, change.action)

        # An instruction that answers questions is planned before anything is
        # decided about it, so what the turn reports is what the plan actually
        # found — not what the sentence looked like. A plan that could not be
        # read in full is always put to the human, whatever the action's own
        # materiality says.
        plan: Optional[ApplicationPlan] = None
        if change.action == _states.ACTION_ANSWER:
            raw = change.payload.get("interpretation") or {}
            plan = self.plan_interpretation(agent_case, Interpretation(
                **{k: v for k, v in raw.items()
                   if k in Interpretation.__dataclass_fields__}))
            if plan.material:
                change.requires_confirmation = True
                change.summary = plan.summary()

        if change.requires_confirmation and not confirm:
            proposal = {"proposal_id": new_id("prop"), "action": change.action,
                        "payload": change.payload, "summary": change.summary,
                        "basis": change.basis, "material": change.material,
                        "confidence": change.confidence}
            if plan is not None:
                proposal["disclosure"] = plan.disclosure()
                proposal["plan"] = plan.to_dict()
                proposal["complete"] = plan.complete
            run.messages.append(Message(
                role="agent",
                text=self._proposal_text(proposal, plan),
                refs=[proposal["proposal_id"]]).to_dict())
            self.store.save(run)
            self._audit(run, "change_proposed", actor_type=ACTOR_AGENT,
                        actor=actor, classification=EXEC_MODEL_PROPOSED,
                        decision_basis=change.basis,
                        detail={"action": change.action})
            return TurnResult(case=agent_case,
                              reply=self._proposal_text(proposal, plan),
                              proposal=proposal)

        if plan is not None:
            agent_case = self.apply_plan(agent_case, plan=plan, actor=actor,
                                         confirm=confirm)
            reply = self.describe_plan(agent_case, plan)
            agent_case.run.messages.append(
                Message(role="agent", text=reply).to_dict())
            self.store.save(agent_case.run)
            return TurnResult(case=agent_case, reply=reply, applied=True,
                              proposal={"disclosure": plan.disclosure()},
                              decisions=agent_case.run.open_decisions)

        agent_case = self._apply(agent_case, change, actor=actor)
        reply = self.status_sentence(agent_case)
        agent_case.run.messages.append(
            Message(role="agent", text=reply).to_dict())
        self.store.save(agent_case.run)
        return TurnResult(case=agent_case, reply=reply, applied=True,
                          decisions=agent_case.run.open_decisions)

    @staticmethod
    def _proposal_text(proposal: Dict[str, Any],
                       plan: Optional[ApplicationPlan]) -> str:
        """What the agent says when it wants a human to look.

        The four populations are always reported in the same order, so an
        operator learns where to look for "and what did you NOT understand?".
        """
        if plan is None:
            return f"Proposed: {proposal['summary']} Confirm to apply."
        lines: List[str] = []
        understood = plan.disclosure()["understood"]
        if understood:
            lines.append("I understood:")
            lines += [f"- {line}" for line in understood]
        for question in plan.questions:
            lines.append(f"- {question.question}"
                         + (f" ({', '.join(question.candidates)})"
                            if question.candidates else ""))
        if plan.unrecognised:
            lines.append("I could not read:")
            lines += [f'- "{fragment}"' for fragment in plan.unrecognised]
        if plan.complete:
            lines.append("Confirm to apply.")
        else:
            lines.append("Nothing has been applied. Confirm to apply only "
                         "what is listed above, or tell me the rest.")
        return "\n".join(lines)

    def _apply(self, agent_case: AgentCase, change: ProposedChange, *,
               actor: str) -> AgentCase:
        action = change.action
        payload = change.payload
        if action == _states.ACTION_ANSWER:
            raw = payload.get("interpretation") or {}
            interpretation = Interpretation(
                **{k: v for k, v in raw.items()
                   if k in Interpretation.__dataclass_fields__})
            plan = self.plan_interpretation(agent_case, interpretation)
            # Reaching here means the human has already seen the proposal and
            # confirmed it, disclosed remainder included.
            return self.apply_plan(agent_case, plan=plan, actor=actor,
                                   confirm=True)
        if action == _states.ACTION_REQUEST_INFORMATION:
            return self.request_client_information(agent_case, actor=actor)
        if action == _states.ACTION_RECORD_RESPONSE:
            request_id = payload.get("request_id") or _first_open_request(
                agent_case.case)
            return self.record_client_response(
                agent_case, request_id=request_id, actor=actor,
                answers=payload.get("answers") or {})
        if action == _states.ACTION_SUBMIT_FOR_APPROVAL:
            return self.submit_for_approval(agent_case, actor=actor)
        if action == _states.ACTION_APPROVE_ONBOARDING:
            return self.approve_onboarding(agent_case, actor=actor,
                                           reason=payload.get("reason", ""))
        if action == _states.ACTION_REQUEST_CHANGES:
            return self.request_changes(
                agent_case, actor=actor,
                reason=payload.get("reason") or "Changes requested.")
        if action == _states.ACTION_WITHDRAW:
            return self.withdraw(
                agent_case, actor=actor,
                reason=payload.get("reason") or "Withdrawn by the operator.")
        if action == _states.ACTION_RUN_ONBOARDING:
            return self.run_synthetic_onboarding(agent_case, actor=actor)
        if action == _states.ACTION_RESOLVE_DECISION:
            return self._resolve_from_language(agent_case, payload, actor=actor)
        if action == _states.ACTION_ACKNOWLEDGE_EXCEPTION:
            decision_id = payload.get("decision_id") or \
                _first_non_blocking(agent_case.run)
            return self.acknowledge_exception(agent_case,
                                              decision_id=decision_id,
                                              actor=actor)
        if action == _states.ACTION_GENERATE_PLAN:
            return self.generate_orchestration_plan(agent_case, actor=actor)
        if action == _states.ACTION_APPROVE_EXECUTION:
            return self.approve_execution_readiness(agent_case, actor=actor)
        if action == _states.ACTION_DRAFT_PACK:
            return self.draft_pack(agent_case, actor=actor)
        if action == _states.ACTION_APPROVE_PACK:
            return self.approve_pack(agent_case, actor=actor,
                                     reason=payload.get("reason", ""))
        if action == _states.ACTION_SEND_PACK:
            return self.send_pack(agent_case, actor=actor)
        if action == _states.ACTION_REQUEST_ACTIVATION:
            return self.request_activation(agent_case, actor=actor)
        if action == _states.ACTION_APPROVE_ACTIVATION:
            return self.approve_activation(agent_case, actor=actor,
                                           reason=payload.get("reason", ""))
        if action == _states.ACTION_CONFIRM_ACTIVATION:
            return self.confirm_activation(
                agent_case, actor=actor,
                confirmation=payload.get("confirmation", ""))
        if action == _states.ACTION_CANCEL:
            # The reason is carried, not dropped. This called cancel() with no
            # reason at all, so the audit read "cancelled by the operator" and
            # the withdrawal read "The practice case was cancelled." whatever
            # the operator had actually typed. The governed dialog refuses a
            # blank reason; the chat path collected one and threw it away.
            return self.cancel(agent_case, actor=actor,
                               reason=payload.get("reason", ""))
        raise ActionNotAllowed(action, agent_case.run.state)  # pragma: no cover

    def _resolve_from_language(self, agent_case: AgentCase,
                               payload: Dict[str, Any], *,
                               actor: str) -> AgentCase:
        """Turn 'map X to Y' into a decision resolution on the right decision."""
        source = str(payload.get("source_column") or "")
        canonical = str(payload.get("canonical_field") or "")
        decision = next(
            (d for d in agent_case.run.open_decisions
             if d.get("status", "open") == "open"
             and (str(d.get("subject", {}).get("source_column", "")).lower()
                  == source.lower()
                  or source.lower() in
                  str(d.get("subject", {}).get("source_columns", "")).lower())),
            None)
        if decision is None:
            raise OpsError(
                "OCC_AGENT_DECISION_NOT_FOUND",
                f"There is no open mapping decision for '{source}' on this "
                "case.", http_status=404)
        return self.resolve_decision(
            agent_case, decision_id=decision["decision_id"], action="amend",
            value=canonical, actor=actor,
            reason=f"operator mapped '{source}' to '{canonical}'")

    # ------------------------------------------------------------------ #
    # Questions and status
    # ------------------------------------------------------------------ #
    def describe_case(self, agent_case: AgentCase) -> str:
        """What Trakt now holds, in the catalogue's own labels."""
        case = agent_case.case
        facts = self.facts(agent_case)
        cat = self.onboarding.catalogue
        lines: List[str] = [f"Onboarding {case.case_id}."]
        client = case.answers.get("client") or {}
        for key in ("client_name", "client_id", "jurisdiction",
                    "reporting_currency"):
            value = client.get(key)
            if value:
                f = cat.field("client", key)
                lines.append(f"{f.label if f else key}: {value}")
        for portfolio in case.items("portfolios"):
            label = portfolio.get("display_name") or portfolio.get(
                "portfolio_id") or "Portfolio"
            detail = ", ".join(
                str(portfolio.get(k)) for k in
                ("portfolio_id", "asset_class", "portfolio_type")
                if portfolio.get(k))
            lines.append(f"Portfolio: {label}" + (f" ({detail})" if detail
                                                  else ""))
        if facts.products:
            lines.append("Products: " + ", ".join(
                _derive.product_label(p, cat) for p in facts.products))
        if agent_case.run.reporting_period:
            lines.append(f"Reporting period: "
                         f"{agent_case.run.reporting_period}")
        outstanding = self.onboarding.client_checklist(case)
        if outstanding:
            lines.append("Still needed from the client:")
            for row in outstanding[:8]:
                lines.append(f"- {row['label']}")
        return "\n".join(lines)

    def audit_mail_ingest(self, run: SyntheticRun, *, actor: str,
                          message: Any, correlation: Any, result: Any) -> None:
        """Record that a client's reply was taken into this case.

        Classified as ``human_confirmed`` rather than ``deterministic``: an
        operator asked for the mailbox to be read and chose this message, and
        the audit should name the person who did rather than imply the system
        decided on its own. The evidence the message was matched on is recorded
        with it, so "why is this file on this case?" has an answer that does
        not depend on anyone remembering.
        """
        self._audit(run, "client_reply_ingested", actor_type=ACTOR_HUMAN,
                    actor=actor, classification=EXEC_HUMAN_CONFIRMED,
                    input_reference=str(getattr(message, "internet_message_id",
                                                "") or ""),
                    decision_basis=("a reply in the OCC mailbox, matched to "
                                    "this case on "
                                    + ", ".join(getattr(correlation, "bases",
                                                        []) or ["nothing"])),
                    detail={"sender": getattr(message, "sender", ""),
                            "subject": getattr(message, "subject", ""),
                            "received_at": getattr(message, "received_at", ""),
                            "registered": list(getattr(result, "registered",
                                                       [])),
                            "skipped": [s.get("name") for s in
                                        getattr(result, "skipped", [])],
                            "recorded_text": bool(
                                getattr(result, "recorded_text", False))})

    def available_actions(self, agent_case: AgentCase) -> List[str]:
        """Every action an operator could take on this case right now.

        The lifecycle table governs the run's own actions. Client Onboarding
        governs its own, so those are asked of IT — a case whose information is
        still incomplete cannot be submitted for approval however far the run
        has got, and saying otherwise would send an operator to a button that
        refuses them. The two sets are the same union the case screen renders
        its buttons from; keeping the agent's spoken answer on the same rule is
        the point, because "what can I do next" and "what buttons do I have"
        must never be different questions.
        """
        run, case = agent_case.run, agent_case.case
        onboarding = self.onboarding_readiness(agent_case)
        actions = list(_states.spec(run.state).allowed_human_actions)
        if onboarding.get("client_checklist"):
            actions.append(_states.ACTION_REQUEST_INFORMATION)
        if onboarding.get("outstanding_requests"):
            actions.append(_states.ACTION_RECORD_RESPONSE)
        if onboarding.get("ready"):
            if case.status in (DRAFT, INFORMATION_REQUESTED, IN_REVIEW,
                               CHANGES_REQUIRED):
                actions.append(_states.ACTION_SUBMIT_FOR_APPROVAL)
            if case.status in (READY_FOR_APPROVAL, IN_REVIEW):
                actions.append(_states.ACTION_APPROVE_ONBOARDING)
        seen: List[str] = []
        for action in actions:
            if action not in seen:
                seen.append(action)
        return seen

    def pending(self, agent_case: AgentCase) -> Dict[str, Any]:
        """What actually stands between this case and its next milestone.

        Deliberately NOT the readiness table. Readiness lists the criteria for
        the END of a rehearsal, and at the start of a case every one of them is
        unmet — so reading it back as "what is pending" answers with a wall
        whose remedies restate their own labels ("Onboarding approved: work the
        onboarding through to approval"). True, and no use to anyone.

        An operator asking what is pending is asking two things: who is this
        waiting on, and what is the next thing I can do. Both are answered from
        state Client Onboarding and the run already hold — the client
        checklist, the case's own blocking problems, its open information
        requests, the open decisions and the allowed actions. Nothing here is
        inferred and nothing is invented.
        """
        case, run = agent_case.case, agent_case.run
        onboarding = self.onboarding_readiness(agent_case)
        checklist = [dict(row) for row in
                     (onboarding.get("client_checklist") or [])]
        # An item the client has already been ASKED for is still the client's,
        # and the checklist alone does not know that: ``client_checklist``
        # excludes anything sitting in an open information request, so pressing
        # "ask the client" empties it. Reasoning from the checklist alone
        # therefore flipped the whole list from "waiting on them" to "needs
        # you" at the exact moment it became most true that we were waiting on
        # them — and told an operator that seven things were theirs to supply
        # while the request asking the client for those same seven was open.
        #
        # So what is outstanding ON THE CLIENT is the checklist plus the items
        # of every open request that have not since been answered.
        requested = [dict(item) for r in
                     (onboarding.get("outstanding_requests") or [])
                     for item in (r.get("items") or [])]
        still_open = {(row["section"], row["field"], row["index"])
                      for row in self.onboarding.catalogue
                      .outstanding_for_client(case.answers)}
        seen: set = {(row.get("section"), row.get("field"), row.get("index"))
                     for row in checklist}
        for item in requested:
            key = (item.get("section"), item.get("field"), item.get("index"))
            if key in seen or key not in still_open:
                continue        # already listed, or answered since it was asked
            seen.add(key)
            checklist.append({**item, "asked": True})

        # The client list and the case's blocking problems overlap almost
        # entirely — every unanswered client field is both. Listing both in
        # full is how an answer to "what is pending" became nine lines that say
        # the same seven things twice, burying the two that were different. So
        # the problems are reported only where they are NOT already above,
        # which is precisely the set the operator has to supply themselves;
        # that residue is the real answer to "why is this stuck".
        yours = [str(p.get("message") or "")
                 for p in (onboarding.get("blocking") or [])
                 if p.get("message")
                 and (p.get("section"), p.get("field"),
                      p.get("index")) not in seen]
        problems = [str(p.get("message") or "")
                    for p in (onboarding.get("blocking") or [])
                    if p.get("message")]
        requests = [{"request_id": str(r.get("request_id") or ""),
                     "items": len(r.get("items") or [])}
                    for r in (onboarding.get("outstanding_requests") or [])]
        decisions = [str(d.get("title") or d.get("question") or "")
                     for d in run.blocking_decisions()]
        issued = run.pack_status == _comms.SENT
        # Waiting on the CLIENT only once something has actually been issued to
        # them. Before that the questions are outstanding on Trakt, not on
        # them, and telling an operator they are waiting for a client who has
        # never been written to is the kind of false comfort that loses a week.
        waiting_on = "client" if issued and checklist else "you"
        actions = [a for a in self.available_actions(agent_case)
                   if a not in _NOT_A_WAY_FORWARD]
        nxt = [n for n in _states.spec(run.state).next_states
               if n not in (_states.BLOCKED, _states.CANCELLED)]
        return {
            "waiting_on": waiting_on,
            "issued": issued,
            "client_name": str((case.answers.get("client") or {}).get(
                "client_name") or ""),
            "client": checklist,
            "yours": yours,
            "problems": problems,
            "requests": requests,
            "decisions": decisions,
            "blockers": list(run.blockers or []),
            "actions": actions,
            "next_milestone": _states.spec_label(nxt[0]) if nxt else "",
            "finished": not actions and not nxt,
        }

    def pending_sentences(self, agent_case: AgentCase) -> str:
        """:meth:`pending`, in the words an operator would use."""
        p = self.pending(agent_case)
        who = p["client_name"] or "the client"
        lines: List[str] = []

        if p["waiting_on"] == "client":
            lines.append(f"You are waiting on {who}. The pack has been issued "
                         "and these answers have not come back yet.")
        elif p["client"]:
            lines.append(f"You are waiting on yourself, not on {who}: the "
                         "questions below have not been put to them yet.")

        if p["client"]:
            lines.append("")
            lines.append(f"Still needed from {who} "
                         f"({len(p['client'])}):")
            lines += [f"- {row['label']}" for row in p["client"]]

        if p["yours"] or p["decisions"] or p["blockers"]:
            lines.append("")
            lines.append("Needs you, not the client "
                         f"({len(p['yours'] + p['decisions'] + p['blockers'])}"
                         "):")
            lines += [f"- {m}" for m in
                      p["yours"] + p["decisions"] + p["blockers"]]

        if p["client"] or p["yours"]:
            lines.append("")
            lines.append("You do not have to wait for an email to move: type "
                         "the answers into this chat as you get them, or fill "
                         "them in under the client questions.")

        if p["requests"]:
            total = sum(r["items"] for r in p["requests"])
            lines.append("")
            lines.append(
                f"{len(p['requests'])} information request"
                f"{'s' if len(p['requests']) != 1 else ''} covering {total} "
                "item(s) are still open, and the onboarding cannot be "
                "submitted for approval while one is. Each closes itself once "
                "every item it asked for has been answered.")

        lines.append("")
        if p["actions"]:
            lines.append("What you can do now: " + _join(
                [action_phrase(a) for a in p["actions"]]) + ".")
        elif p["finished"]:
            lines.append("This practice case is finished; there is nothing "
                         "further to do.")
        else:
            lines.append("There is nothing you can do from here yet.")
        if p["next_milestone"]:
            lines.append(f"Next milestone: {p['next_milestone']}.")
        return "\n".join(lines).strip()

    def answer(self, agent_case: AgentCase, question: str) -> str:
        """Answer from case state and the lifecycle table. Never invents."""
        lower = question.lower()
        run = agent_case.run
        if "why" in lower and run.open_decisions:
            first = run.blocking_decisions() or run.open_decisions
            d = first[0]
            return (f"{d.get('title') or d.get('question')}\n"
                    f"{d.get('question', '')}\n"
                    f"{(d.get('evidence') or [{}])[0].get('detail', '')}").strip()
        # A question about the CLIENT is answered from the checklist; only a
        # question about the case as a whole falls through.
        if "client" in lower and any(w in lower for w in
                                     ("ask", "need", "send", "outstanding",
                                      "waiting", "chase")):
            checklist = self.onboarding.client_checklist(agent_case.case)
            if not checklist:
                return "There is nothing outstanding from the client."
            return "The client still has to tell us:\n" + "\n".join(
                f"- {row['label']}" for row in checklist)
        # "Readiness" is a named thing in this product — the criteria for
        # READY_FOR_EXECUTION — so a question that names it gets the criteria.
        # A question that does NOT name it is asking what to do next, and gets
        # what is actually pending. Answering both from the criteria table is
        # what once told an operator with an unissued pack that what remained
        # was to "work the onboarding through to approval".
        if "readiness" in lower or "ready for execution" in lower \
                or "criteria" in lower:
            verdict = self._verdict(agent_case)
            if verdict.ready:
                return ("Every readiness criterion is satisfied. Approve "
                        "readiness to reach READY_FOR_EXECUTION.")
            return "Still outstanding:\n" + "\n".join(
                f"- {c.label}: {c.remedy or c.detail}"
                for c in verdict.outstanding)
        if any(w in lower for w in ("pending", "what is left", "what remains",
                                    "still", "outstanding", "waiting",
                                    "stuck", "blocked", "hold", "what's left",
                                    "next", "can i", "what should",
                                    "what now", "do now")):
            return self.pending_sentences(agent_case)
        if "stage" in lower or "where" in lower or "status" in lower:
            return self.status_sentence(agent_case)
        return self.status_sentence(agent_case)

    def status_sentence(self, agent_case: AgentCase) -> str:
        case, run = agent_case.case, agent_case.run
        parts = [f"The onboarding is "
                 f"{STATUS_LABELS.get(case.status, case.status).lower()}; the "
                 f"practice run is at {_states.spec_label(run.state)}."]
        if run.blockers:
            parts.append("In the way: " + "; ".join(run.blockers[:3]))
        blocking = run.blocking_decisions()
        if blocking:
            parts.append(f"{len(blocking)} decision"
                         f"{'s' if len(blocking) != 1 else ''} need"
                         f"{'' if len(blocking) != 1 else 's'} you.")
        checklist = self.onboarding.client_checklist(case)
        if checklist:
            parts.append(f"{len(checklist)} item"
                         f"{'s' if len(checklist) != 1 else ''} still "
                         f"outstanding from the client.")
        return " ".join(parts)

    def status(self, agent_case: AgentCase) -> Dict[str, Any]:
        """The full status projection the tab renders."""
        case, run = agent_case.case, agent_case.run
        verdict = self._verdict(agent_case)
        onboarding = self.onboarding_readiness(agent_case)
        reached = _reached_states(run)
        return {
            "case_ref": case.case_id,
            "run": run.to_dict(),
            "summary": run.summary_row(),
            "onboarding": {
                **self.onboarding.present_case(case),
                "steps": [{"key": s, "label": STEP_LABELS[s],
                           "problems": len(
                               (onboarding.get("by_step") or {}).get(s) or [])}
                          for s in STEPS],
            },
            "facts": self.facts(agent_case).to_dict(),
            # One row per source registration: the operational data streams
            # this onboarding declares, funded and pipeline kept separate.
            "streams": _derive.stream_summaries(case,
                                                self.onboarding.catalogue),
            "state": _states.describe(run.state),
            "lifecycle": [
                {**entry,
                 "reached": entry["state"] in reached,
                 "current": entry["state"] == run.state}
                for entry in _states.lifecycle()],
            "stage_outcomes": run.stage_outcomes,
            "readiness": verdict.to_dict(),
            "policy": self.policy.to_dict(),
            "open_decisions": run.open_decisions,
            # Every source column and what became of it — including the ones
            # the mapper settled on its own, which are the majority and were
            # not visible anywhere before.
            "mapping": _mapping_view.overview(run),
            "observations": run.observations,
            "blockers": run.blockers,
            "occ_links": _occ_links(case, run),
            # The client-facing half: what has been drafted, approved and
            # issued, and — honestly — whether anything actually left Trakt.
            # The drafted pack document is carried WHOLE (confirmations,
            # not-asked, summary and all): re-projecting a subset here is what
            # once made the tab render a pack shape the API never sent.
            "pack": {
                "outstanding": 0,
                "questions": 0,
                "sections": [],
                "email": {},
                "artefacts": [],
                "confirmations": [],
                "not_asked": [],
                "summary": {},
                **{k: v for k, v in (run.pack or {}).items()},
                "status": run.pack_status,
                "history": run.pack_history,
                "mapping_statement": _pack.MAPPING_STATEMENT,
                "receipt": run.pack_receipt,
                "sent": bool((run.pack_receipt or {}).get("sent")),
            },
            "review_package_ref": run.review_package_ref,
            "activation": {
                "mode": run.mode,
                "adapter": self.adapter.mode,
                "live_enabled": _adapters.live_enabled(),
                "intent": run.activation_intent,
                "result": run.activation_result,
                "approval": run.approval("configuration"),
            },
            "running": self.job_running(run.tenant, run.case_ref),
            # Surfaced separately so the tab can never present a simulated or
            # blocked stage as a completed one, nor imply anything was created.
            "anything_simulated": _readiness.anything_simulated(run),
            "anything_blocked": _readiness.anything_blocked(run),
            "configuration_written": case.status not in NO_ACTIVE_CONFIGURATION,
        }

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _block(self, agent_case: AgentCase, blockers: List[str], *, actor: str,
               reason: str) -> AgentCase:
        run = agent_case.run
        run.blockers = [b for b in blockers if b]
        prior = run.state
        if _states.is_transition_allowed(run.state, _states.BLOCKED):
            run.state = _states.BLOCKED
        self.store.save(run)
        self._audit(run, "run_blocked", actor_type=ACTOR_SYSTEM, actor=actor,
                    prior_state=prior, classification=EXEC_BLOCKED,
                    decision_basis=reason, detail={"blockers": run.blockers})
        return agent_case

    @staticmethod
    def _record_control(run: SyntheticRun, kind: str,
                        result: Dict[str, Any]) -> None:
        run.control_results.append({"kind": kind, "at": now_iso(), **result})

    def _artefact_paths(self, run: SyntheticRun) -> List[Path]:
        base = self.store.artefact_dir(run.tenant, run.case_ref)
        paths: List[Path] = []
        for artefact in run.artefacts():
            candidate = base / artefact.source_file
            if candidate.exists():
                paths.append(candidate)
        return paths

    @staticmethod
    def _approved_mappings(run: SyntheticRun) -> Dict[str, str]:
        """``file::column`` -> canonical field, from resolved mapping decisions.

        An empty target means "do not use this column", which is how the losing
        side of an ambiguity is recorded: both columns are answered, so the next
        run has nothing left to ask about.

        KEYED ON THE PAIR, NOT THE NAME. Every file in a pack carries a loan
        identifier and most carry a valuation date, so a flat column-name key
        made an answer about the property extract's 'Pool' an answer about the
        tape's. A decision that predates file-scoping carries no
        ``source_file``; it is keyed on the bare name, which the adapter reads
        as the fallback it always was.
        """
        out: Dict[str, str] = {}
        # A decision about SEVERAL columns is read first, so a per-column
        # answer written later wins. An ambiguity says which of two columns is
        # the balance; each of those columns then has its own record saying
        # what it feeds, and that record is the operator's actual answer about
        # that column. Read the other way round, setting the losing column
        # aside would drop the winning one too.
        for decision in sorted(
                run.open_decisions,
                key=lambda d: len((d.get("subject") or {}).get(
                    "source_columns") or []) < 2):
            if decision.get("status") != "approved":
                continue
            subject = decision.get("subject") or {}
            target = str(subject.get("target_field") or "")
            columns = [str(c) for c in (subject.get("source_columns") or [])]
            primary = str(subject.get("source_column") or "")
            source_file = str(subject.get("source_file") or "")
            if not primary and not columns:
                continue
            value = str(decision.get("resolved_value") or "")
            # A resolution is either a canonical field (the operator named a
            # different target), a competing column (they chose which column
            # wins), or one of the decision's own actions.
            if value in columns:
                chosen, target_field = value, target
            elif value in ("__ignore__", "mark_unavailable"):
                chosen, target_field = primary, ""
            elif value and value not in ("accept_mapping", "confirm_selected",
                                         "approve", ""):
                chosen, target_field = primary, value
            else:
                chosen, target_field = primary, target
            key = (lambda c: _mapping_key(source_file, c) if source_file else c)
            if chosen:
                out[key(chosen)] = target_field
            for column in columns:
                if column != chosen:
                    out[key(column)] = ""  # the losing candidate is not used
        return out

    @staticmethod
    def _decisions_from_run(run: SyntheticRun, facts: ExecutionFacts,
                            run_root: Path) -> List[Dict[str, Any]]:
        """Read pending decisions from the run using the EXISTING extractor.

        ``operations_control.adapters.extract_mapping_decisions`` reads the
        ``34_target_first_decisions.yaml`` artefact the adapter writes, so a
        synthetic decision and a live one are produced by the same code. The
        raw artefact is then read again for the source-column detail the
        operator-facing contract does not carry, so a resolution can be applied
        back to the right column on the rerun.
        """
        from ..adapters import extract_mapping_decisions
        from ..contracts import WorkflowRun

        shim = WorkflowRun(
            workflow_id=run.case_ref, client_id=facts.client_id,
            portfolio_id=facts.portfolio_id, outcome=facts.outcome,
            workflow_type="new_client", delivery={})
        found = extract_mapping_decisions(Path(run_root), shim)
        raw = _raw_decisions(Path(run_root))
        return [_decision_card(d, raw) for d in found]

    @staticmethod
    def _purge_stale_decisions(run_root: Path) -> None:
        """Remove the previous run's pending-decision artefact.

        Without this a decision answered by a human would be re-read from the
        old file on the rerun and reappear as open — the case would never
        converge.
        """
        from .execution import DECISIONS_FILE
        for path in Path(run_root).rglob(DECISIONS_FILE):
            path.unlink()


#: How each provenance category reads on the case's own screens, which show the
#: sentence rather than the category.
_PROVENANCE_SENTENCE = {
    PROV_HUMAN: "an operator told Trakt",
    PROV_CLIENT: "the client told Trakt",
    PROV_ARTEFACT: "a file the client sent",
    PROV_AGENT: "proposed by the agent",
    PROV_APPROVED: "proposed by the agent and approved by an operator",
    PROV_INHERITED: "existing configuration",
}

_AGENT_FOR_STEP = {
    "onboard": "Onboarding Agent",
    "transform": "Transformation Agent",
    "validate": "Validation Agent",
    "stamp": "Provenance stamping",
}

def _decision_type_of(decision: Dict[str, Any]) -> str:
    """What kind of decision this is, from either shape it may be in.

    The adapter's raw row keeps it at the top level; the run's card keeps it
    under ``subject``. Both reach this module.
    """
    value = decision.get("decision_type")
    if value in (None, ""):
        value = (decision.get("subject") or {}).get("decision_type")
    return str(value or "")


def _raw_decisions(run_root: Path) -> Dict[str, Dict[str, Any]]:
    """The adapter's own pending-decision rows, keyed by decision id.

    The operator-facing :class:`DecisionRequired` contract deliberately carries
    no source-column detail; the raw artefact does, and a resolution has to be
    applied back to a column.
    """
    import yaml
    from .execution import DECISIONS_FILE
    out: Dict[str, Dict[str, Any]] = {}
    for path in sorted(Path(run_root).rglob(DECISIONS_FILE)):
        try:
            doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError:
            continue
        for row in doc.get("decisions") or []:
            if isinstance(row, dict) and row.get("decision_id"):
                out[str(row["decision_id"])] = row
    return out


def _decision_card(decision,
                   raw: Optional[Dict[str, Dict[str, Any]]] = None
                   ) -> Dict[str, Any]:
    """A :class:`DecisionRequired` as the decision card the tab renders."""
    d = decision.to_dict()
    recommendation = d.get("recommendation") or {}
    subject = dict(d.get("subject") or {})
    source = (raw or {}).get(str(subject.get("decision_id") or ""))
    if source:
        # Carried so the run's own copy of a decision still says WHAT KIND it
        # is. Dropping it made promotion silently no-op: every card reported an
        # empty decision type and every settled mapping failed the first test.
        subject.setdefault("decision_type", source.get("decision_type", ""))
        subject.setdefault("source_column", source.get("source_column", ""))
        subject.setdefault("source_columns", source.get("source_columns", []))
        # WHICH FILE'S COLUMN. Every file in a pack carries a loan identifier
        # and most carry a valuation date, so a card that says only "Pool"
        # names three different columns at once — and the table, which matched
        # rows to decisions on the name alone, marked all three.
        subject.setdefault("source_file", source.get("source_file", ""))
        subject.setdefault("target_field", source.get("target_field", ""))
        subject.setdefault("proposed_mapping",
                           source.get("proposed_mapping", ""))
    return {
        "decision_id": d["decision_id"],
        "kind": d["kind"],
        "title": d["title"],
        "question": d["question"],
        "blocking": bool(d.get("blocking")),
        "status": "open",
        "issue": d["title"],
        "evidence": d.get("evidence") or [],
        "recommendation": recommendation.get("value", ""),
        "recommendation_source": recommendation.get("source", ""),
        "confidence": recommendation.get("confidence"),
        "materiality": "BLOCKING" if d.get("blocking") else "REVIEW",
        "downstream_consequence": (
            "The practice run cannot continue until this is answered."
            if d.get("blocking") else
            "Recorded as an observation; it does not stop the run."),
        "options": d.get("options") or [],
        "subject": subject,
    }


def _planned_actions(state) -> List[Dict[str, Any]]:
    """What a live run would do next, from the conductor's own state."""
    actions: List[Dict[str, Any]] = []
    for portfolio in getattr(state, "portfolios", []) or []:
        for name, step in (portfolio.steps or {}).items():
            actions.append({"portfolio": portfolio.source_portfolio_id,
                            "step": name, "status": step.status,
                            "execution_status": "synthetic_only"})
    for name in ("assemble", "route", "project"):
        step = getattr(state, name, None)
        if step is not None and step.status != "pending":
            actions.append({"portfolio": "*", "step": name,
                            "status": step.status,
                            "execution_status": "synthetic_only"})
    return actions


def _missing_role_messages(readiness: RoleReadiness) -> List[str]:
    vocab = artefact_vocabulary()
    out = [f"Trakt still needs the {vocab.label(role)}."
           for role in readiness.missing]
    out += [item["question"] for item in readiness.low_confidence]
    return out


def _first_non_blocking(run: SyntheticRun) -> str:
    for decision in run.open_decisions:
        if not decision.get("blocking") and decision.get("status") == "open":
            return str(decision["decision_id"])
    return ""


def _first_open_request(case: OnboardingCase) -> str:
    outstanding = case.outstanding_requests
    if not outstanding:
        raise OpsError("OCC_AGENT_REQUEST_NOT_FOUND",
                       "There is no open information request to record a "
                       "response against.", http_status=409)
    return outstanding[0].request_id


def _reached_states(run: SyntheticRun) -> set:
    """States this run has actually been in, from its own evidence.

    Derived from recorded evidence rather than from position in the table, so a
    run that skipped nothing still shows exactly what happened.
    """
    reached = {_states.AWAITING_ONBOARDING}
    if run.received_artefacts:
        reached.add(_states.READY_TO_RUN)
    if run.stage_outcomes:
        reached |= {_states.READY_TO_RUN,
                    _states.SYNTHETIC_ONBOARDING_RUNNING}
    if run.open_decisions:
        reached.add(_states.EXCEPTIONS_REQUIRE_INPUT)
    if run.stage_outcomes.get("assemble") == STAGE_DETERMINISTIC_COMPLETED:
        reached.add(_states.SYNTHETIC_ONBOARDING_PASSED)
    if run.orchestration_plan:
        reached |= {_states.ORCHESTRATION_PLAN_GENERATED,
                    _states.EXECUTION_APPROVAL_REQUIRED}
    if run.readiness_status == _states.READY_FOR_EXECUTION:
        reached.add(_states.READY_FOR_EXECUTION)
    reached.add(run.state)
    return reached


def _occ_links(case: OnboardingCase, run: SyntheticRun) -> List[Dict[str, str]]:
    """Deep links into the EXISTING OCC views, rather than reproducing them."""
    links: List[Dict[str, str]] = []
    # THE ONBOARDING CASE, only once there is one to open.
    #
    # This link was offered on every case and was broken on all of them, twice
    # over. An Agent case lives in the synthetic container and reaches the
    # governed one at activation, through `promotion.promote`, so the governed
    # wizard 404s on anything earlier. And the path was `/onboarding/{id}`,
    # which is not a route at all — the wizard is `/onboarding/cases/{id}`,
    # which is what Client Onboarding's own home links to.
    #
    # A link an operator cannot follow is worse than no link: it reads as the
    # place the real work lives, so its failure reads as the case being lost.
    if case.status == ACTIVATED:
        links.append(
            {"label": "Client onboarding",
             "to": f"/onboarding/cases/{case.case_id}",
             "why": "The activated onboarding case, in the screens an operator "
                    "normally works it in."})
    links += [
        {"label": "Platform configuration", "to": "/admin/config",
         "why": "The asset, regime and system packages this case resolved "
                "against."},
        {"label": "Rules", "to": "/rules",
         "why": "The approved mapping and alias rules the platform applies."},
    ]
    if run.blocking_decisions():
        links.append({"label": "Review", "to": "/reviews",
                      "why": "How the same decisions are answered on live "
                             "deliveries."})
    if run.stage_outcomes:
        links.append({"label": "Workflows", "to": "/workflows",
                      "why": "Where a live run of this package would appear."})
    return links
