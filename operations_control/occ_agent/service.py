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
    APPROVED,
    NO_ACTIVE_CONFIGURATION,
    STATUS_LABELS,
    TERMINAL,
    OnboardingCase,
)
from ..onboarding.service import STEP_LABELS, STEPS, OnboardingService
from . import adapters as _adapters
from . import classification as _classification
from . import client_form as _client_form
from . import communication as _comms
from . import derive as _derive
from . import pack as _pack
from . import planning as _planning
from . import readiness as _readiness
from . import review as _review
from . import states as _states
from .adapters import (
    ActivationPreconditions,
    ExecutionAdapter,
    LiveExecutionAdapter,
    SyntheticExecutionAdapter,
)
from .artefacts import ArtefactService, RoleReadiness, sample_manifest
from .derive import ExecutionFacts
from .execution import SyntheticOnboardingAdapters, run_synthetic_orchestration
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
                 engine: Optional[Any] = None):
        self.store = store or SyntheticRunStore(storage, container=container,
                                                sandbox=sandbox)
        # The onboarding service, pinned to the synthetic container. Everything
        # it writes — cases, versions, artefacts — lands there and nowhere near
        # the live operations container.
        self.onboarding = onboarding or OnboardingService(
            synthetic_ops_store(storage, self.store.container))
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
        """
        if _adapters.live_enabled() and engine is not None:
            return LiveExecutionAdapter(self.onboarding, engine)
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
                    fixture_id: str = "") -> AgentCase:
        """Open a practice case.

        The onboarding case is opened by Client Onboarding itself — same
        reference series, same blank start, same event history — and the run
        record is created beside it.
        """
        validate_segment(tenant, "tenant")
        case = self.onboarding.start_new_client(by=initiating_user)
        run = SyntheticRun(case_ref=case.case_id, tenant=tenant,
                           initiating_user=initiating_user,
                           fixture_id=fixture_id)
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
            confidence=interpretation.confidence)
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

        if agent_case.case.status not in (APPROVED,) + TERMINAL:
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
            regime=facts.regime,
            approved_mappings=self._approved_mappings(run),
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

        # Resolved decisions are kept (they are the record of what the human
        # settled); a decision the rerun raises again replaces its open twin
        # rather than being appended beside it.
        decisions = self._decisions_from_run(run, facts, run_root)
        settled = {d["decision_id"]: d for d in run.open_decisions
                   if d.get("status") != "open"}
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

        result = self.adapter.activate(pre=pre, intent=intent, actor=actor,
                                       payloads=self._payloads(run))
        run.activation_result = result.to_dict()
        if result.ok:
            self._move(run, _states.INGESTION_STARTED)
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
        return _adapters.build_intent(
            agent_case.case, facts, reporting_period=run.reporting_period,
            files=files,
            configuration_artefacts=[str(a.get("path") or a.get("name") or a)
                                     for a in (preview.get("artefacts") or [])])

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
            return self.cancel(agent_case, actor=actor)
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
        # question about the case as a whole falls through to readiness.
        if "client" in lower and any(w in lower for w in
                                     ("ask", "need", "send", "outstanding",
                                      "waiting", "chase")):
            checklist = self.onboarding.client_checklist(agent_case.case)
            if not checklist:
                return "There is nothing outstanding from the client."
            return "The client still has to tell us:\n" + "\n".join(
                f"- {row['label']}" for row in checklist)
        if any(w in lower for w in ("what is left", "what remains", "still",
                                    "outstanding", "before readiness",
                                    "what's left")):
            verdict = self._verdict(agent_case)
            if verdict.ready:
                return ("Every readiness criterion is satisfied. Approve "
                        "readiness to reach READY_FOR_EXECUTION.")
            return "Still outstanding:\n" + "\n".join(
                f"- {c.label}: {c.remedy or c.detail}"
                for c in verdict.outstanding)
        if "stage" in lower or "where" in lower or "status" in lower:
            return self.status_sentence(agent_case)
        spec = _states.spec(run.state)
        if "next" in lower or "can i" in lower or "what should" in lower:
            return ("From here you can: " + ", ".join(
                a.replace("_", " ") for a in spec.allowed_human_actions)
                + ".") if spec.allowed_human_actions else \
                "This practice case is finished; there is nothing further to do."
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
        """source column -> canonical field, from resolved mapping decisions.

        An empty target means "do not use this column", which is how the losing
        side of an ambiguity is recorded: both columns are answered, so the next
        run has nothing left to ask about.
        """
        out: Dict[str, str] = {}
        for decision in run.open_decisions:
            if decision.get("status") != "approved":
                continue
            subject = decision.get("subject") or {}
            target = str(subject.get("target_field") or "")
            columns = [str(c) for c in (subject.get("source_columns") or [])]
            primary = str(subject.get("source_column") or "")
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
            if chosen:
                out[chosen] = target_field
            for column in columns:
                if column != chosen:
                    out[column] = ""       # the losing candidate is not used
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
        subject.setdefault("source_column", source.get("source_column", ""))
        subject.setdefault("source_columns", source.get("source_columns", []))
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
    links = [
        {"label": "Client onboarding", "to": f"/onboarding/{case.case_id}",
         "why": "The onboarding case itself, in the screens an operator "
                "normally works it in."},
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
