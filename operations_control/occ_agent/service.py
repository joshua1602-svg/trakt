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
from . import derive as _derive
from . import readiness as _readiness
from . import states as _states
from .artefacts import ArtefactService, RoleReadiness, sample_manifest
from .derive import ExecutionFacts
from .execution import SyntheticOnboardingAdapters, run_synthetic_orchestration
from .input_roles import artefact_vocabulary
from .interpretation import (
    DeterministicInterpreter,
    Interpretation,
    InterpretationError,
    Interpreter,
    ProposedChange,
)
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
                 onboarding: Optional[OnboardingService] = None):
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
    def answer_from_instruction(self, agent_case: AgentCase, *,
                                instruction: str, actor: str) -> AgentCase:
        """Turn one instruction into answers on the onboarding case."""
        interpretation = self.interpreter.interpret_instruction(instruction)
        interpretation.validate(self.onboarding.catalogue)
        agent_case.run.messages.append(
            Message(role="operator", text=instruction[:4000]).to_dict())
        return self.apply_interpretation(agent_case,
                                         interpretation=interpretation,
                                         actor=actor)

    def apply_interpretation(self, agent_case: AgentCase, *,
                             interpretation: Interpretation,
                             actor: str) -> AgentCase:
        """Write a CONFIRMED interpretation onto the case, step by step."""
        interpretation.validate(self.onboarding.catalogue)
        run = agent_case.run
        case = agent_case.case
        written: List[str] = []
        for step in STEPS:
            payload = (interpretation.steps or {}).get(step)
            if not payload:
                continue
            case = self.onboarding.save_step(
                case_id=case.case_id, step=step,
                payload=self._merge_step(case, step, payload), by=actor)
            written.append(step)
        if interpretation.cadence:
            case = self._apply_cadence(case, interpretation.cadence, actor)
            written.append("sources")
        if interpretation.reporting_period:
            run.reporting_period = interpretation.reporting_period

        agent_case.case = case
        run.facts = self.facts(agent_case).to_dict()
        self.store.save(run)
        self._audit(run, "onboarding_answered", actor_type=ACTOR_AGENT,
                    actor=actor, classification=EXEC_MODEL_PROPOSED,
                    decision_basis="structured answers read from the "
                                   "instruction and written through Client "
                                   "Onboarding",
                    detail={"steps": written,
                            "reporting_period": run.reporting_period})
        run.messages.append(Message(
            role="agent", text=self.describe_case(agent_case)).to_dict())
        self.store.save(run)
        return agent_case

    def _merge_step(self, case: OnboardingCase, step: str,
                    payload: Dict[str, Any]) -> Dict[str, Any]:
        """Merge a proposed answer into what the case already holds.

        A repeatable section is replaced wholesale by ``save_step``, so a second
        instruction that mentions the portfolio must not silently drop the
        answers the first one gave. Merging by identity keeps them.
        """
        section = self.onboarding.catalogue.section(step)
        if section is None or not section.repeatable:
            return payload
        incoming = list(payload.get(section.key) or payload.get("items") or [])
        existing = list(case.items(step))
        if not existing:
            return {section.key: incoming}
        key = _identity_field(step)
        merged = [dict(item) for item in existing]
        for candidate in incoming:
            match = next(
                (m for m in merged
                 if key and candidate.get(key)
                 and str(m.get(key) or "") == str(candidate.get(key) or "")),
                None)
            if match is None and len(incoming) == 1 and len(merged) == 1:
                match = merged[0]     # one book named two ways is one book
            if match is None:
                merged.append(dict(candidate))
            else:
                match.update({k: v for k, v in candidate.items() if v not in
                              (None, "", [])})
        return {section.key: merged}

    def _apply_cadence(self, case: OnboardingCase, cadence: str,
                       actor: str) -> OnboardingCase:
        """Set the expected cadence on every delivery Trakt has derived."""
        sources = [dict(s) for s in case.items("sources")]
        if not sources:
            return case
        for source in sources:
            source["cadence"] = cadence
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
        if run.state == _states.AWAITING_ONBOARDING and \
                _states.is_transition_allowed(run.state, _states.READY_TO_RUN):
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

        if change.requires_confirmation and not confirm:
            proposal = {"proposal_id": new_id("prop"), "action": change.action,
                        "payload": change.payload, "summary": change.summary,
                        "basis": change.basis, "material": change.material,
                        "confidence": change.confidence}
            run.messages.append(Message(
                role="agent",
                text=f"Proposed: {proposal['summary']} Confirm to apply.",
                refs=[proposal["proposal_id"]]).to_dict())
            self.store.save(run)
            self._audit(run, "change_proposed", actor_type=ACTOR_AGENT,
                        actor=actor, classification=EXEC_MODEL_PROPOSED,
                        decision_basis=change.basis,
                        detail={"action": change.action})
            return TurnResult(case=agent_case, reply=proposal["summary"],
                              proposal=proposal)

        agent_case = self._apply(agent_case, change, actor=actor)
        reply = self.status_sentence(agent_case)
        agent_case.run.messages.append(
            Message(role="agent", text=reply).to_dict())
        self.store.save(agent_case.run)
        return TurnResult(case=agent_case, reply=reply, applied=True,
                          decisions=agent_case.run.open_decisions)

    def _apply(self, agent_case: AgentCase, change: ProposedChange, *,
               actor: str) -> AgentCase:
        action = change.action
        payload = change.payload
        if action == _states.ACTION_ANSWER:
            raw = payload.get("interpretation") or {}
            interpretation = Interpretation(
                **{k: v for k, v in raw.items()
                   if k in Interpretation.__dataclass_fields__})
            return self.apply_interpretation(
                agent_case, interpretation=interpretation, actor=actor)
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


_AGENT_FOR_STEP = {
    "onboard": "Onboarding Agent",
    "transform": "Transformation Agent",
    "validate": "Validation Agent",
    "stamp": "Provenance stamping",
}

#: Which field identifies an item within a repeatable catalogue section, so a
#: later instruction updates a book rather than adding a second one.
_IDENTITY_FIELDS = {"portfolios": "portfolio_id", "entities": "legal_name",
                    "sources": "source_key"}


def _identity_field(step: str) -> str:
    return _IDENTITY_FIELDS.get(step, "")


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
