"""operations_control.occ_agent.service — the OCC Agent's typed tool surface.

This is the bounded set of operations the agent can perform. Each method is one
of the tools named in §15; there is no general-purpose "do what the text says"
entry point, and the interpreter cannot reach the store, the filesystem or the
pipeline except through these methods.

The order of every state-changing method is the same, and it matters:

1. check the current state permits the action (:mod:`.states`);
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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from apps.blob_trigger_app.storage import Storage

from ..contracts import new_id, now_iso
from ..engine import OpsError
from . import pack as _pack
from . import readiness as _readiness
from . import states as _states
from .artefacts import ArtefactService, RoleReadiness
from .case import (
    ACTOR_AGENT,
    ACTOR_HUMAN,
    ACTOR_SYSTEM,
    Approval,
    EXEC_BLOCKED,
    EXEC_DETERMINISTIC,
    EXEC_HUMAN_CONFIRMED,
    EXEC_MODEL_PROPOSED,
    EXEC_SYNTHETICALLY_EXECUTED,
    ExtractedRequirements,
    Message,
    ProposedValue,
    STAGE_DETERMINISTIC_COMPLETED,
    STAGE_HARD_BLOCKED,
    SyntheticArtefact,
    SyntheticCase,
    new_case_id,
)
from .configuration import ConfigurationService
from .execution import (
    SyntheticOnboardingAdapters,
    run_synthetic_orchestration,
)
from .interpretation import (
    DeterministicInterpreter,
    Interpreter,
    InterpretationError,
    ProposedChange,
)
from .policy import (
    CAP_EXTERNAL_EMAIL,
    SyntheticPolicy,
    synthetic_policy,
    validate_segment,
)
from .store import SyntheticCaseStore
from .vocabulary import artefact_vocabulary, asset_vocabulary, product_vocabulary


class ActionNotAllowed(OpsError):
    """The action is not available from the case's current state."""

    def __init__(self, action: str, state: str):
        super().__init__(
            "OCC_AGENT_ACTION_NOT_ALLOWED",
            f"'{action.replace('_', ' ')}' is not something you can do while "
            f"this case is at {_states.spec_label(state)}.",
            http_status=409)


@dataclass
class TurnResult:
    """What one natural-language turn produced."""

    case: SyntheticCase
    reply: str = ""
    proposal: Optional[Dict[str, Any]] = None
    applied: bool = False
    decisions: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {"case": self.case.to_dict(), "reply": self.reply,
                "proposal": self.proposal, "applied": self.applied,
                "decisions": self.decisions}


class OccAgentService:
    """Every OCC Agent operation, in one injectable service."""

    def __init__(self, storage: Storage, *,
                 container: Optional[str] = None,
                 sandbox: Optional[Path] = None,
                 policy: Optional[SyntheticPolicy] = None,
                 interpreter: Optional[Interpreter] = None,
                 store: Optional[SyntheticCaseStore] = None):
        self.store = store or SyntheticCaseStore(storage, container=container,
                                                 sandbox=sandbox)
        self._pending_audit: List[Tuple[str, str, Dict[str, Any]]] = []
        self.policy = policy or synthetic_policy(audit_sink=self._audit_refusal)
        self.interpreter = interpreter or DeterministicInterpreter()
        self.artefacts = ArtefactService(self.store, self.policy)
        self.configuration = ConfigurationService(self.store, self.policy)

    # ------------------------------------------------------------------ #
    # Audit plumbing
    # ------------------------------------------------------------------ #
    def _audit_refusal(self, event: Dict[str, Any]) -> None:
        """Sink for policy refusals.

        A refusal can happen with no case in hand (a misconfigured call), so it
        is recorded against the case when there is one and dropped into the
        service log when there is not — never silently discarded.
        """
        case_id = str(event.get("case_id") or "")
        tenant = str(event.get("tenant") or "")
        if not case_id or not tenant:
            self._pending_audit.append(("", "", event))
            return
        self.store.append_audit(
            tenant, case_id, action=str(event.get("action") or "refused"),
            actor_type=ACTOR_SYSTEM,
            actor_identity=str(event.get("actor_identity") or ""),
            decision_basis=str(event.get("decision_basis") or ""),
            execution_classification=EXEC_BLOCKED,
            detail={"capability": event.get("capability"),
                    "detail": event.get("detail")})

    def _audit(self, case: SyntheticCase, action: str, *,
               actor_type: str = ACTOR_SYSTEM, actor: str = "",
               prior_state: str = "", decision_basis: str = "",
               classification: str = EXEC_DETERMINISTIC,
               input_reference: str = "", output_reference: str = "",
               detail: Optional[Dict[str, Any]] = None) -> None:
        self.store.append_audit(
            case.tenant, case.case_id, action=action, actor_type=actor_type,
            actor_identity=actor, prior_state=prior_state or case.state,
            resulting_state=case.state, decision_basis=decision_basis,
            execution_classification=classification,
            input_reference=input_reference, output_reference=output_reference,
            detail=detail or {})

    def _move(self, case: SyntheticCase, to_state: str) -> str:
        """Assert and apply a lifecycle transition. Returns the prior state."""
        prior = case.state
        _states.assert_transition(prior, to_state)
        case.state = to_state
        return prior

    @staticmethod
    def _require_action(case: SyntheticCase, action: str) -> None:
        if not _states.action_allowed(case.state, action):
            raise ActionNotAllowed(action, case.state)

    # ------------------------------------------------------------------ #
    # 1. create_case
    # ------------------------------------------------------------------ #
    def create_case(self, *, tenant: str, initiating_user: str,
                    instruction: str = "",
                    fixture_id: str = "") -> SyntheticCase:
        validate_segment(tenant, "tenant")
        case = SyntheticCase(case_id=new_case_id(), tenant=tenant,
                             initiating_user=initiating_user,
                             fixture_id=fixture_id)
        self.store.save(case)
        self._audit(case, "case_created", actor_type=ACTOR_HUMAN,
                    actor=initiating_user,
                    decision_basis="an operator opened a synthetic case")
        if instruction:
            case = self.extract_requirements(case, instruction=instruction,
                                             actor=initiating_user)
        return case

    def load_case(self, tenant: str, case_id: str) -> SyntheticCase:
        return self.store.load(tenant, case_id)

    def list_cases(self, tenant: str, *,
                   state: Optional[str] = None) -> List[Dict[str, Any]]:
        return self.store.list_cases(tenant, state=state)

    # ------------------------------------------------------------------ #
    # 2. extract_requirements
    # ------------------------------------------------------------------ #
    def extract_requirements(self, case: SyntheticCase, *, instruction: str,
                             actor: str) -> SyntheticCase:
        self._require_action(case, _states.ACTION_CORRECT_REQUIREMENTS)
        extracted = self.interpreter.interpret_instruction(instruction)
        extracted.validate()          # every interpreter's output, model or not
        case.extracted_requirements = extracted.to_dict()
        case.unresolved_questions = list(extracted.unresolved_questions)
        case.messages.append(Message(role="operator",
                                     text=instruction[:4000]).to_dict())
        prior = self._move(case, _states.REQUIREMENTS_EXTRACTED)
        self.store.save(case)
        self._audit(case, "requirements_extracted", actor_type=ACTOR_AGENT,
                    actor=actor, prior_state=prior,
                    classification=EXEC_MODEL_PROPOSED,
                    decision_basis="structured facts read from the instruction",
                    detail={"unresolved": len(case.unresolved_questions)})
        # Showing the interpretation IS the next step, so the case moves on to
        # the state that says a human must look at it.
        prior = self._move(case, _states.REQUIREMENTS_CONFIRMATION_REQUIRED)
        case.messages.append(Message(
            role="agent",
            text=self.describe_interpretation(case)).to_dict())
        self.store.save(case)
        self._audit(case, "interpretation_presented", actor_type=ACTOR_AGENT,
                    actor=actor, prior_state=prior,
                    classification=EXEC_MODEL_PROPOSED,
                    decision_basis="the interpretation needs confirmation")
        return case

    def describe_interpretation(self, case: SyntheticCase) -> str:
        """The proposed interpretation, in the operator's own vocabulary."""
        req = case.extracted
        assets = asset_vocabulary()
        products = product_vocabulary()
        roles = artefact_vocabulary()
        lines = [f"Client: {req.client_name or '—'}"]
        if req.proposed_client_id:
            lines.append(f"Proposed client identifier: {req.proposed_client_id}")
        if req.portfolio_name or req.proposed_portfolio_id:
            lines.append(f"Portfolio: {req.portfolio_name or '—'}"
                         + (f" ({req.proposed_portfolio_id})"
                            if req.proposed_portfolio_id else ""))
        lines.append(f"Asset type: {assets.label(req.asset_type) if req.asset_type else '—'}")
        if req.jurisdiction:
            lines.append(f"Jurisdiction: {req.jurisdiction}")
        if req.required_products:
            lines.append("Products:")
            for pid in req.required_products:
                product = products.by_id(pid)
                lines.append(f"- {product.label if product else pid}")
        if req.reporting_frequency:
            lines.append(f"Frequency: {req.reporting_frequency.title()}")
        if req.reporting_date:
            lines.append(f"First reporting date: {req.reporting_date}")
        if req.target_go_live_date:
            lines.append(f"Target go-live: {req.target_go_live_date}")
        if req.expected_artefacts:
            lines.append("Expected artefacts:")
            for role in req.expected_artefacts:
                lines.append(f"- {roles.label(role)}")
        if req.known_counterparties:
            lines.append("Counterparties:")
            for cp in req.known_counterparties:
                lines.append(f"- {cp}")
        if req.unresolved_questions:
            lines.append("Outstanding:")
            for question in req.unresolved_questions:
                lines.append(f"- {question}")
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    # 3. propose_requirement_change / 4. confirm_requirements
    # ------------------------------------------------------------------ #
    def propose_requirement_change(self, case: SyntheticCase, *,
                                   updates: Dict[str, Any]) -> Dict[str, Any]:
        """The structured change a correction would make. Applies nothing."""
        current = case.confirmed if case.confirmed_requirements else case.extracted
        before = current.to_dict()
        after = dict(before)
        after.update(updates)
        candidate = ExtractedRequirements.from_dict(after)
        candidate.validate()
        changed = [k for k in updates if before.get(k) != after.get(k)]
        return {
            "kind": "requirement_change",
            "updates": {k: after[k] for k in changed},
            "before": {k: before.get(k) for k in changed},
            "affected_pack_sections": _pack.affected_sections(changed),
            "material": True,
            "summary": "Update " + ", ".join(
                k.replace("_", " ") for k in sorted(changed)) + "."
            if changed else "No change.",
        }

    def apply_requirement_change(self, case: SyntheticCase, *,
                                 updates: Dict[str, Any],
                                 actor: str) -> SyntheticCase:
        """Apply a CONFIRMED correction to the structured facts."""
        self._require_action(case, _states.ACTION_CORRECT_REQUIREMENTS)
        target = "confirmed_requirements" if case.confirmed_requirements \
            else "extracted_requirements"
        current = ExtractedRequirements.from_dict(getattr(case, target))
        before = current.to_dict()
        merged = dict(before)
        merged.update(updates)
        candidate = ExtractedRequirements.from_dict(merged)
        candidate.validate()
        # Identity is never changed as a side effect of a fact correction.
        if candidate.proposed_client_id and case.client_id and \
                candidate.proposed_client_id != case.client_id:
            case.rebind_identity(client_id=candidate.proposed_client_id)
        if candidate.proposed_portfolio_id and case.portfolio_id and \
                candidate.proposed_portfolio_id != case.portfolio_id:
            case.rebind_identity(portfolio_id=candidate.proposed_portfolio_id)
        candidate.unresolved_questions = DeterministicInterpreter.open_questions(
            candidate)
        setattr(case, target, candidate.to_dict())
        case.unresolved_questions = list(candidate.unresolved_questions)

        changed = sorted(k for k in updates if before.get(k) != merged.get(k))
        prior = case.state
        # A correction to CONFIRMED facts reopens confirmation: the human must
        # see the amended interpretation before it is authoritative again.
        if target == "confirmed_requirements":
            case.confirmed_requirements = candidate.to_dict()
            if _states.is_transition_allowed(case.state,
                                             _states.REQUIREMENTS_EXTRACTED):
                self._move(case, _states.REQUIREMENTS_EXTRACTED)
                self._move(case, _states.REQUIREMENTS_CONFIRMATION_REQUIRED)
        # Regenerate the affected pack sections rather than the whole pack.
        if case.onboarding_pack and changed:
            pack = _pack.OnboardingPack(**{
                k: v for k, v in case.onboarding_pack.items()
                if k in _pack.OnboardingPack.__dataclass_fields__})
            regenerated = _pack.regenerate_sections(
                pack, candidate, changed, client_id=case.client_id,
                portfolio_id=case.portfolio_id)
            case.onboarding_pack = regenerated.to_dict()
        self.store.save(case)
        self._audit(case, "requirements_corrected", actor_type=ACTOR_HUMAN,
                    actor=actor, prior_state=prior,
                    classification=EXEC_HUMAN_CONFIRMED,
                    decision_basis="an operator corrected the facts",
                    detail={"changed": changed})
        return case

    def confirm_requirements(self, case: SyntheticCase, *,
                             actor: str) -> SyntheticCase:
        self._require_action(case, _states.ACTION_CONFIRM_REQUIREMENTS)
        extracted = case.extracted
        extracted.validate()
        missing = extracted.missing_mandatory()
        if missing:
            return self._block(
                case, [f"Trakt still needs the {m}." for m in missing],
                actor=actor, reason="mandatory facts missing")
        case.confirmed_requirements = extracted.to_dict()
        case.client_name = extracted.client_name
        case.portfolio_name = extracted.portfolio_name
        case.asset_type = extracted.asset_type
        case.rebind_identity(client_id=extracted.proposed_client_id,
                             portfolio_id=extracted.proposed_portfolio_id)
        case.approval_history.append(Approval(
            approval_id=new_id("appr"), subject="requirements",
            decision="approved", actor=actor,
            reference=str(len(extracted.to_dict()))).to_dict())
        prior = self._move(case, _states.REQUIREMENTS_CONFIRMED)
        self.store.save(case)
        self._audit(case, "requirements_confirmed", actor_type=ACTOR_HUMAN,
                    actor=actor, prior_state=prior,
                    classification=EXEC_HUMAN_CONFIRMED,
                    decision_basis="the operator confirmed the interpretation")
        return case

    # ------------------------------------------------------------------ #
    # 5. generate_onboarding_pack
    # ------------------------------------------------------------------ #
    def generate_onboarding_pack(self, case: SyntheticCase, *,
                                 actor: str) -> SyntheticCase:
        self._require_action(case, _states.ACTION_GENERATE_PACK)
        pack = _pack.build_pack(case.confirmed, client_id=case.client_id,
                                portfolio_id=case.portfolio_id)
        case.onboarding_pack = pack.to_dict()
        case.required_artefacts = [
            {"role": role,
             "label": artefact_vocabulary().label(role),
             "required": role in pack.required_roles}
            for role in list(pack.required_roles) + list(pack.optional_roles)]
        prior = case.state
        if case.state != _states.ONBOARDING_PACK_GENERATED:
            prior = self._move(case, _states.ONBOARDING_PACK_GENERATED)
        self._move(case, _states.ONBOARDING_PACK_APPROVAL_REQUIRED)
        self.store.save(case)
        self._audit(case, "onboarding_pack_generated", actor_type=ACTOR_AGENT,
                    actor=actor, prior_state=prior,
                    classification=EXEC_DETERMINISTIC,
                    output_reference=pack.content_hash,
                    decision_basis="built from the onboarding configuration "
                                   "framework")
        return case

    def approve_onboarding_pack(self, case: SyntheticCase, *,
                                actor: str) -> SyntheticCase:
        self._require_action(case, _states.ACTION_APPROVE_PACK)
        if not case.onboarding_pack:
            raise ActionNotAllowed(_states.ACTION_APPROVE_PACK, case.state)
        case.approval_history.append(Approval(
            approval_id=new_id("appr"), subject="onboarding_pack",
            decision="approved", actor=actor,
            reference=str(case.onboarding_pack.get("content_hash"))).to_dict())
        prior = self._move(case, _states.ONBOARDING_PACK_ISSUED_SYNTHETICALLY)
        case.pack_issued_synthetically_at = now_iso()
        self._move(case, _states.AWAITING_CLIENT_INPUT)
        self.store.save(case)
        self._audit(case, "onboarding_pack_issued_synthetically",
                    actor_type=ACTOR_HUMAN, actor=actor, prior_state=prior,
                    classification=EXEC_HUMAN_CONFIRMED,
                    decision_basis="recorded as issued; no email was sent",
                    output_reference=str(
                        case.onboarding_pack.get("content_hash")))
        return case

    def send_pack_by_email(self, case: SyntheticCase, *, actor: str) -> None:
        """The external-email seam. Always refused in synthetic mode."""
        self.policy.require(CAP_EXTERNAL_EMAIL, detail="onboarding pack",
                            case_id=case.case_id, tenant=case.tenant,
                            actor=actor)

    # ------------------------------------------------------------------ #
    # 6. register_synthetic_artefact / 7. classify_artefact
    # ------------------------------------------------------------------ #
    def register_synthetic_artefact(self, case: SyntheticCase, *,
                                    filename: str, data: bytes, actor: str,
                                    fixture_id: str = "",
                                    declared_type: str = "") -> SyntheticCase:
        self._require_action(case, _states.ACTION_REGISTER_ARTEFACT)
        artefact = self.artefacts.register(
            case, filename=filename, data=data, provided_by=actor,
            fixture_id=fixture_id, declared_type=declared_type)
        case.received_artefacts.append(artefact.to_dict())
        prior = case.state
        if case.state != _states.ARTEFACTS_RECEIVED:
            self._move(case, _states.ARTEFACTS_RECEIVED)
        self.store.save(case)
        self._audit(case, "synthetic_artefact_registered",
                    actor_type=ACTOR_HUMAN, actor=actor, prior_state=prior,
                    classification=EXEC_SYNTHETICALLY_EXECUTED,
                    input_reference=artefact.source_file,
                    output_reference=artefact.synthetic_location,
                    decision_basis="stored in the synthetic case sandbox; the "
                                   "intended live location was derived but not "
                                   "written",
                    detail={"intended_live_uri": artefact.intended_live_uri,
                            "execution_status": "simulated_only",
                            "sha256": artefact.sha256})
        return case

    def classify_artefacts(self, case: SyntheticCase, *,
                           actor: str) -> SyntheticCase:
        self._require_action(case, _states.ACTION_CLASSIFY_ARTEFACTS)
        artefacts = [SyntheticArtefact.from_dict(a)
                     for a in case.received_artefacts]
        classified, findings = self.artefacts.classify(case, artefacts)
        case.received_artefacts = [a.to_dict() for a in classified]
        for finding in findings:
            if finding not in case.observations:
                case.observations.append(finding)
        outcome = product_vocabulary().outcome_for(
            case.confirmed.required_products)
        readiness = self.artefacts.readiness(case, outcome)
        self._record_control(case, "artefact_readiness", readiness.to_dict())
        prior = self._move(case, _states.ARTEFACTS_CLASSIFIED)
        self.store.save(case)
        self._audit(case, "artefacts_classified", actor_type=ACTOR_AGENT,
                    actor=actor, prior_state=prior,
                    classification=EXEC_DETERMINISTIC,
                    decision_basis="apps.blob_trigger_app.file_roles",
                    detail=readiness.to_dict())
        if not readiness.ready:
            return self._block(case, _missing_role_messages(readiness),
                               actor=actor,
                               reason="required input roles not satisfied")
        return case

    # ------------------------------------------------------------------ #
    # 8. draft_client_config / 9. validate_client_config
    # ------------------------------------------------------------------ #
    def draft_client_config(self, case: SyntheticCase, *,
                            actor: str) -> SyntheticCase:
        self._require_action(case, _states.ACTION_DRAFT_CONFIG)
        draft = self.configuration.draft(case)
        problems = self.configuration.validate(draft)
        case.proposed_configuration = draft.to_dict()
        case.configuration_provenance = draft.provenance
        self._record_control(case, "configuration", {
            "status": draft.status, "blockers": draft.blockers,
            "warnings": draft.warnings, "schema_problems": problems})
        for warning in draft.warnings:
            if warning not in case.observations:
                case.observations.append(warning)
        prior = case.state
        if case.state != _states.CONFIG_DRAFTED:
            self._move(case, _states.CONFIG_DRAFTED)
        self.store.save(case)
        self._audit(case, "configuration_drafted", actor_type=ACTOR_AGENT,
                    actor=actor, prior_state=prior,
                    classification=EXEC_DETERMINISTIC,
                    output_reference=draft.content_hash,
                    decision_basis="resolved through the existing "
                                   "configuration hierarchy",
                    detail={"status": draft.status,
                            "blockers": len(draft.blockers)})
        if draft.blockers or problems:
            return self._block(case, list(draft.blockers) + problems,
                               actor=actor,
                               reason="configuration did not validate")
        self._move(case, _states.CONFIG_APPROVAL_REQUIRED)
        self.store.save(case)
        return case

    def correct_client_config(self, case: SyntheticCase, *,
                              updates: Dict[str, Any],
                              actor: str) -> SyntheticCase:
        """Supply or correct a configuration value, then redraft.

        This is how a product-configuration gap is closed: the operator provides
        the value the client answered with, and the configuration is resolved
        again through the real hierarchy — so the resolver, not the operator,
        decides whether the gap is actually closed.
        """
        self._require_action(case, _states.ACTION_CORRECT_CONFIG)
        from .configuration import configuration_key_vocabulary
        allowed = set(configuration_key_vocabulary().keys())
        unknown = [key for key in updates if key not in allowed]
        if unknown:
            raise OpsError(
                "OCC_AGENT_UNKNOWN_CONFIG_KEY",
                "Trakt does not ask for " + ", ".join(unknown) + " here.",
                http_status=400)
        case.client_answers.update({k: str(v) for k, v in updates.items()})
        case.blockers = []
        self.store.save(case)
        self._audit(case, "configuration_answer_supplied",
                    actor_type=ACTOR_HUMAN, actor=actor,
                    classification=EXEC_HUMAN_CONFIRMED,
                    decision_basis="an operator supplied a configuration value",
                    detail={"keys": sorted(updates)})
        # A blocked case returns to the drafting stage so the resolver runs
        # again; a case still at drafting simply redrafts.
        if case.state == _states.BLOCKED and _states.is_transition_allowed(
                case.state, _states.CONFIG_DRAFTED):
            self._move(case, _states.CONFIG_DRAFTED)
        return self.draft_client_config(case, actor=actor)

    def approve_client_config(self, case: SyntheticCase, *,
                              actor: str) -> SyntheticCase:
        self._require_action(case, _states.ACTION_APPROVE_CONFIG)
        draft = case.proposed_configuration or {}
        if not draft:
            raise ActionNotAllowed(_states.ACTION_APPROVE_CONFIG, case.state)
        # Approving the configuration confirms the material values it proposed:
        # the operator is shown each one, so approval IS their confirmation.
        confirmed: List[Dict[str, Any]] = []
        for raw in case.configuration_provenance:
            value = ProposedValue.from_dict(raw)
            if value.needs_confirmation():
                value.confirmed = True
                value.confirmed_by = actor
                value.confirmed_at = now_iso()
            confirmed.append(value.to_dict())
        case.configuration_provenance = confirmed
        case.confirmed_configuration = draft.get("candidate") or {}
        case.approval_history.append(Approval(
            approval_id=new_id("appr"), subject="client_configuration",
            decision="approved", actor=actor,
            reference=str(draft.get("content_hash"))).to_dict())
        prior = self._move(case, _states.CONFIG_APPROVED)
        self.store.save(case)
        self._audit(case, "configuration_approved", actor_type=ACTOR_HUMAN,
                    actor=actor, prior_state=prior,
                    classification=EXEC_HUMAN_CONFIRMED,
                    output_reference=str(draft.get("content_hash")),
                    decision_basis="the operator approved the proposed "
                                   "configuration and its material values")
        return case

    # ------------------------------------------------------------------ #
    # 10-14. run_synthetic_onboarding (drives the real conductor)
    # ------------------------------------------------------------------ #
    def run_synthetic_onboarding(self, case: SyntheticCase, *,
                                 actor: str) -> SyntheticCase:
        self._require_action(case, _states.ACTION_RUN_ONBOARDING)
        prior = self._move(case, _states.SYNTHETIC_ONBOARDING_RUNNING)
        self.store.save(case)
        self._audit(case, "synthetic_onboarding_started", actor_type=ACTOR_AGENT,
                    actor=actor, prior_state=prior,
                    classification=EXEC_SYNTHETICALLY_EXECUTED,
                    decision_basis="the existing orchestration conductor was "
                                   "run over the synthetic adapter")

        paths = self._artefact_paths(case)
        products = product_vocabulary()
        regime = next((p.regime_id for p in
                       (products.by_id(x) for x in
                        case.confirmed.required_products)
                       if p is not None and p.regime_id), "")
        adapters = SyntheticOnboardingAdapters(
            artefact_paths=paths, policy=self.policy,
            sandbox=self.store.case_dir(case.tenant, case.case_id),
            asset_type=case.asset_type,
            regime=regime,
            approved_mappings=self._approved_mappings(case),
            case_id=case.case_id, tenant=case.tenant)
        run_root = self.store.run_dir(case.tenant, case.case_id)
        self._purge_stale_decisions(run_root)
        state = run_synthetic_orchestration(
            adapters, client_id=case.client_id,
            portfolio_id=case.portfolio_id, out_root=run_root,
            created_at=case.created_at,
            target=("regime" if regime else "mi"), regime=regime or None,
            run_id=f"syn_{case.case_id.lower().replace('-', '_')}")

        for record in adapters.records:
            case.stage_outcomes[record.stage] = record.outcome
            self._record_control(case, "stage", record.to_dict())
        if adapters.validation_report:
            self._record_control(case, "validation",
                                 {"findings": adapters.validation_report})
        case.mapping_decisions = adapters.mapping_report

        # Resolved decisions are kept (they are the record of what the human
        # settled); a decision the rerun raises again replaces its open twin
        # rather than being appended beside it.
        decisions = self._decisions_from_run(case, run_root)
        settled = {d["decision_id"]: d for d in case.open_decisions
                   if d.get("status") != "open"}
        for decision in decisions:
            settled.setdefault(decision["decision_id"], decision)
        case.open_decisions = list(settled.values())
        decisions = [d for d in case.open_decisions
                     if d.get("status", "open") == "open"]
        case.planned_pipeline_actions = _planned_actions(state)

        if decisions:
            self._move(case, _states.EXCEPTIONS_REQUIRE_INPUT)
            self.store.save(case)
            self._audit(case, "synthetic_onboarding_needs_input",
                        actor_type=ACTOR_AGENT, actor=actor,
                        classification=EXEC_SYNTHETICALLY_EXECUTED,
                        decision_basis="a governed control needs a human",
                        detail={"open_decisions": len(decisions)})
            return case

        blocked = [r for r in adapters.records
                   if r.outcome == STAGE_HARD_BLOCKED]
        if blocked or state.status not in ("done",):
            blockers = [b for r in blocked for b in r.blockers] or \
                list(state.blockers)
            self._move(case, _states.EXCEPTIONS_REQUIRE_INPUT)
            self.store.save(case)
            return self._block(case, blockers, actor=actor,
                               reason="a deterministic control blocked the run")

        self._move(case, _states.SYNTHETIC_ONBOARDING_PASSED)
        self.store.save(case)
        self._audit(case, "synthetic_onboarding_passed", actor_type=ACTOR_AGENT,
                    actor=actor, classification=EXEC_SYNTHETICALLY_EXECUTED,
                    decision_basis="every deterministic control passed",
                    detail={"stages": dict(case.stage_outcomes)})
        return case

    # ------------------------------------------------------------------ #
    # 15. record_human_decision (mappings, exceptions)
    # ------------------------------------------------------------------ #
    def resolve_decision(self, case: SyntheticCase, *, decision_id: str,
                         action: str, value: str = "", reason: str = "",
                         actor: str) -> SyntheticCase:
        """Record a human decision, then rerun only the affected controls."""
        self._require_action(case, _states.ACTION_RESOLVE_DECISION)
        target = next((d for d in case.open_decisions
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
        target["resolved_value"] = value or target.get("recommended_value", "")
        target["resolved_by"] = actor
        target["resolved_at"] = now_iso()
        target["reason"] = reason
        case.approval_history.append(Approval(
            approval_id=new_id("appr"), subject="decision",
            decision=("approved" if action != "reject" else "rejected"),
            actor=actor, reason=reason, reference=decision_id).to_dict())
        self.store.save(case)
        self._audit(case, "human_decision_recorded", actor_type=ACTOR_HUMAN,
                    actor=actor, classification=EXEC_HUMAN_CONFIRMED,
                    input_reference=decision_id,
                    decision_basis=reason or f"operator chose '{action}'",
                    detail={"resolved_value": target["resolved_value"]})

        # Rerun only what the decision affects: an unresolved mapping means the
        # run must repeat from onboarding; everything else is already recorded.
        if not case.blocking_decisions():
            if _states.is_transition_allowed(
                    case.state, _states.SYNTHETIC_ONBOARDING_RUNNING):
                return self.run_synthetic_onboarding(case, actor=actor)
        return case

    def acknowledge_exception(self, case: SyntheticCase, *, decision_id: str,
                              actor: str, reason: str = "") -> SyntheticCase:
        """Acknowledge a NON-blocking exception.

        A blocking exception is refused here: acknowledgement is not a
        resolution, and a deterministic blocker cannot be talked past.
        """
        self._require_action(case, _states.ACTION_ACKNOWLEDGE_EXCEPTION)
        target = next((d for d in case.open_decisions
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
        self.store.save(case)
        self._audit(case, "exception_acknowledged", actor_type=ACTOR_HUMAN,
                    actor=actor, classification=EXEC_HUMAN_CONFIRMED,
                    input_reference=decision_id, decision_basis=reason)
        return case

    # ------------------------------------------------------------------ #
    # 16-17. generate_orchestration_plan / validate_assembler_prerequisites
    # ------------------------------------------------------------------ #
    def generate_orchestration_plan(self, case: SyntheticCase, *,
                                    actor: str) -> SyntheticCase:
        self._require_action(case, _states.ACTION_GENERATE_PLAN)
        from engine.orchestrator_agent.orchestrator import (
            onboarding_mode_for_target,
            steps_for_target,
        )
        products = product_vocabulary()
        outcome = products.outcome_for(case.confirmed.required_products)
        regime = next((p.regime_id for p in
                       (products.by_id(x) for x in
                        case.confirmed.required_products)
                       if p is not None and p.regime_id), "")
        target = "regime" if regime else "mi"
        # The step sequence is the conductor's own, not a list written here.
        step_names = list(steps_for_target(target, full_pipeline=True))
        produces = {
            "onboard": ["18_central_lender_tape.csv",
                        "24_onboarding_handoff_manifest.json"],
            "transform": ["31_transformed_canonical_tape.csv",
                          "30_transformation_manifest.json"],
            "validate": ["40_validation_manifest.json"],
            "stamp": [f"{case.portfolio_id}_canonical_typed.csv"],
        }
        steps = [{"step": name, "agent": _AGENT_FOR_STEP.get(name, ""),
                  "produces": produces.get(name, []),
                  "observed_outcome": case.stage_outcomes.get(name, "")}
                 for name in step_names]
        steps.append({"step": "assemble", "agent": "Assembler Agent",
                      "produces": ["platform_canonical_typed.csv"],
                      "observed_outcome": case.stage_outcomes.get("assemble",
                                                                  "")})
        if target == "mi":
            steps.append({"step": "route", "agent": "MI route",
                          "produces": [],
                          "observed_outcome": case.stage_outcomes.get("route",
                                                                      "")})
        else:
            steps.append({"step": "project", "agent": "Regime projector",
                          "produces": [f"central_{regime}_projected.csv"],
                          "observed_outcome": case.stage_outcomes.get("project",
                                                                      "")})
        case.orchestration_plan = {
            "target": target, "outcome": outcome, "regime": regime,
            "onboarding_mode": onboarding_mode_for_target(target),
            "steps": steps,
            "valid": all(s["observed_outcome"] for s in steps),
            "source": "engine.orchestrator_agent.orchestrator",
            "execution_status": "not_executed",
        }
        case.assembler_plan = self._assembler_prerequisites(case)
        prior = self._move(case, _states.ORCHESTRATION_PLAN_GENERATED)
        self.store.save(case)
        self._audit(case, "orchestration_plan_generated", actor_type=ACTOR_AGENT,
                    actor=actor, prior_state=prior,
                    classification=EXEC_DETERMINISTIC,
                    decision_basis="sequence taken from the orchestration "
                                   "conductor; nothing was executed",
                    detail={"steps": [s["step"] for s in steps]})
        if not case.assembler_plan.get("satisfied"):
            return self._block(case, case.assembler_plan.get("problems", []),
                               actor=actor,
                               reason="assembler prerequisites not satisfied")
        self._move(case, _states.EXECUTION_APPROVAL_REQUIRED)
        self.store.save(case)
        return case

    def _assembler_prerequisites(self, case: SyntheticCase) -> Dict[str, Any]:
        """Check the Assembler Agent's real prerequisites against the run."""
        from engine.platform_assembler import LOAN_KEY_FIELDS
        problems: List[str] = []
        mapped = {m.get("canonical_field") for m in case.mapping_decisions
                  if m.get("canonical_field")}
        if not (set(LOAN_KEY_FIELDS) & mapped):
            problems.append(
                "The Assembler needs a loan identifier "
                f"({' or '.join(LOAN_KEY_FIELDS)}) and none was mapped.")
        if case.stage_outcomes.get("assemble") != STAGE_DETERMINISTIC_COMPLETED:
            problems.append("The synthetic run did not produce an assembled "
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

    # ------------------------------------------------------------------ #
    # 18-20. evaluate_readiness / approve / generate_readiness_package
    # ------------------------------------------------------------------ #
    def evaluate_readiness(self, case: SyntheticCase) -> Dict[str, Any]:
        return _readiness.evaluate(case, self.policy).to_dict()

    def approve_execution_readiness(self, case: SyntheticCase, *,
                                    actor: str) -> SyntheticCase:
        self._require_action(case, _states.ACTION_APPROVE_EXECUTION)
        case.approval_history.append(Approval(
            approval_id=new_id("appr"), subject="execution_readiness",
            decision="approved", actor=actor).to_dict())
        # The approval is one criterion, not the verdict: readiness is
        # re-derived AFTER it is recorded, and only a full pass moves the case.
        verdict = _readiness.evaluate(case, self.policy)
        case.readiness = verdict.to_dict()
        if not verdict.ready:
            case.readiness_status = "NOT_READY"
            self.store.save(case)
            self._audit(case, "readiness_refused", actor_type=ACTOR_SYSTEM,
                        actor=actor, classification=EXEC_DETERMINISTIC,
                        decision_basis="deterministic criteria are not all "
                                       "satisfied",
                        detail={"outstanding": [c.key
                                                for c in verdict.outstanding]})
            return self._block(
                case, [c.remedy or c.detail for c in verdict.outstanding],
                actor=actor, reason="readiness criteria not satisfied")
        prior = self._move(case, _states.READY_FOR_EXECUTION)
        case.readiness_status = _states.READY_FOR_EXECUTION
        package = _readiness.build_package(
            case, verdict, self.store.list_audit(case.tenant, case.case_id),
            self.policy)
        path = self.store.package_dir(case.tenant, case.case_id) \
            / "readiness_package.json"
        path.write_text(json.dumps(package, indent=2, default=str),
                        encoding="utf-8")
        case.readiness_package_ref = self.store.relative(
            case.tenant, case.case_id, path)
        self.store.save(case)
        self._audit(case, "ready_for_execution", actor_type=ACTOR_SYSTEM,
                    actor=actor, prior_state=prior,
                    classification=EXEC_DETERMINISTIC,
                    output_reference=case.readiness_package_ref,
                    decision_basis="every readiness criterion passed "
                                   "deterministically",
                    detail={"manifest_hash":
                            package["execution_manifest"]["content_hash"]})
        return case

    def readiness_package(self, case: SyntheticCase) -> Dict[str, Any]:
        verdict = _readiness.evaluate(case, self.policy)
        return _readiness.build_package(
            case, verdict, self.store.list_audit(case.tenant, case.case_id),
            self.policy)

    # ------------------------------------------------------------------ #
    # Returning to an earlier stage / cancelling
    # ------------------------------------------------------------------ #
    def return_to_stage(self, case: SyntheticCase, *, target_state: str,
                        actor: str, reason: str = "") -> SyntheticCase:
        self._require_action(case, _states.ACTION_RETURN_TO_STAGE)
        if target_state not in _states.RETURNABLE_STATES:
            raise _states.IllegalCaseTransition(case.state, target_state)
        prior = self._move(case, target_state)
        self.store.save(case)
        self._audit(case, "returned_to_earlier_stage", actor_type=ACTOR_HUMAN,
                    actor=actor, prior_state=prior,
                    classification=EXEC_HUMAN_CONFIRMED,
                    decision_basis=reason or "the operator reopened a stage")
        return case

    def cancel(self, case: SyntheticCase, *, actor: str,
               reason: str = "") -> SyntheticCase:
        self._require_action(case, _states.ACTION_CANCEL)
        prior = self._move(case, _states.CANCELLED)
        self.store.save(case)
        self._audit(case, "case_cancelled", actor_type=ACTOR_HUMAN, actor=actor,
                    prior_state=prior, classification=EXEC_HUMAN_CONFIRMED,
                    decision_basis=reason or "cancelled by the operator")
        return case

    # ------------------------------------------------------------------ #
    # The natural-language door
    # ------------------------------------------------------------------ #
    def instruct(self, case: SyntheticCase, *, text: str, actor: str,
                 confirm: bool = False) -> TurnResult:
        """One natural-language turn.

        A non-material action is applied straight away. A material one comes
        back as a proposal the human confirms — which is what keeps "natural
        language must not override a governed control" true in practice.
        """
        case.messages.append(Message(role="operator",
                                     text=text[:4000]).to_dict())
        try:
            change = self.interpreter.interpret_action(text, case)
        except InterpretationError:
            self.store.save(case)
            raise
        change.validate()

        if change.action == _states.ACTION_ASK:
            reply = self.answer(case, change.payload.get("question", ""))
            case.messages.append(Message(role="agent", text=reply).to_dict())
            self.store.save(case)
            return TurnResult(case=case, reply=reply)

        self._require_action(case, change.action)

        if change.requires_confirmation and not confirm:
            proposal = self._proposal_for(case, change)
            case.messages.append(Message(
                role="agent",
                text=f"Proposed: {proposal['summary']} Confirm to apply.",
                refs=[proposal.get("proposal_id", "")]).to_dict())
            self.store.save(case)
            self._audit(case, "change_proposed", actor_type=ACTOR_AGENT,
                        actor=actor, classification=EXEC_MODEL_PROPOSED,
                        decision_basis=change.basis,
                        detail={"action": change.action})
            return TurnResult(case=case, reply=proposal["summary"],
                              proposal=proposal)

        case = self._apply(case, change, actor=actor)
        reply = self.status_sentence(case)
        case.messages.append(Message(role="agent", text=reply).to_dict())
        self.store.save(case)
        return TurnResult(case=case, reply=reply, applied=True,
                          decisions=case.open_decisions)

    def _proposal_for(self, case: SyntheticCase,
                      change: ProposedChange) -> Dict[str, Any]:
        proposal = {"proposal_id": new_id("prop"), "action": change.action,
                    "payload": change.payload, "summary": change.summary,
                    "basis": change.basis, "material": change.material,
                    "confidence": change.confidence}
        if change.action == _states.ACTION_CORRECT_REQUIREMENTS:
            proposal["change"] = self.propose_requirement_change(
                case, updates=change.payload.get("updates") or {})
            proposal["summary"] = proposal["change"]["summary"]
        return proposal

    def _apply(self, case: SyntheticCase, change: ProposedChange, *,
               actor: str) -> SyntheticCase:
        action = change.action
        payload = change.payload
        if action == _states.ACTION_CONFIRM_REQUIREMENTS:
            return self.confirm_requirements(case, actor=actor)
        if action == _states.ACTION_CORRECT_REQUIREMENTS:
            return self.apply_requirement_change(
                case, updates=payload.get("updates") or {}, actor=actor)
        if action == _states.ACTION_GENERATE_PACK:
            return self.generate_onboarding_pack(case, actor=actor)
        if action in (_states.ACTION_APPROVE_PACK, _states.ACTION_ISSUE_PACK):
            return self.approve_onboarding_pack(case, actor=actor)
        if action == _states.ACTION_CLASSIFY_ARTEFACTS:
            return self.classify_artefacts(case, actor=actor)
        if action == _states.ACTION_DRAFT_CONFIG:
            return self.draft_client_config(case, actor=actor)
        if action == _states.ACTION_APPROVE_CONFIG:
            return self.approve_client_config(case, actor=actor)
        if action == _states.ACTION_CORRECT_CONFIG:
            return self.correct_client_config(
                case, updates=payload.get("updates") or {}, actor=actor)
        if action == _states.ACTION_RUN_ONBOARDING:
            return self.run_synthetic_onboarding(case, actor=actor)
        if action == _states.ACTION_RESOLVE_DECISION:
            return self._resolve_from_language(case, payload, actor=actor)
        if action == _states.ACTION_ACKNOWLEDGE_EXCEPTION:
            decision_id = payload.get("decision_id") or _first_non_blocking(case)
            return self.acknowledge_exception(case, decision_id=decision_id,
                                              actor=actor)
        if action == _states.ACTION_GENERATE_PLAN:
            return self.generate_orchestration_plan(case, actor=actor)
        if action == _states.ACTION_APPROVE_EXECUTION:
            return self.approve_execution_readiness(case, actor=actor)
        if action == _states.ACTION_RETURN_TO_STAGE:
            return self.return_to_stage(
                case, target_state=payload.get("target_state", ""), actor=actor)
        if action == _states.ACTION_CANCEL:
            return self.cancel(case, actor=actor)
        raise ActionNotAllowed(action, case.state)  # pragma: no cover

    def _resolve_from_language(self, case: SyntheticCase,
                               payload: Dict[str, Any], *,
                               actor: str) -> SyntheticCase:
        """Turn 'map X to Y' into a decision resolution on the right decision."""
        source = str(payload.get("source_column") or "")
        canonical = str(payload.get("canonical_field") or "")
        decision = next(
            (d for d in case.open_decisions
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
            case, decision_id=decision["decision_id"], action="amend",
            value=canonical, actor=actor,
            reason=f"operator mapped '{source}' to '{canonical}'")

    # ------------------------------------------------------------------ #
    # Questions and status
    # ------------------------------------------------------------------ #
    def answer(self, case: SyntheticCase, question: str) -> str:
        """Answer from case state and the lifecycle table. Never invents."""
        lower = question.lower()
        spec = _states.spec(case.state)
        if "why" in lower and case.open_decisions:
            first = case.blocking_decisions() or case.open_decisions
            d = first[0]
            return (f"{d.get('title') or d.get('question')}\n"
                    f"{d.get('question', '')}\n"
                    f"{(d.get('evidence') or [{}])[0].get('detail', '')}").strip()
        if any(w in lower for w in ("what is left", "what remains", "still",
                                    "outstanding", "before readiness",
                                    "what's left")):
            verdict = _readiness.evaluate(case, self.policy)
            if verdict.ready:
                return ("Every readiness criterion is satisfied. Approve "
                        "readiness to reach READY_FOR_EXECUTION.")
            return "Still outstanding:\n" + "\n".join(
                f"- {c.label}: {c.remedy or c.detail}"
                for c in verdict.outstanding)
        if "stage" in lower or "where" in lower or "status" in lower:
            return self.status_sentence(case)
        if "next" in lower or "can i" in lower or "what should" in lower:
            return ("From here you can: " + ", ".join(
                a.replace("_", " ") for a in spec.allowed_human_actions)
                + ".") if spec.allowed_human_actions else \
                "This case is finished; there is nothing further to do."
        return self.status_sentence(case)

    def status_sentence(self, case: SyntheticCase) -> str:
        spec = _states.spec(case.state)
        parts = [f"This case is at {spec.label}."]
        if case.blockers:
            parts.append("In the way: " + "; ".join(case.blockers[:3]))
        blocking = case.blocking_decisions()
        if blocking:
            parts.append(f"{len(blocking)} decision"
                         f"{'s' if len(blocking) != 1 else ''} need"
                         f"{'' if len(blocking) != 1 else 's'} you.")
        if case.unresolved_questions:
            parts.append("Still unanswered: "
                         + "; ".join(case.unresolved_questions[:3]))
        return " ".join(parts)

    def status(self, case: SyntheticCase) -> Dict[str, Any]:
        """The full status projection the tab renders."""
        verdict = _readiness.evaluate(case, self.policy)
        lifecycle = _states.lifecycle()
        reached = _reached_states(case)
        return {
            "case": case.to_dict(),
            "summary": case.summary_row(),
            "state": _states.describe(case.state),
            "lifecycle": [
                {**entry,
                 "reached": entry["state"] in reached,
                 "current": entry["state"] == case.state}
                for entry in lifecycle],
            "stage_outcomes": case.stage_outcomes,
            "readiness": verdict.to_dict(),
            "policy": self.policy.to_dict(),
            "open_decisions": case.open_decisions,
            "observations": case.observations,
            "blockers": case.blockers,
            "occ_links": _occ_links(case),
            # Surfaced separately so the tab can never present a simulated or
            # blocked stage as a completed one.
            "anything_simulated": _readiness.anything_simulated(case),
            "anything_blocked": _readiness.anything_blocked(case),
        }

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _block(self, case: SyntheticCase, blockers: List[str], *, actor: str,
               reason: str) -> SyntheticCase:
        case.blockers = [b for b in blockers if b]
        prior = case.state
        if _states.is_transition_allowed(case.state, _states.BLOCKED):
            case.state = _states.BLOCKED
        self.store.save(case)
        self._audit(case, "case_blocked", actor_type=ACTOR_SYSTEM, actor=actor,
                    prior_state=prior, classification=EXEC_BLOCKED,
                    decision_basis=reason,
                    detail={"blockers": case.blockers})
        return case

    def _record_control(self, case: SyntheticCase, kind: str,
                        result: Dict[str, Any]) -> None:
        case.control_results.append({"kind": kind, "at": now_iso(), **result})

    def _artefact_paths(self, case: SyntheticCase) -> List[Path]:
        base = self.store.artefact_dir(case.tenant, case.case_id)
        paths: List[Path] = []
        for raw in case.received_artefacts:
            artefact = SyntheticArtefact.from_dict(raw)
            candidate = base / artefact.source_file
            if candidate.exists():
                paths.append(candidate)
        return paths

    def _approved_mappings(self, case: SyntheticCase) -> Dict[str, str]:
        """source column -> canonical field, from resolved mapping decisions.

        An empty target means "do not use this column", which is how the losing
        side of an ambiguity is recorded: both columns are answered, so the next
        run has nothing left to ask about.
        """
        out: Dict[str, str] = {}
        for decision in case.open_decisions:
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

    def _decisions_from_run(self, case: SyntheticCase,
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
            workflow_id=case.case_id, client_id=case.client_id,
            portfolio_id=case.portfolio_id, outcome="mi",
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
            "The synthetic run cannot continue until this is answered."
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


def _first_non_blocking(case: SyntheticCase) -> str:
    for decision in case.open_decisions:
        if not decision.get("blocking") and decision.get("status") == "open":
            return decision["decision_id"]
    return ""


def _reached_states(case: SyntheticCase) -> set:
    """States this case has actually been in, from its own audit-bearing fields.

    Derived from recorded evidence rather than from position in the table, so a
    case that skipped nothing still shows exactly what happened.
    """
    reached = {_states.CASE_CREATED}
    if case.extracted_requirements:
        reached |= {_states.REQUIREMENTS_EXTRACTED,
                    _states.REQUIREMENTS_CONFIRMATION_REQUIRED}
    if case.confirmed_requirements:
        reached.add(_states.REQUIREMENTS_CONFIRMED)
    if case.onboarding_pack:
        reached |= {_states.ONBOARDING_PACK_GENERATED,
                    _states.ONBOARDING_PACK_APPROVAL_REQUIRED}
    if case.pack_issued_synthetically_at:
        reached |= {_states.ONBOARDING_PACK_ISSUED_SYNTHETICALLY,
                    _states.AWAITING_CLIENT_INPUT}
    if case.received_artefacts:
        reached.add(_states.ARTEFACTS_RECEIVED)
    if any(SyntheticArtefact.from_dict(a).recognition_basis
           for a in case.received_artefacts):
        reached.add(_states.ARTEFACTS_CLASSIFIED)
    if case.proposed_configuration:
        reached |= {_states.CONFIG_DRAFTED, _states.CONFIG_APPROVAL_REQUIRED}
    if case.confirmed_configuration:
        reached.add(_states.CONFIG_APPROVED)
    if case.stage_outcomes:
        reached.add(_states.SYNTHETIC_ONBOARDING_RUNNING)
    if case.open_decisions:
        reached.add(_states.EXCEPTIONS_REQUIRE_INPUT)
    if case.stage_outcomes.get("assemble") == STAGE_DETERMINISTIC_COMPLETED:
        reached.add(_states.SYNTHETIC_ONBOARDING_PASSED)
    if case.orchestration_plan:
        reached |= {_states.ORCHESTRATION_PLAN_GENERATED,
                    _states.EXECUTION_APPROVAL_REQUIRED}
    if case.readiness_status == _states.READY_FOR_EXECUTION:
        reached.add(_states.READY_FOR_EXECUTION)
    reached.add(case.state)
    return reached


def _occ_links(case: SyntheticCase) -> List[Dict[str, str]]:
    """Deep links into the EXISTING OCC views, rather than reproducing them."""
    links = [
        {"label": "Platform configuration", "to": "/admin/config",
         "why": "The asset, regime and system packages this case resolved "
                "against."},
        {"label": "Rules", "to": "/rules",
         "why": "The approved mapping and alias rules the platform applies."},
    ]
    if case.blocking_decisions():
        links.append({"label": "Review", "to": "/reviews",
                      "why": "How the same decisions are answered on live "
                             "deliveries."})
    if case.stage_outcomes:
        links.append({"label": "Workflows", "to": "/workflows",
                      "why": "Where a live run of this package would appear."})
    return links
