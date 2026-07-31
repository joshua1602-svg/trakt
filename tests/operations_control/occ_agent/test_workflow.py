"""The operating process, end to end.

The five fixtures and the human loop around them: what reaches
``READY_FOR_EXECUTION``, what does not, and what a human can and cannot do to
change that.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from operations_control.occ_agent import fixtures as _fixtures
from operations_control.occ_agent import readiness as _readiness
from operations_control.occ_agent import states as _states
from operations_control.occ_agent.scenarios import run_scenario
from operations_control.occ_agent.service import ActionNotAllowed

from .conftest import ACTOR, TENANT_A


# --------------------------------------------------------------------------- #
# The five scenarios
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("scenario", _fixtures.SCENARIOS,
                         ids=lambda s: s.fixture_id)
def test_each_fixture_reaches_its_declared_outcome(service, scenario):
    run = run_scenario(service, scenario, tenant=TENANT_A, actor=ACTOR)
    assert run.case.state == scenario.expected_state, run.stopped_because


def test_scenario_a_reaches_ready_for_execution(service):
    run = run_scenario(service, "scenario_a_clean", tenant=TENANT_A, actor=ACTOR)
    assert run.case.state == _states.READY_FOR_EXECUTION
    assert run.case.readiness_status == _states.READY_FOR_EXECUTION
    assert run.case.blockers == []
    verdict = service.evaluate_readiness(run.case)
    assert verdict["ready"] is True
    assert verdict["outstanding"] == []


def test_scenario_b_needs_a_human_then_reruns_the_affected_controls(service):
    """The halt is real, the human settles it, and the controls run again."""
    halted = run_scenario(service, "scenario_b_ambiguous_mapping",
                          tenant=TENANT_A, actor=ACTOR,
                          resolve_decisions=False)
    assert halted.case.state == _states.EXCEPTIONS_REQUIRE_INPUT
    decision = next(d for d in halted.case.open_decisions
                    if d["status"] == "open")
    assert decision["blocking"] is True
    # Before the decision, validation has not run at all.
    assert "validate" not in halted.case.stage_outcomes

    case = service.resolve_decision(
        halted.case, decision_id=decision["decision_id"], action="approve",
        value=decision["recommendation"], actor=ACTOR,
        reason="accepted the recommendation")
    # Resolving reran the affected controls, which now completed.
    assert case.stage_outcomes.get("validate") == "deterministic_execution_completed"
    assert case.state == _states.SYNTHETIC_ONBOARDING_PASSED


def test_scenario_c_stays_blocked_with_a_targeted_question(service):
    run = run_scenario(service, "scenario_c_missing_artefact",
                       tenant=TENANT_A, actor=ACTOR)
    assert run.case.state == _states.BLOCKED
    assert any("loan tape" in b.lower() for b in run.case.blockers)
    assert service.evaluate_readiness(run.case)["ready"] is False


def test_scenario_d_blocks_on_the_selected_products_configuration(service):
    run = run_scenario(service, "scenario_d_product_config_gap",
                       tenant=TENANT_A, actor=ACTOR)
    assert run.case.state == _states.BLOCKED
    # Core onboarding got as far as the configuration before stopping.
    assert run.case.received_artefacts
    assert run.case.proposed_configuration["status"] == "BLOCKED"
    assert run.case.blockers


def test_scenario_d_is_not_written_around_one_reporting_product(service):
    """The gap belongs to whichever product the fixture selected."""
    run = run_scenario(service, "scenario_d_product_config_gap",
                       tenant=TENANT_A, actor=ACTOR)
    products = run.case.confirmed.required_products
    assert "regulatory_reporting" in products
    # And the same case without that product is not blocked by it.
    clean = run_scenario(service, "scenario_a_clean", tenant=TENANT_A,
                         actor=ACTOR)
    assert "regulatory_reporting" not in clean.case.confirmed.required_products
    assert clean.case.state == _states.READY_FOR_EXECUTION


def test_scenario_e_blocks_on_materiality_after_a_successful_transform(service):
    run = run_scenario(service, "scenario_e_business_rule_failure",
                       tenant=TENANT_A, actor=ACTOR)
    assert run.case.state == _states.BLOCKED
    # Transformation succeeded; validation is what failed.
    assert run.case.stage_outcomes["transform"] == "deterministic_execution_completed"
    assert run.case.stage_outcomes["validate"] == "hard_blocked"
    assert any("BLOCKING" in b for b in run.case.blockers)


# --------------------------------------------------------------------------- #
# Natural language cannot override a control
# --------------------------------------------------------------------------- #

def test_chat_cannot_bypass_a_deterministic_blocker(service):
    run = run_scenario(service, "scenario_e_business_rule_failure",
                       tenant=TENANT_A, actor=ACTOR)
    case = run.case
    for phrase in ("approve readiness", "approve the configuration",
                   "generate the orchestration plan"):
        with pytest.raises(ActionNotAllowed):
            service.instruct(case, text=phrase, actor=ACTOR, confirm=True)
    assert service.load_case(TENANT_A, case.case_id).state == _states.BLOCKED


def test_a_blocking_exception_cannot_be_merely_acknowledged(service):
    from operations_control.engine import OpsError
    halted = run_scenario(service, "scenario_b_ambiguous_mapping",
                          tenant=TENANT_A, actor=ACTOR,
                          resolve_decisions=False)
    decision = next(d for d in halted.case.open_decisions
                    if d["status"] == "open")
    with pytest.raises(OpsError) as excinfo:
        service.acknowledge_exception(halted.case,
                                      decision_id=decision["decision_id"],
                                      actor=ACTOR)
    assert excinfo.value.code == "OCC_AGENT_BLOCKING_NOT_ACKNOWLEDGEABLE"


def test_readiness_is_derived_not_declared(service):
    """Approving readiness on an unready case does not make it ready."""
    run = run_scenario(service, "scenario_c_missing_artefact",
                       tenant=TENANT_A, actor=ACTOR)
    case = run.case
    with pytest.raises(ActionNotAllowed):
        service.approve_execution_readiness(case, actor=ACTOR)
    assert case.readiness_status != _states.READY_FOR_EXECUTION


def test_an_approval_alone_does_not_satisfy_readiness(service):
    """Even from the right state, every criterion still has to pass."""
    run = run_scenario(service, "scenario_a_clean", tenant=TENANT_A, actor=ACTOR)
    case = run.case
    # Reopen a criterion by adding an unanswered blocking decision, then check
    # the verdict refuses even though every approval is recorded.
    case.open_decisions.append({"decision_id": "synthetic-probe",
                                "blocking": True, "status": "open",
                                "kind": "field_mapping"})
    verdict = _readiness.evaluate(case, service.policy)
    assert verdict.ready is False
    assert any(c.key == "exceptions_cleared" for c in verdict.outstanding)


# --------------------------------------------------------------------------- #
# Human corrections
# --------------------------------------------------------------------------- #

def test_a_correction_updates_structured_state_not_just_the_chat(service):
    case = service.create_case(
        tenant=TENANT_A, initiating_user=ACTOR,
        instruction="Onboard Northstar Lending. UK equity release. Monthly "
                    "portfolio MI. Portfolio id direct_101.")
    assert case.extracted.reporting_date == ""
    case = service.apply_requirement_change(
        case, updates={"reporting_date": "2026-06-30"}, actor=ACTOR)
    reloaded = service.load_case(TENANT_A, case.case_id)
    assert reloaded.extracted.reporting_date == "2026-06-30"
    # The question it answered is gone from the outstanding list.
    assert not any("first reporting date" in q.lower()
                   for q in reloaded.unresolved_questions)


def test_a_correction_regenerates_only_the_affected_pack_sections(service):
    from operations_control.occ_agent import pack as _pack
    case = service.create_case(
        tenant=TENANT_A, initiating_user=ACTOR,
        instruction="Onboard Northstar Lending. UK equity release. Monthly "
                    "portfolio MI. Portfolio id direct_101. First reporting "
                    "date 2026-06-30. Go-live 2026-09-30.")
    case = service.confirm_requirements(case, actor=ACTOR)
    case = service.generate_onboarding_pack(case, actor=ACTOR)
    before = {s["key"]: s for s in case.onboarding_pack["sections"]}

    proposal = service.propose_requirement_change(
        case, updates={"reporting_date": "2026-09-30"})
    affected = set(proposal["affected_pack_sections"])
    assert _pack.SEC_EMAIL in affected
    # A section that does not depend on the reporting date is untouched.
    assert _pack.SEC_PRODUCT_QUESTIONS not in affected

    case = service.apply_requirement_change(
        case, updates={"reporting_date": "2026-09-30"}, actor=ACTOR)
    after = {s["key"]: s for s in case.onboarding_pack["sections"]}
    assert after[_pack.SEC_EMAIL] != before[_pack.SEC_EMAIL]
    assert after[_pack.SEC_PRODUCT_QUESTIONS] == before[_pack.SEC_PRODUCT_QUESTIONS]


def test_a_correction_to_confirmed_facts_reopens_confirmation(service):
    case = service.create_case(
        tenant=TENANT_A, initiating_user=ACTOR,
        instruction="Onboard Northstar Lending. UK equity release. Monthly "
                    "portfolio MI. Portfolio id direct_101. First reporting "
                    "date 2026-06-30.")
    case = service.confirm_requirements(case, actor=ACTOR)
    assert case.state == _states.REQUIREMENTS_CONFIRMED
    case = service.apply_requirement_change(
        case, updates={"reporting_date": "2026-07-31"}, actor=ACTOR)
    assert case.state == _states.REQUIREMENTS_CONFIRMATION_REQUIRED


def test_a_material_instruction_is_proposed_before_it_is_applied(service):
    case = service.create_case(
        tenant=TENANT_A, initiating_user=ACTOR,
        instruction="Onboard Northstar Lending. UK equity release. Monthly "
                    "portfolio MI. Portfolio id direct_101. First reporting "
                    "date 2026-06-30.")
    turn = service.instruct(case, text="confirm the interpretation",
                            actor=ACTOR)
    assert turn.applied is False
    assert turn.proposal is not None
    assert turn.case.state == _states.REQUIREMENTS_CONFIRMATION_REQUIRED

    turn = service.instruct(turn.case, text="confirm the interpretation",
                            actor=ACTOR, confirm=True)
    assert turn.applied is True
    assert turn.case.state == _states.REQUIREMENTS_CONFIRMED


def test_a_question_changes_nothing(service):
    case = service.create_case(
        tenant=TENANT_A, initiating_user=ACTOR,
        instruction="Onboard Northstar Lending. UK equity release.")
    before = case.state
    turn = service.instruct(case, text="What is still needed?", actor=ACTOR)
    assert turn.applied is False
    assert turn.case.state == before
    assert "outstanding" in turn.reply.lower() or "criterion" in turn.reply.lower()


def test_a_mapping_instruction_resolves_the_right_decision(service):
    halted = run_scenario(service, "scenario_b_ambiguous_mapping",
                          tenant=TENANT_A, actor=ACTOR,
                          resolve_decisions=False)
    turn = service.instruct(
        halted.case,
        text="Map Current Principal Balance to current principal balance.",
        actor=ACTOR, confirm=True)
    resolved = [d for d in turn.case.open_decisions
                if d.get("status") == "approved"]
    assert resolved, "the mapping instruction did not settle a decision"


def test_a_case_can_be_returned_to_an_earlier_permitted_stage(service):
    case = service.create_case(
        tenant=TENANT_A, initiating_user=ACTOR,
        instruction="Onboard Northstar Lending. UK equity release. Monthly "
                    "portfolio MI. Portfolio id direct_101. First reporting "
                    "date 2026-06-30.")
    case = service.confirm_requirements(case, actor=ACTOR)
    case = service.return_to_stage(
        case, target_state=_states.REQUIREMENTS_EXTRACTED, actor=ACTOR)
    assert case.state == _states.REQUIREMENTS_EXTRACTED


def test_a_case_cannot_be_returned_to_a_stage_that_skips_a_control(service):
    from operations_control.occ_agent.states import IllegalCaseTransition
    case = service.create_case(
        tenant=TENANT_A, initiating_user=ACTOR,
        instruction="Onboard Northstar Lending. UK equity release. Monthly "
                    "portfolio MI. Portfolio id direct_101. First reporting "
                    "date 2026-06-30.")
    case = service.confirm_requirements(case, actor=ACTOR)
    with pytest.raises(IllegalCaseTransition):
        service.return_to_stage(case, target_state=_states.READY_FOR_EXECUTION,
                                actor=ACTOR)


# --------------------------------------------------------------------------- #
# The readiness package
# --------------------------------------------------------------------------- #

def test_the_readiness_package_carries_every_required_part(service):
    run = run_scenario(service, "scenario_a_clean", tenant=TENANT_A, actor=ACTOR)
    package = service.readiness_package(run.case)
    for part in ("case_summary", "confirmed_facts", "approved_product_scope",
                 "approved_configuration", "artefact_inventory",
                 "intended_live_storage_paths", "field_mapping_report",
                 "validation_summary", "approved_exceptions",
                 "outstanding_observations", "orchestration_execution_plan",
                 "assembler_input_plan", "expected_downstream_outputs",
                 "human_approvals", "audit_trail", "execution_manifest"):
        assert part in package, part


def test_the_execution_manifest_is_deterministic(service):
    """The same case produces the same manifest, twice running."""
    run = run_scenario(service, "scenario_a_clean", tenant=TENANT_A, actor=ACTOR)
    first = service.readiness_package(run.case)["execution_manifest"]
    second = service.readiness_package(run.case)["execution_manifest"]
    assert first == second
    assert first["content_hash"] == second["content_hash"]
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


def test_two_runs_of_the_same_fixture_produce_the_same_manifest_body(service,
                                                                     storage,
                                                                     agent_env):
    """Determinism across cases, not just across calls.

    The identity fields differ (a case id is unique); everything the manifest
    says about the WORK must not.
    """
    from operations_control.occ_agent.service import OccAgentService
    first = run_scenario(service, "scenario_a_clean", tenant=TENANT_A,
                         actor=ACTOR)
    other = OccAgentService(storage, container="operations-control-synthetic",
                            sandbox=agent_env["sandbox"])
    second = run_scenario(other, "scenario_a_clean", tenant=TENANT_A,
                          actor=ACTOR)
    volatile = {"case_id"}
    a = {k: v for k, v in
         service.readiness_package(first.case)["execution_manifest"].items()
         if k not in volatile | {"content_hash"}}
    b = {k: v for k, v in
         other.readiness_package(second.case)["execution_manifest"].items()
         if k not in volatile | {"content_hash"}}
    assert a == b


def test_the_package_states_what_did_not_happen(service):
    run = run_scenario(service, "scenario_a_clean", tenant=TENANT_A, actor=ACTOR)
    package = service.readiness_package(run.case)
    statement = package["statement"]
    assert statement["headline"] == _readiness.READY_HEADLINE
    assert statement["not_done"] == list(_readiness.NOT_DONE_STATEMENTS)


def test_the_package_never_claims_completion_or_publication(service):
    run = run_scenario(service, "scenario_a_clean", tenant=TENANT_A, actor=ACTOR)
    package = service.readiness_package(run.case)
    text = json.dumps(package).lower()
    assert "production successful" not in text
    assert '"published": true' not in text
    assert package["execution_manifest"]["readiness_status"] == \
        _states.READY_FOR_EXECUTION


def test_the_package_is_written_inside_the_case_sandbox(service, agent_env):
    run = run_scenario(service, "scenario_a_clean", tenant=TENANT_A, actor=ACTOR)
    assert run.case.readiness_package_ref.startswith(
        f"synthetic_cases/{run.case.case_id}/")
    written = service.store.package_dir(TENANT_A, run.case.case_id) \
        / "readiness_package.json"
    assert written.exists()
    assert Path(agent_env["sandbox"]).resolve() in written.parents


def test_every_stage_declares_which_of_the_five_outcomes_it_reached(service):
    from operations_control.occ_agent.case import STAGE_OUTCOMES
    run = run_scenario(service, "scenario_a_clean", tenant=TENANT_A, actor=ACTOR)
    assert run.case.stage_outcomes
    for stage, outcome in run.case.stage_outcomes.items():
        assert outcome in STAGE_OUTCOMES, f"{stage} -> {outcome}"


def test_a_simulated_stage_is_never_reported_as_executed(service):
    """The regime projection is simulated, and says so."""
    from operations_control.occ_agent.execution import SyntheticOnboardingAdapters
    from operations_control.occ_agent.case import STAGE_SIMULATED
    case = service.create_case(tenant=TENANT_A, initiating_user=ACTOR,
                               instruction="Onboard Northstar Lending.")
    sandbox = service.store.case_dir(TENANT_A, case.case_id)
    adapters = SyntheticOnboardingAdapters(
        artefact_paths=[], policy=service.policy, sandbox=sandbox,
        case_id=case.case_id, tenant=TENANT_A)
    central = sandbox / "central.csv"
    central.write_text("a,b\n1,2\n", encoding="utf-8")
    result = adapters.project(str(central), sandbox / "out", "ESMA_Annex2")
    assert result.ok is True
    assert "simulated" in result.message
    assert result.readiness["execution_status"] == "simulated_only"
    assert adapters.records[-1].outcome == STAGE_SIMULATED
    plan = json.loads(Path(result.output_path).read_text(encoding="utf-8"))
    assert plan["execution_status"] == "simulated_only"


# --------------------------------------------------------------------------- #
# Configuration provenance and materiality
# --------------------------------------------------------------------------- #

def test_every_proposed_configuration_value_carries_provenance(service):
    run = run_scenario(service, "scenario_a_clean", tenant=TENANT_A, actor=ACTOR)
    from operations_control.occ_agent.case import PROVENANCE_SOURCES
    provenance = run.case.configuration_provenance
    assert provenance
    for value in provenance:
        assert value["source"] in PROVENANCE_SOURCES
        assert "evidence" in value


def test_a_material_value_cannot_pass_unconfirmed(service):
    from operations_control.occ_agent.case import ProposedValue
    run = run_scenario(service, "scenario_a_clean", tenant=TENANT_A, actor=ACTOR)
    # After approval every material value is confirmed.
    assert run.case.unconfirmed_material_values() == []
    # And an unconfirmed one would hold readiness.
    probe = ProposedValue(key="portfolio.base_currency", value="GBP",
                          source="agent_inference", confidence=0.2,
                          material=True)
    assert probe.needs_confirmation() is True
    run.case.configuration_provenance.append(probe.to_dict())
    verdict = _readiness.evaluate(run.case, service.policy)
    assert verdict.ready is False
    assert any(c.key == "material_values_confirmed" for c in verdict.outstanding)


def test_a_low_confidence_inference_needs_confirmation_even_when_not_material():
    from operations_control.occ_agent.case import ProposedValue
    low = ProposedValue(key="pipeline.mi_enabled", value=True,
                        source="agent_inference", confidence=0.3)
    assert low.needs_confirmation() is True
    high = ProposedValue(key="pipeline.mi_enabled", value=True,
                         source="agent_inference", confidence=0.95)
    assert high.needs_confirmation() is False


# --------------------------------------------------------------------------- #
# Closing a product-configuration gap
# --------------------------------------------------------------------------- #

def test_supplying_the_missing_configuration_unblocks_scenario_d(service):
    """The gap closes because the RESOLVER accepts it, not because we said so."""
    run = run_scenario(service, "scenario_d_product_config_gap",
                       tenant=TENANT_A, actor=ACTOR)
    assert run.case.state == _states.BLOCKED

    case = service.correct_client_config(
        run.case,
        updates={
            "defaults.originator_legal_entity_identifier":
                "894500SYNTHETIC00042",
            "defaults.originator_name": "Aldermere Advances",
            "defaults.originator_establishment_country": "GB",
        },
        actor=ACTOR)
    assert case.state == _states.CONFIG_APPROVAL_REQUIRED
    assert case.proposed_configuration["status"] in ("READY",
                                                     "READY_WITH_WARNINGS")
    assert case.blockers == []

    # And the supplied values are recorded as a client response, not an
    # inference.
    from operations_control.occ_agent.case import PROV_CLIENT_RESPONSE
    supplied = [v for v in case.configuration_provenance
                if v["key"].startswith("defaults.originator")]
    assert supplied
    assert all(v["source"] == PROV_CLIENT_RESPONSE for v in supplied)


def test_the_supplied_case_then_reaches_ready_for_execution(service):
    run = run_scenario(service, "scenario_d_product_config_gap",
                       tenant=TENANT_A, actor=ACTOR)
    case = service.correct_client_config(
        run.case,
        updates={
            "defaults.originator_legal_entity_identifier":
                "894500SYNTHETIC00042",
            "defaults.originator_name": "Aldermere Advances",
            "defaults.originator_establishment_country": "GB",
        },
        actor=ACTOR)
    case = service.approve_client_config(case, actor=ACTOR)
    case = service.run_synthetic_onboarding(case, actor=ACTOR)
    if case.state == _states.EXCEPTIONS_REQUIRE_INPUT:
        for decision in list(case.open_decisions):
            if decision.get("status") == "open":
                case = service.resolve_decision(
                    case, decision_id=decision["decision_id"],
                    action="approve", value=str(decision.get("recommendation")
                                                or ""), actor=ACTOR)
    assert case.state == _states.SYNTHETIC_ONBOARDING_PASSED, case.blockers
    case = service.generate_orchestration_plan(case, actor=ACTOR)
    case = service.approve_execution_readiness(case, actor=ACTOR)
    assert case.state == _states.READY_FOR_EXECUTION


def test_an_unknown_configuration_key_cannot_be_set(service):
    from operations_control.engine import OpsError
    run = run_scenario(service, "scenario_d_product_config_gap",
                       tenant=TENANT_A, actor=ACTOR)
    with pytest.raises(OpsError) as excinfo:
        service.correct_client_config(
            run.case, updates={"pipeline.esma_enabled": "false"}, actor=ACTOR)
    assert excinfo.value.code == "OCC_AGENT_UNKNOWN_CONFIG_KEY"


def test_a_configuration_answer_can_be_given_in_language(service):
    run = run_scenario(service, "scenario_d_product_config_gap",
                       tenant=TENANT_A, actor=ACTOR)
    turn = service.instruct(
        run.case,
        text="Set the originator legal entity identifier to 894500SYNTHETIC00042.",
        actor=ACTOR)
    assert turn.applied is False           # material: proposed first
    assert turn.proposal["action"] == _states.ACTION_CORRECT_CONFIG
    turn = service.instruct(
        turn.case,
        text="Set the originator legal entity identifier to 894500SYNTHETIC00042.",
        actor=ACTOR, confirm=True)
    assert turn.applied is True
    assert turn.case.client_answers[
        "defaults.originator_legal_entity_identifier"] == "894500SYNTHETIC00042"


def test_no_forbidden_outcome_word_describes_a_ready_case(service):
    """§17: the outcome is READY_FOR_EXECUTION, never 'complete' or 'published'."""
    run = run_scenario(service, "scenario_a_clean", tenant=TENANT_A, actor=ACTOR)
    package = service.readiness_package(run.case)
    described = " ".join([
        package["case_summary"]["status"],
        package["statement"]["headline"],
        package["readiness"]["status"],
        package["execution_manifest"]["readiness_status"],
    ]).lower()
    for word in _readiness.FORBIDDEN_OUTCOME_WORDS:
        assert word not in described, word
    assert _states.READY_FOR_EXECUTION in package["case_summary"]["status"]
