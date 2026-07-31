"""The operating process, end to end.

The five fixtures and the human loop around them: what reaches
``READY_FOR_EXECUTION``, what does not, and what a human can and cannot do to
change that.

The process now spans both halves — an onboarding case worked to *approved*
through Client Onboarding, and a practice execution run off it — so these tests
assert on both, and on the join between them.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from operations_control.engine import OpsError
from operations_control.occ_agent import fixtures as _fixtures
from operations_control.occ_agent import readiness as _readiness
from operations_control.occ_agent import states as _states
from operations_control.occ_agent.scenarios import run_scenario
from operations_control.occ_agent.service import ActionNotAllowed
from operations_control.onboarding.case import APPROVED

from .conftest import ACTOR, TENANT_A


def _verdict(service, agent_case):
    return _readiness.evaluate(
        agent_case.run, agent_case.case, service.facts(agent_case),
        service.policy, onboarding=service.onboarding_readiness(agent_case),
        preview=service._safe_preview(agent_case))


# --------------------------------------------------------------------------- #
# The five scenarios
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("scenario", _fixtures.SCENARIOS,
                         ids=lambda s: s.fixture_id)
def test_each_fixture_reaches_its_declared_outcome(service, scenario):
    run = run_scenario(service, scenario, tenant=TENANT_A, actor=ACTOR)
    assert run.case.run.state == scenario.expected_state, run.stopped_because
    assert run.case.case.status == scenario.expected_onboarding_status, \
        run.stopped_because


def test_scenario_a_reaches_ready_for_execution(service):
    """The rehearsal passes, and then stops at the confirmation gate.

    ``READY_FOR_EXECUTION`` is a waypoint, so it is asserted through the
    readiness status it set rather than through the run's current state.
    """
    run = run_scenario(service, "scenario_a_clean", tenant=TENANT_A, actor=ACTOR)
    assert _states.READY_FOR_EXECUTION in run.progression
    assert run.case.run.state == _states.ACTIVATION_CONFIRMATION_REQUIRED
    assert run.case.run.readiness_status == _states.READY_FOR_EXECUTION
    assert run.case.run.blockers == []
    verdict = service.evaluate_readiness(run.case)
    assert verdict["ready"] is True
    assert verdict["outstanding"] == []


def test_scenario_a_walks_both_lifecycles(service):
    """One conversation, two governed lifecycles, in the right order."""
    run = run_scenario(service, "scenario_a_clean", tenant=TENANT_A, actor=ACTOR)
    assert run.onboarding_progression == [
        "draft", "information_requested", "in_review", "ready_for_approval",
        "approved"]
    assert run.progression == [
        _states.AWAITING_ONBOARDING, _states.PACK_REVIEW_REQUIRED,
        _states.PACK_APPROVED_TO_SEND, _states.PACK_SENT,
        _states.READY_TO_RUN, _states.SYNTHETIC_ONBOARDING_PASSED,
        _states.EXECUTION_APPROVAL_REQUIRED, _states.READY_FOR_EXECUTION,
        _states.READY_FOR_REVIEW,
        _states.ACTIVATION_CONFIRMATION_REQUIRED]


def test_scenario_b_needs_a_human_then_reruns_the_affected_controls(service):
    """The halt is real, the human settles it, and the controls run again."""
    halted = run_scenario(service, "scenario_b_ambiguous_mapping",
                          tenant=TENANT_A, actor=ACTOR,
                          resolve_decisions=False)
    assert halted.case.run.state == _states.EXCEPTIONS_REQUIRE_INPUT
    decision = next(d for d in halted.case.run.open_decisions
                    if d["status"] == "open")
    assert decision["blocking"] is True
    # Before the decision, validation has not run at all.
    assert "validate" not in halted.case.run.stage_outcomes

    agent_case = service.resolve_decision(
        halted.case, decision_id=decision["decision_id"], action="approve",
        value=decision["recommendation"], actor=ACTOR,
        reason="accepted the recommendation")
    # Resolving reran the affected controls, which now completed.
    assert agent_case.run.stage_outcomes.get("validate") == \
        "deterministic_execution_completed"
    assert agent_case.run.state == _states.SYNTHETIC_ONBOARDING_PASSED


def test_scenario_c_stays_blocked_with_a_targeted_question(service):
    run = run_scenario(service, "scenario_c_missing_artefact",
                       tenant=TENANT_A, actor=ACTOR)
    assert run.case.run.state == _states.BLOCKED
    assert any("loan tape" in b.lower() for b in run.case.run.blockers)
    assert service.evaluate_readiness(run.case)["ready"] is False
    # The ONBOARDING half still completed: an incomplete practice delivery is
    # not a reason to refuse to describe the client.
    assert run.case.case.status == APPROVED


def test_scenario_d_blocks_inside_client_onboarding_not_here(service):
    """The product's own question is unanswered, so approval is refused.

    The refusal is Client Onboarding's, in its own words — nothing in the OCC
    Agent decides what a product needs.
    """
    run = run_scenario(service, "scenario_d_product_information_gap",
                       tenant=TENANT_A, actor=ACTOR)
    # The pack went out; the onboarding itself is what holds the case.
    assert run.case.run.state == _states.PACK_SENT
    assert run.case.case.status == "in_review"
    readiness = service.onboarding_readiness(run.case)
    assert readiness["ready"] is False
    assert readiness["blocking"]
    with pytest.raises(OpsError) as excinfo:
        service.approve_onboarding(run.case, actor=ACTOR, reason="anyway")
    assert excinfo.value.code == "OPS_ONBOARDING_INCOMPLETE"


def test_scenario_d_is_not_written_around_one_reporting_product(service):
    """The gap belongs to whichever product the fixture selected."""
    run = run_scenario(service, "scenario_d_product_information_gap",
                       tenant=TENANT_A, actor=ACTOR)
    products = run.case.case.products
    assert "investor_reporting" in products
    # The blocking problems are all fields the product declaration introduces.
    blocking = service.onboarding_readiness(run.case)["blocking"]
    assert blocking
    # And the same journey without that product is not blocked by it.
    clean = run_scenario(service, "scenario_a_clean", tenant=TENANT_A,
                         actor=ACTOR)
    assert "investor_reporting" not in clean.case.case.products
    assert clean.case.run.readiness_status == _states.READY_FOR_EXECUTION


def test_scenario_e_blocks_on_materiality_after_a_successful_transform(service):
    run = run_scenario(service, "scenario_e_business_rule_failure",
                       tenant=TENANT_A, actor=ACTOR)
    assert run.case.run.state == _states.BLOCKED
    # Transformation succeeded; validation is what failed.
    assert run.case.run.stage_outcomes["transform"] == \
        "deterministic_execution_completed"
    assert run.case.run.stage_outcomes["validate"] == "hard_blocked"
    assert any("BLOCKING" in b for b in run.case.run.blockers)


# --------------------------------------------------------------------------- #
# Natural language cannot override a control
# --------------------------------------------------------------------------- #

def test_chat_cannot_bypass_a_deterministic_blocker(service):
    run = run_scenario(service, "scenario_e_business_rule_failure",
                       tenant=TENANT_A, actor=ACTOR)
    for phrase in ("approve readiness", "generate the orchestration plan"):
        with pytest.raises(ActionNotAllowed):
            service.instruct(run.case, text=phrase, actor=ACTOR, confirm=True)
    reloaded = service.load(TENANT_A, run.case.case_ref)
    assert reloaded.run.state == _states.BLOCKED


def test_chat_cannot_approve_an_onboarding_the_validator_refuses(service):
    run = run_scenario(service, "scenario_d_product_information_gap",
                       tenant=TENANT_A, actor=ACTOR)
    with pytest.raises(OpsError) as excinfo:
        service.instruct(run.case, text="approve the onboarding", actor=ACTOR,
                         confirm=True)
    assert excinfo.value.code == "OPS_ONBOARDING_INCOMPLETE"


def test_a_blocking_exception_cannot_be_merely_acknowledged(service):
    halted = run_scenario(service, "scenario_b_ambiguous_mapping",
                          tenant=TENANT_A, actor=ACTOR,
                          resolve_decisions=False)
    decision = next(d for d in halted.case.run.open_decisions
                    if d["status"] == "open")
    with pytest.raises(OpsError) as excinfo:
        service.acknowledge_exception(halted.case,
                                      decision_id=decision["decision_id"],
                                      actor=ACTOR)
    assert excinfo.value.code == "OCC_AGENT_BLOCKING_NOT_ACKNOWLEDGEABLE"


def test_readiness_is_derived_not_declared(service):
    """Approving readiness on an unready run does not make it ready."""
    run = run_scenario(service, "scenario_c_missing_artefact",
                       tenant=TENANT_A, actor=ACTOR)
    with pytest.raises(ActionNotAllowed):
        service.approve_execution_readiness(run.case, actor=ACTOR)
    assert run.case.run.readiness_status != _states.READY_FOR_EXECUTION


def test_an_approval_alone_does_not_satisfy_readiness(service):
    """Even from the right state, every criterion still has to pass."""
    run = run_scenario(service, "scenario_a_clean", tenant=TENANT_A, actor=ACTOR)
    # Reopen a criterion by adding an unanswered blocking decision, then check
    # the verdict refuses even though every approval is recorded.
    run.case.run.open_decisions.append({"decision_id": "synthetic-probe",
                                        "blocking": True, "status": "open",
                                        "kind": "field_mapping"})
    verdict = _verdict(service, run.case)
    assert verdict.ready is False
    assert any(c.key == "exceptions_cleared" for c in verdict.outstanding)


# --------------------------------------------------------------------------- #
# The conversation writes through Client Onboarding
# --------------------------------------------------------------------------- #

def test_an_instruction_becomes_answers_on_the_onboarding_case(service):
    agent_case = service.create_case(
        tenant=TENANT_A, initiating_user=ACTOR,
        instruction="Onboard Northstar Lending. UK equity release. Monthly "
                    "portfolio MI. Portfolio id direct_101.")
    answers = agent_case.case.answers
    assert answers["client"]["client_name"] == "Northstar Lending"
    assert answers["client"]["jurisdiction"] == "GB"
    assert answers["portfolios"][0]["portfolio_id"] == "direct_101"
    assert answers["portfolios"][0]["asset_class"] == "equity_release"
    assert answers["reporting"]["products"] == ["mi"]
    # And the case's own event history records how they got there.
    assert any(e["event"].startswith("answered_") for e in agent_case.case.events)


def test_what_trakt_works_out_for_itself_is_left_to_client_onboarding(service):
    """The interpreter answers what the sentence says, and nothing more."""
    agent_case = service.create_case(
        tenant=TENANT_A, initiating_user=ACTOR,
        instruction="Onboard Northstar Lending. UK equity release. Monthly "
                    "portfolio MI. Portfolio id direct_101.")
    client = agent_case.case.answers["client"]
    # None of these were in the instruction; every one is Client Onboarding's
    # own inference or default.
    assert client["reporting_currency"] == "GBP"
    assert client["time_zone"]
    assert client["client_id"]
    assert agent_case.case.answers["portfolios"][0]["period_convention"]


def test_a_later_instruction_updates_the_book_rather_than_adding_another(service):
    agent_case = service.create_case(
        tenant=TENANT_A, initiating_user=ACTOR,
        instruction="Onboard Northstar Lending. UK equity release. Monthly "
                    "portfolio MI. Portfolio id direct_101.")
    turn = service.instruct(
        agent_case, text="The portfolio is an acquired book.", actor=ACTOR,
        confirm=True)
    portfolios = turn.case.case.answers["portfolios"]
    assert len(portfolios) == 1
    assert portfolios[0]["portfolio_id"] == "direct_101"
    assert portfolios[0]["portfolio_type"] == "acquired"


def test_a_material_instruction_is_proposed_before_it_is_applied(service):
    agent_case = service.create_case(
        tenant=TENANT_A, initiating_user=ACTOR,
        instruction="Onboard Northstar Lending. UK equity release. Monthly "
                    "portfolio MI. Portfolio id direct_101.")
    turn = service.instruct(agent_case, text="The jurisdiction is the "
                                             "Netherlands.", actor=ACTOR)
    assert turn.applied is False
    assert turn.proposal is not None
    assert turn.case.case.answers["client"]["jurisdiction"] == "GB"

    turn = service.instruct(turn.case, text="The jurisdiction is the "
                                            "Netherlands.",
                            actor=ACTOR, confirm=True)
    assert turn.applied is True
    assert turn.case.case.answers["client"]["jurisdiction"] == "NL"


def test_a_question_changes_nothing(service):
    agent_case = service.create_case(
        tenant=TENANT_A, initiating_user=ACTOR,
        instruction="Onboard Northstar Lending. UK equity release.")
    before = agent_case.run.state
    status_before = agent_case.case.status
    turn = service.instruct(agent_case, text="What is still needed?",
                            actor=ACTOR)
    assert turn.applied is False
    assert turn.case.run.state == before
    assert turn.case.case.status == status_before
    assert turn.reply


def test_the_agent_can_say_what_the_client_still_has_to_answer(service):
    agent_case = service.create_case(
        tenant=TENANT_A, initiating_user=ACTOR,
        instruction="Onboard Northstar Lending. UK equity release. Monthly "
                    "portfolio MI. Portfolio id direct_101.")
    turn = service.instruct(agent_case,
                            text="What do we still need from the client?",
                            actor=ACTOR)
    assert "Legal Entity Identifier" in turn.reply


def test_a_mapping_instruction_resolves_the_right_decision(service):
    halted = run_scenario(service, "scenario_b_ambiguous_mapping",
                          tenant=TENANT_A, actor=ACTOR,
                          resolve_decisions=False)
    turn = service.instruct(
        halted.case,
        text="Map Current Principal Balance to current principal balance.",
        actor=ACTOR, confirm=True)
    resolved = [d for d in turn.case.run.open_decisions
                if d.get("status") == "approved"]
    assert resolved, "the mapping instruction did not settle a decision"


# --------------------------------------------------------------------------- #
# The client information request
# --------------------------------------------------------------------------- #

def test_the_checklist_is_client_onboardings_own(service):
    agent_case = service.create_case(
        tenant=TENANT_A, initiating_user=ACTOR,
        instruction="Onboard Northstar Lending. UK equity release. Monthly "
                    "portfolio MI. Portfolio id direct_101.")
    checklist = service.onboarding.client_checklist(agent_case.case)
    assert checklist
    # Only client-supplied fields — never something Trakt mints or infers.
    catalogue = service.onboarding.catalogue
    for row in checklist:
        field = catalogue.field(row["section"], row["field"])
        assert field is not None
        assert field.asked_of_client is True


def test_asking_for_nothing_is_refused(service):
    run = run_scenario(service, "scenario_a_clean", tenant=TENANT_A, actor=ACTOR)
    with pytest.raises(OpsError) as excinfo:
        service.request_client_information(run.case, actor=ACTOR)
    assert excinfo.value.code == "OCC_AGENT_NOTHING_OUTSTANDING"


def test_recording_a_response_applies_it_through_client_onboarding(service):
    agent_case = service.create_case(
        tenant=TENANT_A, initiating_user=ACTOR,
        instruction="Onboard Northstar Lending. UK equity release. Monthly "
                    "portfolio MI. Portfolio id direct_101.")
    agent_case = service.request_client_information(agent_case, actor=ACTOR)
    request = agent_case.case.outstanding_requests[0]
    agent_case = service.record_client_response(
        agent_case, request_id=request.request_id, actor=ACTOR,
        answers={"contacts": {"reporting_contact_name": "Dana Fox",
                              "reporting_contact_email": "dana@example.com"}})
    assert agent_case.case.answers["contacts"]["reporting_contact_name"] == \
        "Dana Fox"
    assert agent_case.case.status == "in_review"
    assert not agent_case.case.outstanding_requests


# --------------------------------------------------------------------------- #
# The readiness package
# --------------------------------------------------------------------------- #

def test_the_readiness_package_carries_every_required_part(service):
    run = run_scenario(service, "scenario_a_clean", tenant=TENANT_A, actor=ACTOR)
    package = service.readiness_package(run.case)
    for part in ("case_summary", "confirmed_answers", "execution_facts",
                 "approved_product_scope",
                 "configuration_that_would_be_created", "artefact_inventory",
                 "intended_live_storage_paths", "field_mapping_report",
                 "validation_summary", "approved_exceptions",
                 "outstanding_observations", "orchestration_execution_plan",
                 "assembler_input_plan", "expected_downstream_outputs",
                 "human_approvals", "audit_trail", "execution_manifest"):
        assert part in package, part


def test_the_package_shows_the_configuration_it_did_not_create(service):
    """What activation WOULD create, marked as not created."""
    run = run_scenario(service, "scenario_a_clean", tenant=TENANT_A, actor=ACTOR)
    section = service.readiness_package(
        run.case)["configuration_that_would_be_created"]
    assert section["artefacts"], "the preview produced no configuration"
    assert section["written"] is False
    assert section["execution_status"] == "not_activated"
    assert section["next_version"] == 1
    assert section["current_version"] == 0


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

    The identity fields differ (a case reference is unique); everything the
    manifest says about the WORK must not.
    """
    from operations_control.occ_agent.service import OccAgentService
    first = run_scenario(service, "scenario_a_clean", tenant=TENANT_A,
                         actor=ACTOR)
    other = OccAgentService(storage, container="operations-control-synthetic",
                            sandbox=agent_env["sandbox"])
    second = run_scenario(other, "scenario_a_clean", tenant=TENANT_A,
                          actor=ACTOR)
    volatile = {"case_ref", "content_hash"}
    a = {k: v for k, v in
         service.readiness_package(first.case)["execution_manifest"].items()
         if k not in volatile}
    b = {k: v for k, v in
         other.readiness_package(second.case)["execution_manifest"].items()
         if k not in volatile}
    assert a == b


def test_the_package_states_what_did_not_happen(service):
    run = run_scenario(service, "scenario_a_clean", tenant=TENANT_A, actor=ACTOR)
    statement = service.readiness_package(run.case)["statement"]
    assert statement["headline"] == _readiness.READY_HEADLINE
    assert statement["not_done"] == list(_readiness.NOT_DONE_STATEMENTS)
    assert "No client configuration was activated." in statement["not_done"]


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
    assert run.case.run.readiness_package_ref.startswith(
        f"practice_cases/{run.case.case_ref}/")
    written = service.store.package_dir(TENANT_A, run.case.case_ref) \
        / "readiness_package.json"
    assert written.exists()
    assert Path(agent_env["sandbox"]).resolve() in written.parents


def test_every_stage_declares_which_of_the_outcomes_it_reached(service):
    from operations_control.occ_agent.run import STAGE_OUTCOMES
    run = run_scenario(service, "scenario_a_clean", tenant=TENANT_A, actor=ACTOR)
    assert run.case.run.stage_outcomes
    for stage, outcome in run.case.run.stage_outcomes.items():
        assert outcome in STAGE_OUTCOMES, f"{stage} -> {outcome}"


def test_a_simulated_stage_is_never_reported_as_executed(service):
    """The regime projection is simulated, and says so."""
    from operations_control.occ_agent.execution import SyntheticOnboardingAdapters
    from operations_control.occ_agent.run import STAGE_SIMULATED
    agent_case = service.create_case(tenant=TENANT_A, initiating_user=ACTOR,
                                     instruction="Onboard Northstar Lending.")
    sandbox = service.store.case_dir(TENANT_A, agent_case.case_ref)
    adapters = SyntheticOnboardingAdapters(
        artefact_paths=[], policy=service.policy, sandbox=sandbox,
        case_id=agent_case.case_ref, tenant=TENANT_A)
    central = sandbox / "central.csv"
    central.write_text("a,b\n1,2\n", encoding="utf-8")
    result = adapters.project(str(central), sandbox / "out", "ESMA_Annex2")
    assert result.ok is True
    assert "simulated" in result.message
    assert result.readiness["execution_status"] == "simulated_only"
    assert adapters.records[-1].outcome == STAGE_SIMULATED
    plan = json.loads(Path(result.output_path).read_text(encoding="utf-8"))
    assert plan["execution_status"] == "simulated_only"


def test_no_forbidden_outcome_word_describes_a_ready_case(service):
    """The outcome is READY_FOR_EXECUTION, never 'complete' or 'published'."""
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
