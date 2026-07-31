"""Reuse of the platform's existing components.

The OCC Agent is worth having only if it exercises the real Client Onboarding
capability, the real onboarding configuration, the real agents and the real
controls. These tests assert that each of them is genuinely on the path, rather
than approximated — mostly by spying on the real component and checking it was
called with a real contract.
"""

from __future__ import annotations

from pathlib import Path

from operations_control.occ_agent import derive as _derive
from operations_control.occ_agent import execution as _execution
from operations_control.occ_agent import input_roles as _roles
from operations_control.occ_agent.scenarios import run_scenario

from .conftest import ACTOR, TENANT_A


# --------------------------------------------------------------------------- #
# Client Onboarding
# --------------------------------------------------------------------------- #

def test_the_onboarding_case_is_client_onboardings_own(service):
    """Not a lookalike: the platform's model, reference series and store."""
    from operations_control.onboarding.case import CASE_ID_RE, OnboardingCase
    agent_case = service.create_case(tenant=TENANT_A, initiating_user=ACTOR,
                                     instruction="Onboard Northstar Lending.")
    assert isinstance(agent_case.case, OnboardingCase)
    assert CASE_ID_RE.match(agent_case.case_ref)
    # And it is loadable through Client Onboarding, not only through here.
    assert service.onboarding.load_case(agent_case.case_ref).case_id == \
        agent_case.case_ref


def test_the_real_onboarding_service_writes_every_answer(service, monkeypatch):
    from operations_control.onboarding.service import OnboardingService
    calls = []
    real = OnboardingService.save_step

    def spy(self, *, case_id, step, payload, by):
        calls.append(step)
        return real(self, case_id=case_id, step=step, payload=payload, by=by)

    monkeypatch.setattr(OnboardingService, "save_step", spy)
    service.create_case(
        tenant=TENANT_A, initiating_user=ACTOR,
        instruction="Onboard Northstar Lending. UK equity release. Monthly "
                    "portfolio MI. Portfolio id direct_101.")
    assert "client" in calls
    assert "portfolios" in calls
    assert "reporting" in calls


def test_the_real_validator_decides_whether_the_case_is_ready(service,
                                                              monkeypatch):
    from operations_control.onboarding.validation import Validator
    calls = []
    real = Validator.validate

    def spy(self, case):
        calls.append(case.case_id)
        return real(self, case)

    monkeypatch.setattr(Validator, "validate", spy)
    run_scenario(service, "scenario_a_clean", tenant=TENANT_A, actor=ACTOR)
    assert calls, "Client Onboarding's own validator did not run"


def test_the_real_inference_answers_what_the_instruction_did_not(service,
                                                                 monkeypatch):
    from operations_control.onboarding import inference
    calls = []
    real = inference.apply

    def spy(case, cat=None, **kwargs):
        calls.append(case.case_id)
        return real(case, cat, **kwargs)

    monkeypatch.setattr(inference, "apply", spy)
    agent_case = service.create_case(
        tenant=TENANT_A, initiating_user=ACTOR,
        instruction="Onboard Northstar Lending. UK equity release.")
    assert calls, "Client Onboarding's own inference did not run"
    assert agent_case.case.answers["client"]["reporting_currency"] == "GBP"


def test_the_client_checklist_is_the_catalogues_own(service, monkeypatch):
    from operations_control.onboarding.catalogue import Catalogue
    calls = []
    real = Catalogue.outstanding_for_client

    def spy(self, answers):
        calls.append(len(answers))
        return real(self, answers)

    monkeypatch.setattr(Catalogue, "outstanding_for_client", spy)
    run_scenario(service, "scenario_a_clean", tenant=TENANT_A, actor=ACTOR)
    assert calls, "the catalogue's own outstanding-for-client list was not used"


def test_the_configuration_preview_is_client_onboardings_own(service,
                                                             monkeypatch):
    from operations_control.onboarding import artefacts as _onboarding_artefacts
    calls = []
    real = _onboarding_artefacts.plan

    def spy(case, **kwargs):
        calls.append(case.case_id)
        return real(case, **kwargs)

    monkeypatch.setattr(_onboarding_artefacts, "plan", spy)
    run = run_scenario(service, "scenario_a_clean", tenant=TENANT_A, actor=ACTOR)
    assert calls, "the real artefact planner did not produce the preview"
    preview = service.preview(run.case)
    assert preview["artefacts"]
    # It planned; it did not write.
    assert service.onboarding.cases.current(run.case.case.client_id) is None


def test_the_agent_never_calls_activation(service, monkeypatch):
    """The one method the whole feature must not reach."""
    from operations_control.onboarding.service import OnboardingService
    called = []

    def spy(self, *, case_id, by):
        called.append(case_id)
        raise AssertionError("activation was reached from a practice case")

    monkeypatch.setattr(OnboardingService, "activate", spy)
    for fixture in ("scenario_a_clean", "scenario_b_ambiguous_mapping",
                    "scenario_c_missing_artefact",
                    "scenario_d_product_information_gap",
                    "scenario_e_business_rule_failure"):
        run_scenario(service, fixture, tenant=TENANT_A, actor=ACTOR)
    assert called == []


# --------------------------------------------------------------------------- #
# The governed input requirements
# --------------------------------------------------------------------------- #

def test_required_source_files_come_from_the_governed_input_requirements():
    """The run asks for what workflow_input_requirements.yaml requires."""
    import yaml
    doc = yaml.safe_load(
        Path("config/system/workflow_input_requirements.yaml").read_text(
            encoding="utf-8"))
    configured = doc["workflows"]["mi"]["required_roles"]
    assert _roles.artefact_vocabulary().required_roles("mi") == configured


def test_a_configuration_change_changes_what_the_run_asks_for(tmp_path):
    """Prove it is read from configuration, not written into the code."""
    altered = tmp_path / "requirements.yaml"
    altered.write_text(
        "version: 1\n"
        "role_labels:\n"
        "  cashflow_extract: 'Cash-flow tape'\n"
        "workflows:\n"
        "  mi:\n"
        "    required_roles: [cashflow_extract]\n"
        "    optional_roles: []\n"
        "min_required_role_confidence: 0.4\n", encoding="utf-8")
    try:
        vocab = _roles.artefact_vocabulary(altered)
        assert vocab.required_roles("mi") == ["cashflow_extract"]
    finally:
        _roles.reset_cache()


def test_the_delivery_outcome_comes_from_the_product_declaration():
    """Nothing here decides what a product implies — the declaration does."""
    from operations_control.onboarding.catalogue import catalogue
    cat = catalogue()
    for product, spec in (cat.regime_products or {}).items():
        needs = str(spec.get("supported_by_assets_declaring") or "")
        outcome = _derive.outcome_for([product], "equity_release", cat)
        assert outcome == ("mi_annex2" if needs == "ESMA_Annex2" else "mi"), \
            product


def test_a_product_an_asset_cannot_support_contributes_no_regime():
    from operations_control.onboarding.catalogue import catalogue
    cat = catalogue()
    # equity_release declares ESMA_Annex2 and nothing else.
    assert _derive.regime_for(["esma_annex2"], "equity_release", cat) == \
        "ESMA_Annex2"
    assert _derive.regime_for(["esma_annex2"], "some_other_asset", cat) == ""


def test_asset_recognition_uses_the_product_profile_signal_tokens():
    """An asset is recognised from the profile's own vocabulary."""
    from operations_control.occ_agent.interpretation import (
        DeterministicInterpreter,
    )
    interpreter = DeterministicInterpreter()
    for phrase in ("a lifetime mortgage book", "uk equity-release portfolio",
                   "roll-up lending"):
        asset, confidence = interpreter.match_asset(phrase)
        assert asset == "equity_release", phrase
        assert confidence > 0


def test_delivery_locations_use_the_production_path_rules(service):
    """The location recorded against a file is one the live parser accepts."""
    from apps.blob_trigger_app.path_parser import parse_blob_path
    from operations_control.manual_intake import raw_container

    run = run_scenario(service, "scenario_a_clean", tenant=TENANT_A, actor=ACTOR)
    artefacts = run.case.run.received_artefacts
    assert artefacts
    for artefact in artefacts:
        uri = artefact["intended_live_uri"]
        assert uri.startswith("blob://")
        assert parse_blob_path(uri[len("blob://"):], raw_container()) is not None


def test_the_expected_delivery_location_is_client_onboardings_own(service):
    """Client Onboarding quotes the location; the OCC Agent does not restate it."""
    run = run_scenario(service, "scenario_a_clean", tenant=TENANT_A, actor=ACTOR)
    sources = run.case.case.items("sources")
    assert sources
    assert all(s["expected_location"].startswith("raw-v2/") for s in sources)


# --------------------------------------------------------------------------- #
# The Onboarding Agent's own components
# --------------------------------------------------------------------------- #

def test_the_real_header_mapper_and_profiler_are_used(service, monkeypatch):
    calls = {"mapper": 0, "profile": 0}

    import engine.gate_1_alignment.semantic_alignment as sa
    import engine.onboarding_agent.file_profiler as fp

    real_map_one = sa.HeaderMapper.map_one
    real_profile = fp.profile_file

    def spy_map_one(self, header):
        calls["mapper"] += 1
        return real_map_one(self, header)

    def spy_profile(path):
        calls["profile"] += 1
        return real_profile(path)

    monkeypatch.setattr(sa.HeaderMapper, "map_one", spy_map_one)
    monkeypatch.setattr(fp, "profile_file", spy_profile)

    run = run_scenario(service, "scenario_a_clean", tenant=TENANT_A, actor=ACTOR)
    assert run.case.run.state == "READY_FOR_EXECUTION"
    assert calls["mapper"] > 10, "the real header mapper did not run"
    assert calls["profile"] >= 1, "the real source profiler did not run"


def test_the_real_canonical_transform_is_used(service, monkeypatch):
    import engine.gate_2_transform.canonical_transform as ct
    calls = []
    real = ct.apply_types

    def spy(df, fields_meta, *args, **kwargs):
        calls.append(len(fields_meta))
        return real(df, fields_meta, *args, **kwargs)

    monkeypatch.setattr(ct, "apply_types", spy)
    run_scenario(service, "scenario_a_clean", tenant=TENANT_A, actor=ACTOR)
    assert calls, "the real canonical transform did not run"
    assert calls[0] > 50, "the transform was not given the real field registry"


def test_the_real_business_rules_and_materiality_are_used(service, monkeypatch):
    import engine.gate_3_validation.aggregate_validation_results as agg
    import engine.gate_3_validation.validate_business_rules as vbr

    seen = {"rules": 0, "materiality": 0}
    real_rules = vbr.run_rules
    real_materiality = agg.determine_materiality

    def spy_rules(df, regime):
        seen["rules"] += 1
        return real_rules(df, regime)

    def spy_materiality(*args, **kwargs):
        seen["materiality"] += 1
        return real_materiality(*args, **kwargs)

    monkeypatch.setattr(vbr, "run_rules", spy_rules)
    monkeypatch.setattr(agg, "determine_materiality", spy_materiality)

    # Scenario E is the one with findings, so materiality is genuinely applied.
    run = run_scenario(service, "scenario_e_business_rule_failure",
                       tenant=TENANT_A, actor=ACTOR)
    assert seen["rules"] >= 1, "the real business-rule engine did not run"
    assert seen["materiality"] >= 1, "the real materiality logic did not run"
    assert run.case.run.state == "BLOCKED"


def test_materiality_comes_from_the_issue_policy(service):
    """The blocking verdict carries the policy's own materiality vocabulary."""
    run = run_scenario(service, "scenario_e_business_rule_failure",
                       tenant=TENANT_A, actor=ACTOR)
    findings = [r for r in run.case.run.control_results
                if r.get("kind") == "validation"]
    assert findings
    rows = findings[-1]["findings"]
    assert any(str(r["materiality"]).upper() == "BLOCKING" for r in rows)
    # And each finding carries the classification the policy assigned.
    assert all(r.get("classification") for r in rows)


# --------------------------------------------------------------------------- #
# The Orchestration Agent
# --------------------------------------------------------------------------- #

def test_the_real_orchestration_conductor_drives_the_run(service, monkeypatch):
    import engine.orchestrator_agent.orchestrator as orch
    calls = []
    real = orch.run_orchestration

    def spy(*args, **kwargs):
        calls.append(kwargs.get("target"))
        return real(*args, **kwargs)

    monkeypatch.setattr(orch, "run_orchestration", spy)
    run_scenario(service, "scenario_a_clean", tenant=TENANT_A, actor=ACTOR)
    assert calls, "the orchestration conductor was not used"


def test_the_execution_adapter_subclasses_the_real_agent_seam():
    from engine.orchestrator_agent.adapters import AgentAdapters
    assert issubclass(_execution.SyntheticOnboardingAdapters, AgentAdapters)


def test_the_orchestration_plan_uses_the_conductors_own_step_sequence(service):
    from engine.orchestrator_agent.orchestrator import steps_for_target
    run = run_scenario(service, "scenario_a_clean", tenant=TENANT_A, actor=ACTOR)
    plan = run.case.run.orchestration_plan
    expected = list(steps_for_target(plan["target"], full_pipeline=True))
    assert [s["step"] for s in plan["steps"]][: len(expected)] == expected
    assert plan["execution_status"] == "not_executed"


def test_the_conductors_gate_sequence_is_actually_walked(service):
    run = run_scenario(service, "scenario_a_clean", tenant=TENANT_A, actor=ACTOR)
    for step in ("onboard", "transform", "validate", "stamp", "assemble"):
        assert step in run.case.run.stage_outcomes, step


# --------------------------------------------------------------------------- #
# The Assembler Agent
# --------------------------------------------------------------------------- #

def test_the_real_assembler_agent_runs(service, monkeypatch):
    import engine.assembler_agent as aa
    calls = []
    real = aa.run_assembler_agent

    def spy(paths, out_dir, **kwargs):
        calls.append((list(paths), kwargs.get("pipeline")))
        return real(paths, out_dir, **kwargs)

    monkeypatch.setattr(aa, "run_assembler_agent", spy)
    run = run_scenario(service, "scenario_a_clean", tenant=TENANT_A, actor=ACTOR)
    assert calls, "the real Assembler Agent did not run"
    assert run.case.run.stage_outcomes["assemble"] == \
        "deterministic_execution_completed"


def test_assembler_prerequisites_come_from_the_real_assembler(service):
    from engine.platform_assembler import LOAN_KEY_FIELDS
    run = run_scenario(service, "scenario_a_clean", tenant=TENANT_A, actor=ACTOR)
    plan = run.case.run.assembler_plan
    assert plan["satisfied"] is True
    assert plan["source"] == "engine.platform_assembler"
    assert any(field in " ".join(plan["prerequisites"])
               for field in LOAN_KEY_FIELDS)


def test_provenance_stamping_is_the_real_one(service, monkeypatch):
    import engine.provenance as prov
    calls = []
    real = prov.build_provenance

    def spy(*args, **kwargs):
        calls.append(kwargs.get("source_portfolio_id") or args[0])
        return real(*args, **kwargs)

    monkeypatch.setattr(prov, "build_provenance", spy)
    run_scenario(service, "scenario_a_clean", tenant=TENANT_A, actor=ACTOR)
    assert calls, "the real provenance builder did not run"


# --------------------------------------------------------------------------- #
# The OCC's own contracts
# --------------------------------------------------------------------------- #

def test_pending_decisions_are_read_by_the_existing_occ_extractor(service):
    """A practice halt produces the artefact the live OCC already reads."""
    from operations_control.adapters import DECISIONS_FILE
    assert _execution.DECISIONS_FILE == DECISIONS_FILE

    run = run_scenario(service, "scenario_b_ambiguous_mapping",
                       tenant=TENANT_A, actor=ACTOR, resolve_decisions=False)
    decisions = [d for d in run.case.run.open_decisions
                 if d["status"] == "open"]
    assert decisions, "the ambiguous mapping did not raise a decision"
    card = decisions[0]
    # The card carries the OCC's own decision vocabulary.
    from operations_control.contracts import DECISION_KINDS
    assert card["kind"] in DECISION_KINDS
    assert card["blocking"] is True
