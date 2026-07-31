"""Reuse of the platform's existing components.

The OCC Agent is worth having only if it exercises the real onboarding
configuration, the real agents and the real controls. These tests assert that
each of them is genuinely on the path, rather than approximated — mostly by
spying on the real component and checking it was called with a real contract.
"""

from __future__ import annotations

from pathlib import Path

from operations_control.occ_agent import execution as _execution
from operations_control.occ_agent import pack as _pack
from operations_control.occ_agent import vocabulary as _vocab
from operations_control.occ_agent.scenarios import run_scenario

from .conftest import ACTOR, TENANT_A


# --------------------------------------------------------------------------- #
# The onboarding configuration framework
# --------------------------------------------------------------------------- #

def test_required_source_files_come_from_the_governed_input_requirements():
    """The pack asks for what workflow_input_requirements.yaml requires."""
    import yaml
    doc = yaml.safe_load(
        Path("config/system/workflow_input_requirements.yaml").read_text(
            encoding="utf-8"))
    configured = doc["workflows"]["mi"]["required_roles"]
    vocab = _vocab.artefact_vocabulary()
    assert vocab.required_roles("mi") == configured


def test_a_configuration_change_changes_what_the_pack_asks_for(tmp_path):
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
        vocab = _vocab.artefact_vocabulary(altered)
        assert vocab.required_roles("mi") == ["cashflow_extract"]
    finally:
        _vocab.reset_cache()


def test_asset_types_come_from_the_platform_asset_model():
    from operations_control.configuration.packages import ASSET_MODEL
    vocab = _vocab.asset_vocabulary()
    assert vocab.keys() == sorted(ASSET_MODEL)
    for key, spec in ASSET_MODEL.items():
        assert vocab.label(key) == spec["label"]
        assert list(vocab.supports_regimes(key)) == spec["supports_regimes"]


def test_products_come_from_the_configured_capability_vocabulary():
    import yaml
    capabilities = yaml.safe_load(
        Path("config/asset/product_profiles.yaml").read_text(
            encoding="utf-8"))["capabilities"]
    products = _vocab.product_vocabulary()
    for product in products.products:
        assert product.capability in capabilities


def test_asset_recognition_uses_the_product_profile_signal_tokens():
    """An asset is recognised from the profile's own vocabulary."""
    vocab = _vocab.asset_vocabulary()
    for phrase in ("a lifetime mortgage book", "uk equity-release portfolio",
                   "roll-up lending"):
        asset, confidence = vocab.match(phrase)
        assert asset == "equity_release", phrase
        assert confidence > 0


def test_delivery_instructions_use_the_production_path_rules(service):
    """The location quoted to the client is one the live parser accepts."""
    from apps.blob_trigger_app.path_parser import parse_blob_path
    from operations_control.manual_intake import raw_container

    run = run_scenario(service, "scenario_a_clean", tenant=TENANT_A, actor=ACTOR)
    section = next(s for s in run.case.onboarding_pack["sections"]
                   if s["key"] == _pack.SEC_DELIVERY)
    assert "blob://" in section["body"]
    prefix = section["body"].split("blob://")[1].split("/ ")[0].rstrip("/")
    # The production parser is the authority; it must accept the quoted path.
    parsed = parse_blob_path(f"{prefix}/probe.csv", raw_container())
    assert parsed is not None


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
    assert run.case.state == "READY_FOR_EXECUTION"
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
    assert run.case.state == "BLOCKED"


def test_materiality_comes_from_the_issue_policy(service):
    """The blocking verdict carries the policy's own materiality vocabulary."""
    run = run_scenario(service, "scenario_e_business_rule_failure",
                       tenant=TENANT_A, actor=ACTOR)
    findings = [r for r in run.case.control_results if r.get("kind") == "validation"]
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
    monkeypatch.setattr(_execution, "run_synthetic_orchestration",
                        _execution.run_synthetic_orchestration)
    run_scenario(service, "scenario_a_clean", tenant=TENANT_A, actor=ACTOR)
    assert calls, "the orchestration conductor was not used"


def test_the_execution_adapter_subclasses_the_real_agent_seam():
    from engine.orchestrator_agent.adapters import AgentAdapters
    assert issubclass(_execution.SyntheticOnboardingAdapters, AgentAdapters)


def test_the_orchestration_plan_uses_the_conductors_own_step_sequence(service):
    from engine.orchestrator_agent.orchestrator import steps_for_target
    run = run_scenario(service, "scenario_a_clean", tenant=TENANT_A, actor=ACTOR)
    plan = run.case.orchestration_plan
    expected = list(steps_for_target(plan["target"], full_pipeline=True))
    assert [s["step"] for s in plan["steps"]][: len(expected)] == expected
    assert plan["execution_status"] == "not_executed"


def test_the_conductors_gate_sequence_is_actually_walked(service):
    run = run_scenario(service, "scenario_a_clean", tenant=TENANT_A, actor=ACTOR)
    for step in ("onboard", "transform", "validate", "stamp", "assemble"):
        assert step in run.case.stage_outcomes, step


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
    assert run.case.stage_outcomes["assemble"] == "deterministic_execution_completed"


def test_assembler_prerequisites_come_from_the_real_assembler(service):
    from engine.platform_assembler import LOAN_KEY_FIELDS
    run = run_scenario(service, "scenario_a_clean", tenant=TENANT_A, actor=ACTOR)
    plan = run.case.assembler_plan
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
    """A synthetic halt produces the artefact the live OCC already reads."""
    from operations_control.adapters import DECISIONS_FILE
    assert _execution.DECISIONS_FILE == DECISIONS_FILE

    run = run_scenario(service, "scenario_b_ambiguous_mapping",
                       tenant=TENANT_A, actor=ACTOR, resolve_decisions=False)
    decisions = [d for d in run.case.open_decisions if d["status"] == "open"]
    assert decisions, "the ambiguous mapping did not raise a decision"
    card = decisions[0]
    # The card carries the OCC's own decision vocabulary.
    from operations_control.contracts import DECISION_KINDS
    assert card["kind"] in DECISION_KINDS
    assert card["blocking"] is True


def test_the_configuration_resolver_is_the_platforms_own(service, monkeypatch):
    from operations_control.configuration import resolver as res
    calls = []
    real = res.EffectiveConfigResolver.resolve

    def spy(self, **kwargs):
        calls.append(kwargs.get("asset_type"))
        return real(self, **kwargs)

    monkeypatch.setattr(res.EffectiveConfigResolver, "resolve", spy)
    run_scenario(service, "scenario_a_clean", tenant=TENANT_A, actor=ACTOR)
    assert calls == ["equity_release"]


def test_the_effective_configuration_contract_is_produced(service):
    run = run_scenario(service, "scenario_a_clean", tenant=TENANT_A, actor=ACTOR)
    config = run.case.proposed_configuration
    assert config["effective_config_id"].startswith("ecfg_")
    assert config["effective_content_hash"].startswith("sha256:")
    # The precedence contract is the platform's, applied to real layers.
    assert config["value_provenance"]


def test_the_configuration_candidate_is_the_pipelines_own_shape(service):
    """Not a second format: the same document the pipeline already reads."""
    import yaml
    run = run_scenario(service, "scenario_a_clean", tenant=TENANT_A, actor=ACTOR)
    candidate = run.case.confirmed_configuration
    live = yaml.safe_load(
        Path("config/client/config_client_ERM_UK.yaml").read_text(
            encoding="utf-8"))
    # Every top-level section the candidate uses exists in the live config.
    for section in candidate:
        assert section in live, section
    assert candidate["client"]["client_id"]
    assert candidate["portfolio"]["asset_class"] in ("equity_release",)
