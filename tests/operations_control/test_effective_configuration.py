"""Effective configuration: resolver precedence/provenance/hashing/pinning,
admin package lifecycle, admin-only scopes, and the narrowed Onboarding Agent
facade contract."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import pytest
import yaml

from operations_control.configuration.contract import EffectiveConfiguration
from operations_control.configuration.packages import (
    ConfigPackageStore,
    LAYER_SYSTEM,
)
from operations_control.configuration.resolver import (
    BLOCKED,
    READY,
    READY_WITH_WARNINGS,
    EffectiveConfigResolver,
)
from operations_control.engine import OpsError
from operations_control.rules import RuleRecord, RuleStore

from .conftest import (
    onboard_client,
    StubAnnex2Stages,
    make_client_config,
    make_engine,
    register_and_create,
    start_and_wait,
    wait_for,
)


def _setting_rule(scope: str, setting: str, value: str, *,
                  client="client_a", portfolio="", run_ref="") -> RuleRecord:
    return RuleRecord(
        rule_id="", version=0, kind="client_rule", scope=scope,
        client_id=(client if scope not in ("global", "asset") else ""),
        portfolio_id=portfolio, run_ref=run_ref,
        payload={"setting": setting, "value": value},
        approved_by="alice", reason="test")


def make_resolver(store, tmp_path, *, valid=True, client="client_a"):
    """A resolver, and a client whose onboarding activated ``make_client_config``.

    The resolver has no configuration path of its own: it reads what a client's
    own onboarding activated, so a test that wants a client configuration has
    to give that client one.
    """
    rules = RuleStore(store)
    onboard_client(store, client, make_client_config(tmp_path, valid=valid))
    return EffectiveConfigResolver(store, rules), rules


@pytest.fixture()
def resolver(store, tmp_path):
    return make_resolver(store, tmp_path)


class TestPrecedenceAndProvenance:
    def test_layers_resolve_with_explicit_precedence(self, resolver):
        res, rules = resolver
        rules.approve(_setting_rule("client", "reporting_channel", "client-v"))
        rules.approve(_setting_rule("portfolio", "reporting_channel",
                                    "portfolio-v", portfolio="pf1"))
        out = res.resolve(client_id="client_a", portfolio_id="pf1",
                          workflow_id="wf_p", reporting_period="2026-06-30")
        eff = out.effective
        assert eff.resolved_values["reporting_channel"] == "portfolio-v"
        assert eff.value_provenance["reporting_channel"] == "portfolio"
        # Portfolio overriding client is a recorded conflict + warning.
        assert any(c["key"] == "reporting_channel" for c in eff.conflicts)
        assert out.status == READY_WITH_WARNINGS

    def test_run_decision_overrides_portfolio(self, resolver):
        res, rules = resolver
        rules.approve(_setting_rule("portfolio", "cutoff_day", "25",
                                    portfolio="pf1"))
        rules.approve(_setting_rule("run", "cutoff_day", "28",
                                    portfolio="pf1", run_ref="wf_r"))
        out = res.resolve(client_id="client_a", portfolio_id="pf1",
                          workflow_id="wf_r", reporting_period="2026-06-30")
        assert out.effective.resolved_values["cutoff_day"] == "28"
        assert out.effective.value_provenance["cutoff_day"] == "run_decisions"

    def test_run_rules_do_not_leak_to_other_runs(self, resolver):
        res, rules = resolver
        rules.approve(_setting_rule("run", "cutoff_day", "28",
                                    run_ref="wf_other"))
        out = res.resolve(client_id="client_a", portfolio_id="pf1",
                          workflow_id="wf_mine", reporting_period="2026-06-30")
        assert out.effective.resolved_values.get("cutoff_day") != "28"

    def test_client_layer_overrides_asset_defaults(self, resolver):
        res, _ = resolver
        out = res.resolve(client_id="client_a", portfolio_id="pf1",
                          workflow_id="wf_a", reporting_period="2026-06-30")
        eff = out.effective
        # Asset defaults are present with asset provenance; client YAML keys
        # carry client provenance.
        assert any(v == "asset" for v in eff.value_provenance.values())
        assert any(v == "client" for v in eff.value_provenance.values())

    def test_override_records_carry_approver_and_reason(self, resolver):
        res, rules = resolver
        rules.approve(_setting_rule("portfolio", "base_currency", "EUR",
                                    portfolio="pf1"))
        out = res.resolve(client_id="client_a", portfolio_id="pf1",
                          workflow_id="wf_o", reporting_period="2026-06-30")
        ov = [o for o in out.effective.overrides
              if o["key"] == "base_currency"]
        assert ov and ov[0]["approved_by"] == "alice"
        assert ov[0]["original_value"] == "GBP"
        assert ov[0]["overridden_value"] == "EUR"


class TestOutcomesHashingPinning:
    def test_missing_client_value_blocks_annex2_with_safe_message(
            self, store, tmp_path):
        res, _ = make_resolver(store, tmp_path, valid=False)
        out = res.resolve(client_id="client_a", portfolio_id="pf1",
                          outcome="mi_annex2", workflow_id="wf_b",
                          reporting_period="2026-06-30")
        assert out.status == BLOCKED
        from operations_control import language
        assert out.blockers and all(language.is_operator_safe(b)
                                    for b in out.blockers)
        # The same gap does NOT block an MI-only run.
        out_mi = res.resolve(client_id="client_a", portfolio_id="pf1",
                             outcome="mi", workflow_id="wf_b2",
                             reporting_period="2026-06-30")
        assert out_mi.status in (READY, READY_WITH_WARNINGS)

    def test_content_hash_is_stable_and_decision_sensitive(self, resolver):
        res, rules = resolver
        a = res.resolve(client_id="client_a", portfolio_id="pf1",
                        workflow_id="", reporting_period="2026-06-30")
        b = res.resolve(client_id="client_a", portfolio_id="pf1",
                        workflow_id="", reporting_period="2026-06-30")
        assert a.effective.content_hash == b.effective.content_hash
        rules.approve(_setting_rule("client", "reporting_channel", "x"))
        c = res.resolve(client_id="client_a", portfolio_id="pf1",
                        workflow_id="", reporting_period="2026-06-30")
        assert c.effective.content_hash != a.effective.content_hash
        assert c.effective.decision_set_version != \
            a.effective.decision_set_version

    def test_immutable_and_reconstructable(self, resolver):
        res, _ = resolver
        out = res.resolve(client_id="client_a", portfolio_id="pf1",
                          workflow_id="wf_pin", reporting_period="2026-06-30",
                          version=1)
        with pytest.raises(Exception):
            out.effective.client_id = "tampered"          # frozen dataclass
        loaded = res.load("client_a", "wf_pin")
        assert loaded is not None
        assert loaded.content_hash == out.effective.content_hash
        assert loaded.effective_config_id == out.effective.effective_config_id

    def test_version_history_preserved(self, resolver):
        res, rules = resolver
        res.resolve(client_id="client_a", portfolio_id="pf1",
                    workflow_id="wf_v", reporting_period="2026-06-30",
                    version=1)
        rules.approve(_setting_rule("client", "reporting_channel", "new"))
        res.resolve(client_id="client_a", portfolio_id="pf1",
                    workflow_id="wf_v", reporting_period="2026-06-30",
                    version=2)
        v1 = res.load("client_a", "wf_v", version=1)
        v2 = res.load("client_a", "wf_v", version=2)
        assert v1 is not None and v2 is not None
        assert v1.content_hash != v2.content_hash
        assert res.load("client_a", "wf_v").version == 2   # current pointer

    def test_tenant_isolation_of_rules_and_persistence(self, store, tmp_path,
                                                       ops_env):
        res, rules = make_resolver(store, tmp_path)
        rules.approve(_setting_rule("client", "reporting_channel", "secret-b",
                                    client="client_b"))
        out = res.resolve(client_id="client_a", portfolio_id="pf1",
                          workflow_id="wf_t", reporting_period="2026-06-30")
        assert out.effective.resolved_values.get("reporting_channel") != "secret-b"
        # Persisted under client_a's prefix only.
        assert (ops_env["blob_root"] / "operations-control" / "client_a"
                / "workflow-runs" / "wf_t" / "effective-config").exists()
        assert not (ops_env["blob_root"] / "operations-control" / "client_b"
                    / "workflow-runs" / "wf_t").exists()


class TestConfigPackages:
    def test_seed_draft_validate_activate_rollback(self, store):
        pkgs = ConfigPackageStore(store)
        v1 = pkgs.ensure_seeded(LAYER_SYSTEM, by="admin")
        assert v1["version"] == 1 and v1["status"] == "active"

        draft = pkgs.create_draft(
            LAYER_SYSTEM, by="admin", notes="tweak",
            edits={"config/system/run_context_policy.yaml": "version: 2\n"})
        assert draft["status"] == "draft"
        # Active version untouched by drafting.
        assert pkgs.active_version(LAYER_SYSTEM)["version"] == 1

        with pytest.raises(ValueError):
            pkgs.activate_version(LAYER_SYSTEM, draft["version"], by="admin")
        pkgs.validate_version(LAYER_SYSTEM, draft["version"])
        pkgs.activate_version(LAYER_SYSTEM, draft["version"], by="admin")
        assert pkgs.active_version(LAYER_SYSTEM)["version"] == draft["version"]
        assert pkgs.get_version(LAYER_SYSTEM, 1)["status"] == "superseded"

        pkgs.rollback(LAYER_SYSTEM, 1, by="admin")
        assert pkgs.active_version(LAYER_SYSTEM)["version"] == 1

    def test_invalid_yaml_draft_fails_validation(self, store):
        pkgs = ConfigPackageStore(store)
        pkgs.ensure_seeded(LAYER_SYSTEM, by="admin")
        draft = pkgs.create_draft(
            LAYER_SYSTEM, by="admin",
            edits={"config/system/run_context_policy.yaml":
                   "not: [valid: yaml"})
        doc = pkgs.validate_version(LAYER_SYSTEM, draft["version"])
        assert doc["validation"]["ok"] is False
        with pytest.raises(ValueError):
            pkgs.activate_version(LAYER_SYSTEM, draft["version"], by="admin")

    def test_existing_run_stays_pinned_after_activation(
            self, store, source_registry, delivery_dir, tmp_path):
        engine = make_engine(store, source_registry)
        run = register_and_create(engine, delivery_dir, outcome="mi_annex2")
        final = start_and_wait(engine, run)
        pinned = final.effective_config["package_versions"]["system"]

        pkgs = ConfigPackageStore(store)
        draft = pkgs.create_draft(LAYER_SYSTEM, by="admin",
                                  edits={"config/system/run_context_policy.yaml":
                                         "version: 3\n"})
        pkgs.validate_version(LAYER_SYSTEM, draft["version"])
        pkgs.activate_version(LAYER_SYSTEM, draft["version"], by="admin")

        # The completed run still reports the version it resolved with.
        again = store.load_workflow("client_a", run.workflow_id)
        assert again.effective_config["package_versions"]["system"] == pinned
        res = EffectiveConfigResolver(store, RuleStore(store))
        loaded = res.load("client_a", run.workflow_id)
        assert loaded.system_config_version == pinned


class TestAdminAuthorisation:
    def test_operator_cannot_use_admin_area_or_scopes(
            self, store, source_registry, delivery_dir):
        from fastapi.testclient import TestClient
        from operations_control.api import app as app_module
        from .conftest import OP_A
        engine = make_engine(store, source_registry, "mapping_halt")
        run = register_and_create(engine, delivery_dir)
        start_and_wait(engine, run)
        app_module.set_engine(engine)
        client = TestClient(app_module.app, raise_server_exceptions=False)
        try:
            r = client.get("/ops/admin/config",
                           headers={"X-Operator-Token": OP_A})
            assert r.status_code == 403
            dec = store.open_decisions("client_a", run.workflow_id)[0]
            r = client.post(f"/ops/reviews/{dec['decision_id']}/decision",
                            headers={"X-Operator-Token": OP_A},
                            json={"action": "approve",
                                  "value": "accept_mapping",
                                  "scope": "global"})
            assert r.status_code == 403
            assert store.load_decision(
                "client_a", dec["decision_id"])["status"] == "open"
        finally:
            app_module.set_engine(None)

    def test_admin_scope_allowed_via_engine_flag(self, store, source_registry,
                                                 delivery_dir):
        engine = make_engine(store, source_registry, "mapping_halt")
        run = register_and_create(engine, delivery_dir)
        start_and_wait(engine, run)
        dec = store.open_decisions("client_a", run.workflow_id)[0]
        with pytest.raises(OpsError) as e:
            engine.resolve_decision(client_id="client_a",
                                    decision_id=dec["decision_id"],
                                    action="approve", actor="alice",
                                    value="accept_mapping", scope="global")
        assert e.value.code == "OPS_ADMIN_SCOPE"
        out = engine.resolve_decision(client_id="client_a",
                                      decision_id=dec["decision_id"],
                                      action="approve", actor="root",
                                      value="accept_mapping", scope="global",
                                      actor_is_admin=True)
        assert out["rule"]["scope"] == "global"


class TestNarrowedAgent:
    """Facade contract tests with the heavy workflow stubbed — proves the
    interface, snapshot verification, config non-resolution and decision
    extraction without running the full agent."""

    @pytest.fixture()
    def stub_workflow(self, monkeypatch, tmp_path):
        calls = {}

        def fake_workflow(**kwargs):
            calls.update(kwargs)
            out = Path(kwargs["project_dir"])
            (out / "40_operator_workflow_summary.json").write_text(json.dumps(
                {"status": "NEEDS_CONFIRMATION", "blocking_decisions_count": 1}))
            (out / "34_target_first_decisions.yaml").write_text(yaml.safe_dump(
                {"decisions": [{"decision_id": "DQ-1",
                                "decision_type": "missing_required_target",
                                "target_field": "RREL2", "blocking": True,
                                "operator_question": "Provide RREL2.",
                                "options": ["map_source_column"],
                                "status": "pending"}]}))
            (out / "18_central_lender_tape.csv").write_text("loan_id\nL1\n")
            return {"status": "NEEDS_CONFIRMATION"}

        from engine.onboarding_agent import workflow as wf
        monkeypatch.setattr(wf, "run_operator_workflow", fake_workflow)
        return calls

    def _effective_and_snapshot(self, store, tmp_path):
        res, _ = make_resolver(store, tmp_path)
        out = res.resolve(client_id="client_a", portfolio_id="pf1",
                          workflow_id="", reporting_period="2026-06-30")
        snap = res.materialise_snapshot(out.effective, tmp_path / "snap")
        return out.effective, snap

    def test_consumes_snapshot_not_live_config(self, store, tmp_path,
                                               stub_workflow, delivery_dir):
        from engine.onboarding_agent.narrowed import (
            ExecutionContext, run_onboarding)
        eff, snap = self._effective_and_snapshot(store, tmp_path)
        ctx = ExecutionContext(run_id="r1", client_id="client_a",
                               tenant_id="client_a",
                               output_root=str(tmp_path / "run1"),
                               config_snapshot=snap,
                               reporting_period="2026-06-30")
        result = run_onboarding(str(delivery_dir), eff, ctx)
        # The agent received the SNAPSHOT registry, not the repo file.
        assert stub_workflow["registry"] == snap["registry"]
        assert stub_workflow["aliases_dir"] == snap["aliases_dir"]
        assert Path(snap["registry"]).is_relative_to(tmp_path)
        assert result.status == "needs_review"
        assert result.effective_config_hash == eff.content_hash
        assert result.canonical_output_reference.endswith(
            "18_central_lender_tape.csv")
        assert result.blocking_decisions
        d = result.blocking_decisions[0]
        assert d.category == "unmapped_source_field"
        assert d.plain_english_message

    def test_snapshot_mutation_is_refused(self, store, tmp_path,
                                          stub_workflow, delivery_dir):
        from engine.onboarding_agent.narrowed import (
            ExecutionContext, ImmutableConfigError, run_onboarding)
        eff, snap = self._effective_and_snapshot(store, tmp_path)
        reg = Path(snap["registry"])
        reg.write_text(reg.read_text() + "\n# mutated\n", encoding="utf-8")
        ctx = ExecutionContext(run_id="r2", client_id="client_a",
                               tenant_id="client_a",
                               output_root=str(tmp_path / "run2"),
                               config_snapshot=snap)
        with pytest.raises(ImmutableConfigError):
            run_onboarding(str(delivery_dir), eff, ctx)

    def test_legacy_wrapper_deprecates(self, tmp_path, stub_workflow,
                                       delivery_dir):
        from engine.onboarding_agent.narrowed import run_onboarding_legacy
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = run_onboarding_legacy(
                input_dir=str(delivery_dir), client_id="client_a",
                output_root=str(tmp_path / "legacy"))
        assert any(issubclass(w.category, DeprecationWarning) for w in caught)
        assert result.run_id == "run"

    def test_no_writes_outside_output_root(self, store, tmp_path,
                                           stub_workflow, delivery_dir):
        """The facade itself writes only under the run's output root (the
        stubbed workflow writes its artefacts there too)."""
        from engine.onboarding_agent.narrowed import (
            ExecutionContext, run_onboarding)
        eff, snap = self._effective_and_snapshot(store, tmp_path)
        before = {p: p.stat().st_mtime_ns
                  for p in Path("config").rglob("*.yaml")}
        ctx = ExecutionContext(run_id="r3", client_id="client_a",
                               tenant_id="client_a",
                               output_root=str(tmp_path / "run3"),
                               config_snapshot=snap)
        run_onboarding(str(delivery_dir), eff, ctx)
        after = {p: p.stat().st_mtime_ns
                 for p in Path("config").rglob("*.yaml")}
        assert before == after, "repository configuration must not be touched"


class TestOccIntegration:
    def test_run_pins_effective_config_and_rerun_versions_it(
            self, store, source_registry, delivery_dir, tmp_path):
        engine = make_engine(store, source_registry,
                             client_config=make_client_config(tmp_path,
                                                              valid=False))
        run = register_and_create(engine, delivery_dir, outcome="mi_annex2")
        parked = start_and_wait(engine, run)
        assert parked.status == "needs_review"
        assert parked.effective_config["status"] == "BLOCKED"
        v1 = parked.effective_config["version"]

        dec = [d for d in store.open_decisions("client_a", run.workflow_id)
               if d["kind"] == "client_rule"][0]
        engine.resolve_decision(client_id="client_a",
                                decision_id=dec["decision_id"],
                                action="amend", actor="alice",
                                value="STUBLEI0000000000042", scope="client")
        final = wait_for(lambda: (
            (lambda r: r if r and r.status == "awaiting_publication"
             else None)(store.load_workflow("client_a", run.workflow_id))))
        assert final.effective_config["status"] in ("READY",
                                                    "READY_WITH_WARNINGS")
        assert final.effective_config["version"] > v1
        assert final.effective_config["decision_set_version"] != \
            parked.effective_config["decision_set_version"]
        # Both versions remain reconstructable.
        res = EffectiveConfigResolver(store, RuleStore(store))
        assert res.load("client_a", run.workflow_id, version=v1) is not None
        assert res.load("client_a", run.workflow_id,
                        version=final.effective_config["version"]) is not None
