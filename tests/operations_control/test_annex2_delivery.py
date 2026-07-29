"""Governed Annex 2 delivery chain: preflight (I3), population reconciliation
(I2), stage execution + idempotency (I1), intervention evidence (I4),
publication metadata, failure handling, and a real-subprocess mini-golden run
of the authoritative normaliser + builder + XSD."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from operations_control.contracts import (
    RUN_AWAITING_PUBLICATION,
    RUN_BLOCKED,
    RUN_NEEDS_REVIEW,
    RUN_PUBLISHED,
)
from operations_control.engine import OpsError

from .conftest import (
    StubAnnex2Stages,
    make_client_config,
    make_engine,
    register_and_create,
    start_and_wait,
    wait_for,
)


def _annex2_run(engine, delivery_dir, **kw):
    return register_and_create(engine, delivery_dir, outcome="mi_annex2", **kw)


class TestChainHappyPath:
    def test_runs_to_awaiting_publication_with_full_metadata(
            self, store, source_registry, delivery_dir):
        engine = make_engine(store, source_registry)
        run = _annex2_run(engine, delivery_dir)
        final = start_and_wait(engine, run)
        assert final.status == RUN_AWAITING_PUBLICATION
        for stage in ("regulatory_config", "projection", "delivery_prep",
                      "xml_delivery"):
            assert final.stage_status(stage) == "completed", stage
        assert final.stage_status("publication") == "ready"
        # Artefact references persisted on the workflow document.
        assert final.annex2["projected_csv"]
        assert final.annex2["delivery_ready_csv"]
        assert Path(final.annex2["xml"]).exists()
        assert final.annex2["xsd_result"] == "PASSED"
        assert final.annex2["xml_sha256"] == "sha256:stub"
        # Publication doc carries the required provenance.
        pub = store.load_publication("client_a", "2026-06-30")
        assert pub["annex2"]["xml"] == final.annex2["xml"]
        assert pub["configuration_versions"]["delivery_rules"] == "rules@test"
        assert pub["agent_versions"]["xml_builder"].endswith(
            "xml_builder_annex2.py")
        # Never published automatically.
        assert pub["status"] == "prepared" and pub["published_at"] is None

    def test_xml_stage_warnings_surface_interventions(
            self, store, source_registry, delivery_dir):
        engine = make_engine(store, source_registry)
        run = _annex2_run(engine, delivery_dir)
        start_and_wait(engine, run)
        gar = store.load_result("client_a", run.workflow_id, "xml_delivery")
        assert "automatic values require review" in gar.summary.lower()
        assert any("144" in w for w in gar.warnings)
        pub_gar = store.load_result("client_a", run.workflow_id, "publication")
        assert any("144" in w for w in pub_gar.warnings)
        # Audit trail carries the intervention evidence.
        events = [a for a in store.list_audit("client_a")
                  if a["event"] == "annex2_xml_generation"]
        assert events and "xml_sha256" in events[-1]["detail"]

    def test_publish_uses_existing_promotion_paths(
            self, store, source_registry, delivery_dir, ops_env):
        engine = make_engine(store, source_registry)
        run = _annex2_run(engine, delivery_dir)
        start_and_wait(engine, run)
        pub = engine.approve_publication(client_id="client_a",
                                         workflow_id=run.workflow_id,
                                         actor="alice")
        assert pub["status"] == "published"
        assert pub["published_artefacts"]["regime"], \
            "Annex 2 artefacts must publish through the regime prefix"
        regime_dir = (ops_env["blob_root"] / "processed-v2" / "regime"
                      / "client_a" / "2026-06-30")
        assert any(regime_dir.rglob("annex2_submission.xml"))
        assert store.load_workflow(
            "client_a", run.workflow_id).status == RUN_PUBLISHED

    def test_duplicate_publish_rejected(self, store, source_registry,
                                        delivery_dir):
        engine = make_engine(store, source_registry)
        run = _annex2_run(engine, delivery_dir)
        start_and_wait(engine, run)
        engine.approve_publication(client_id="client_a",
                                   workflow_id=run.workflow_id, actor="alice")
        with pytest.raises(OpsError):
            engine.approve_publication(client_id="client_a",
                                       workflow_id=run.workflow_id,
                                       actor="alice")


class TestIdempotencyAndRecovery:
    def test_rerun_does_not_rebuild_xml(self, store, source_registry,
                                        delivery_dir):
        stub = StubAnnex2Stages()
        engine = make_engine(store, source_registry, annex2_stages=stub)
        run = _annex2_run(engine, delivery_dir)
        final = start_and_wait(engine, run)
        assert stub.calls.count("xml") == 1
        engine.reject_publication(client_id="client_a",
                                  workflow_id=run.workflow_id,
                                  actor="alice", reason="not yet")
        engine.rerun(store.load_workflow("client_a", run.workflow_id),
                     actor="alice")
        final = wait_for(lambda: (
            (lambda r: r if r and r.status == RUN_AWAITING_PUBLICATION
             else None)(store.load_workflow("client_a", run.workflow_id))))
        # Completed stages with intact artefacts were skipped — one XML build.
        assert stub.calls.count("xml") == 1
        assert stub.calls.count("normalisation") == 1

    def test_reconstruction_after_restart(self, store, storage,
                                          source_registry, delivery_dir):
        from operations_control.stores import OpsLayout, OpsStore
        engine = make_engine(store, source_registry)
        run = _annex2_run(engine, delivery_dir)
        start_and_wait(engine, run)
        fresh = OpsStore(storage, OpsLayout("operations-control"))
        r = fresh.load_workflow("client_a", run.workflow_id)
        assert r.status == RUN_AWAITING_PUBLICATION
        assert r.stage_status("xml_delivery") == "completed"
        assert r.annex2["xsd_result"] == "PASSED"
        assert fresh.load_result("client_a", run.workflow_id,
                                 "xml_delivery") is not None
        assert fresh.verify_audit_chain("client_a")


class TestPreflight:
    def test_invalid_lei_blocks_with_client_rule_decision(
            self, store, source_registry, delivery_dir, tmp_path):
        engine = make_engine(store, source_registry,
                             client_config=make_client_config(tmp_path,
                                                              valid=False))
        run = _annex2_run(engine, delivery_dir)
        parked = start_and_wait(engine, run)
        assert parked.status == RUN_NEEDS_REVIEW
        assert parked.stage_status("regulatory_config") == "needs_review"
        gar = store.load_result("client_a", run.workflow_id,
                                "regulatory_config")
        assert "regulatory detail" in gar.summary
        decs = [d for d in store.open_decisions("client_a", run.workflow_id)
                if d["kind"] == "client_rule"]
        assert decs and decs[0]["blocking"]
        assert "LEI" in decs[0]["title"] or "identifier" in decs[0]["title"].lower()

    def test_client_rule_approval_unblocks_and_materialises_overlay(
            self, store, source_registry, delivery_dir, tmp_path):
        stub = StubAnnex2Stages()
        engine = make_engine(store, source_registry, annex2_stages=stub,
                             client_config=make_client_config(tmp_path,
                                                              valid=False))
        run = _annex2_run(engine, delivery_dir)
        start_and_wait(engine, run)
        dec = [d for d in store.open_decisions("client_a", run.workflow_id)
               if d["kind"] == "client_rule"][0]
        out = engine.resolve_decision(
            client_id="client_a", decision_id=dec["decision_id"],
            action="amend", actor="alice", value="STUBLEI0000000000042",
            scope="client")
        assert out["rule"]["kind"] == "client_rule"
        assert out["rule"]["scope"] == "client"
        final = wait_for(lambda: (
            (lambda r: r if r and r.status == RUN_AWAITING_PUBLICATION
             else None)(store.load_workflow("client_a", run.workflow_id))))
        assert final.stage_status("regulatory_config") == "completed"
        # The projector received the materialised overlay config, not the
        # repository file.
        proj_calls = [c for c in stub.calls
                      if isinstance(c, tuple) and c[0] == "projection"]
        assert proj_calls and "effective_client_config.yaml" in proj_calls[-1][1]

    def test_overlay_materialisation_replaces_nested_value(self, tmp_path):
        from operations_control.annex2.preflight import (
            materialise_effective_config, run_preflight)
        base = make_client_config(tmp_path, valid=False)
        out = materialise_effective_config(
            base_config_path=base,
            overrides={"originator_legal_entity_identifier":
                       "STUBLEI0000000000042"},
            out_path=tmp_path / "effective.yaml")
        pf = run_preflight(client_config_path=out, overrides={},
                           reporting_period="2026-06-30")
        assert pf["status"] in ("ready", "needs_review")
        assert not pf["blocked"]


class TestFailurePaths:
    def test_normalisation_block_parks_workflow(self, store, source_registry,
                                                delivery_dir):
        engine = make_engine(store, source_registry,
                             annex2_scenario="normalise_block")
        run = _annex2_run(engine, delivery_dir)
        final = start_and_wait(engine, run)
        assert final.status == RUN_BLOCKED
        assert final.stage_status("delivery_prep") == "blocked"
        gar = store.load_result("client_a", run.workflow_id, "delivery_prep")
        from operations_control import language
        for text in (gar.summary, *gar.blockers):
            assert language.is_operator_safe(text), text

    def test_xsd_failure_parks_workflow(self, store, source_registry,
                                        delivery_dir):
        engine = make_engine(store, source_registry, annex2_scenario="xsd_fail")
        run = _annex2_run(engine, delivery_dir)
        final = start_and_wait(engine, run)
        assert final.status == RUN_BLOCKED
        assert final.stage_status("xml_delivery") == "blocked"


class TestPopulationReconciliation:
    def test_no_blockers_and_correct_classifications(self):
        from operations_control.annex2.population import (
            assess_population, reconciliation_document)
        doc = reconciliation_document()
        assert doc["universe_count"] == 107
        assert doc["blocking_count"] == 0
        rows = {r["annex2_code"]: r for r in doc["rows"]}
        # RREL20/21: optional, ND-permitted, builder-covered — NOT blockers.
        for code in ("RREL20", "RREL21"):
            assert rows[code]["blocking"] is False
            assert "builder" in rows[code]["population_mechanism"]
        # Unruled-but-registry-mapped codes do not block (rule count is not
        # a completeness measure).
        unruled = [r for r in doc["rows"]
                   if r["population_mechanism"] == "projected registry mapping"]
        assert unruled and all(not r["blocking"] for r in unruled)
        # Deferred code recorded as such.
        assert rows["RREC22"]["applicability"] == "deferred"

    def test_reconciliation_persisted_per_run(self, store, source_registry,
                                              delivery_dir):
        engine = make_engine(store, source_registry)
        run = _annex2_run(engine, delivery_dir)
        final = start_and_wait(engine, run)
        recon_path = Path(final.annex2["population_reconciliation"])
        assert recon_path.exists()
        doc = json.loads(recon_path.read_text())
        assert doc["universe_count"] == 107
        audits = [a for a in store.list_audit("client_a")
                  if a["event"] == "annex2_population_reconciliation"]
        assert audits and audits[-1]["detail"]["blocking_count"] == 0


class TestRealComponentsMiniGolden:
    """Real subprocess run of the authoritative normaliser + builder + XSD on
    the committed synthetic projection (36 records). Proves the correct
    scripts, rules, workbook and XSD are invoked and that validation passes."""

    @pytest.fixture()
    def real_outcome(self, tmp_path):
        from operations_control.annex2.stages import Annex2Stages
        projected = Path("synthetic_demo/output/"
                         "SYNTHETIC_ERE_Portfolio_012026_ESMA_Annex2_projected.csv")
        assert projected.exists()
        stages = Annex2Stages()
        norm = stages.run_normalisation(projected_csv=str(projected),
                                        out_dir=tmp_path / "delivery")
        xml = None
        if norm.ok:
            xml = stages.run_xml(
                delivery_csv=norm.artefacts["delivery_ready_csv"],
                out_dir=tmp_path / "xml")
        return norm, xml

    def test_normaliser_and_builder_pass_xsd(self, real_outcome):
        norm, xml = real_outcome
        assert norm.ok, norm.blockers
        assert norm.summary == ("Annex 2 data has been prepared for XML "
                                "generation.")
        assert norm.metrics["output_records"] == 36
        assert "annex2_delivery_rules.yaml" in norm.metrics["rules_version"] \
            or norm.metrics["rules_version"].startswith("annex2_delivery_rules")
        assert xml is not None and xml.ok, (xml and xml.blockers)
        assert xml.metrics["xsd_result"] == "PASSED"
        assert xml.metrics["records"] == 36
        assert xml.metrics["xml_sha256"].startswith("sha256:")
        assert xml.metrics["xsd"].startswith("DRAFT1auth.099.001.04_1.3.0.xsd")
        assert Path(xml.artefacts["xml"]).name == "annex2_submission.xml"

    def test_builder_interventions_are_captured(self, real_outcome):
        norm, xml = real_outcome
        assert xml is not None and xml.ok
        iv = json.loads(Path(xml.artefacts["interventions"]).read_text())
        # The builder's documented ND scaffolding must be visible as evidence.
        types = {i["intervention_type"] for i in iv["interventions"]}
        assert "nd_insertion" in types
        # On this PRF fixture the builder injects the secondary-income ND5
        # pair (RREL20/RREL21) for every record: 2 x 36 = 72 instances.
        sec = [i for i in iv["interventions"] if i["code"] == "RREL20/RREL21"]
        assert sec and sec[0]["count"] == 2 * 36
        assert iv["nodata_injected_by_builder"] == 2 * 36
        # Currency attributes stamped by the builder are informational evidence.
        ccy = [i for i in iv["interventions"] if i["code"] == "Amt@Ccy"]
        assert ccy and ccy[0]["severity"] == "info"
        assert iv["review_required_instances"] > 0
        assert "Review is required" in iv["summary_sentence"]
        assert "automatic values require review" in xml.summary.lower()

    def test_wrong_xsd_fails_closed(self, tmp_path, real_outcome):
        from operations_control.annex2.stages import Annex2Stages
        norm, _ = real_outcome
        bad_xsd = tmp_path / "bad.xsd"
        bad_xsd.write_text("<xs:schema xmlns:xs='http://www.w3.org/2001/"
                           "XMLSchema'><xs:element name='Wrong'/></xs:schema>",
                           encoding="utf-8")
        out = Annex2Stages().run_xml(
            delivery_csv=norm.artefacts["delivery_ready_csv"],
            out_dir=tmp_path / "xml_bad", xsd_path=bad_xsd)
        assert not out.ok and out.blocking
        assert not (tmp_path / "xml_bad" / "annex2_submission.xml").exists(), \
            "a non-validated XML must never be finalised"
        from operations_control import language
        for text in (out.summary, *out.blockers):
            assert language.is_operator_safe(text), text
