#!/usr/bin/env python3
"""tests/test_ops_workflow.py

The operator feedback loop for blob-trigger onboarding: actionable event
manifests (next_action advisory), the durable run ledger (trakt-state/runs),
and the ``ops`` CLI operations (list-halted / show / show-recommendations /
approve / edit / promote / rerun) across the four supported scenarios:
new source, schema drift, incomplete MI-contract handoff, and force_reprocess
after approval (with force_publish as an explicit break-glass).

Run: python -m unittest tests.test_ops_workflow
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from apps.blob_trigger_app import router as R
from apps.blob_trigger_app import ops as OPS
from apps.blob_trigger_app import approvals as APP
from apps.blob_trigger_app import run_records as RR
from apps.blob_trigger_app.ops_advice import (
    next_operator_action, ACT_APPROVE_MAPPING, ACT_RESOLVE_LLM, ACT_RERUN,
    ACT_INSPECT_ONBOARDING, ACT_INSPECT_TRANSFORM, ACT_INSPECT_VALIDATION,
    ACT_INSPECT_ASSEMBLER, ACT_INSPECT_PROJECTION, ACT_FORCE_PUBLISH, ACT_NONE)
from apps.blob_trigger_app.layout import Layout
from apps.blob_trigger_app.persistence import ProductionPersistence
from apps.blob_trigger_app.schema_fingerprint import fingerprint_from_schema
from apps.blob_trigger_app.source_registry import (
    SourceRecord, SourceRegistry, STATUS_ACTIVE)
from apps.blob_trigger_app.storage import Storage
from apps.blob_trigger_app.llm_recommendations import (
    resolve_llm_policy, generate_recommendations, should_generate)

_NOW = "2026-10-01T00:00:00+00:00"


def _gate(name, status, **kw):
    base = {"gate_name": name, "status": status, "ready_flag_name": None,
            "ready_flag_value": None, "halt_reason": "", "issue_count": 0,
            "blocking_issue_count": 0, "warning_count": 0, "issues": [],
            "affected_fields": [], "severity_counts": {}, "source_artifact_paths": {},
            "persisted_artifact_uris": {}, "next_recommended_operator_action": "rerun",
            "payload": {}}
    base.update(kw)
    return base
_COLUMNS = ["loan_id", "balance", "rate", "origination_date"]


def _fp(columns=_COLUMNS, file_type="xlsx"):
    return fingerprint_from_schema(file_type=file_type, columns=columns)


class RecordingInvoker:
    def __init__(self, status="done", **extra):
        self.calls = []
        self.status = status
        self.extra = extra

    def __call__(self, **kw):
        self.calls.append(kw)
        return {"run_id": "orun_mock", "status": self.status,
                "central_canonical_path": ("/tmp/central.csv" if self.status == "done" else None),
                "blockers": [], **self.extra}


class DiagnosticInvoker:
    """Halts with a configurable diagnostics block (validation vs mapping-gap)."""

    def __init__(self, *, diagnostics):
        self.diagnostics = diagnostics
        self.calls = []

    def __call__(self, **kw):
        self.calls.append(kw)
        return {"run_id": "orun_halt", "status": "halted",
                "central_canonical_path": None,
                "blockers": ["direct_001/transform: halted"],
                "state_path": "/tmp/orun_halt/run_state.json",
                "diagnostics": self.diagnostics}


# --------------------------------------------------------------------------- #
# 1. next_operator_action advisory (pure)
# --------------------------------------------------------------------------- #

class TestNextActionAdvisory(unittest.TestCase):

    def _m(self, *, failed_gate=None, **kw):
        base = {"pack_key": "ERE_direct_001_funded_monthly_2025-11-30",
                "source_portfolio_id": "direct_001", "orchestrator_run_id": "orun_1"}
        base.update(kw)
        if failed_gate is not None:
            diag = dict(base.get("orchestrator_diagnostics") or {})
            diag.setdefault("run_summary", {})["failed_gate"] = failed_gate
            base["orchestrator_diagnostics"] = diag
        return base

    def test_new_source_says_approve_mapping(self):
        na = next_operator_action(self._m(
            event_decision="new_source_pending_review", status="pending_review",
            approval_id="appr_1"))
        self.assertEqual(na["action"], ACT_APPROVE_MAPPING)
        self.assertIn("approve appr_1", na["command"])
        self.assertTrue(any("promote appr_1" in c for c in na["then"]))
        self.assertTrue(any("rerun ERE_direct_001" in c for c in na["then"]))

    def test_new_source_with_llm_recs_says_resolve_llm(self):
        na = next_operator_action(self._m(
            event_decision="new_source_pending_review", status="pending_review",
            approval_id="appr_1", llm={"recommendations_present": True}))
        self.assertEqual(na["action"], ACT_RESOLVE_LLM)
        self.assertIn("show-llm", na["command"])
        self.assertIn("advisory", na["summary"].lower())

    def test_schema_drift_says_approve_mapping(self):
        na = next_operator_action(self._m(
            event_decision="schema_drift_pending_review", status="pending_review",
            approval_id="appr_2"))
        self.assertEqual(na["action"], ACT_APPROVE_MAPPING)

    def test_incomplete_pack_says_rerun(self):
        na = next_operator_action(self._m(
            event_decision="incomplete_pack_pending_review", status="pending_review"))
        self.assertEqual(na["action"], ACT_RERUN)

    def test_transform_gate_says_inspect_transform(self):
        na = next_operator_action(self._m(
            event_decision="known_source_halted", status="halted", failed_gate="transform",
            orchestrator_diagnostics={"transform_readiness": {
                "ready_for_validation": False, "issue_count": 69, "blocking_issue_count": 4,
                "affected_fields": ["current_interest_rate", "current_principal_balance"]}}))
        self.assertEqual(na["action"], ACT_INSPECT_TRANSFORM)
        self.assertEqual(na["failed_gate"], "transform")
        self.assertIn("show-transform", na["command"])
        self.assertIn("not mapping", na["summary"].lower())

    def test_validation_gate_says_inspect_validation_with_breakglass(self):
        na = next_operator_action(self._m(
            event_decision="known_source_halted", status="halted", failed_gate="validation",
            orchestrator_diagnostics={"validation_readiness": {
                "ready_for_validation_complete": False, "issue_count": 5, "blocking_issue_count": 3,
                "mandatory_field_failures": ["current_valuation_amount"],
                "numeric_parse_failures": ["current_interest_rate"]}}))
        self.assertEqual(na["action"], ACT_INSPECT_VALIDATION)
        self.assertIn("show-validation", na["command"])
        self.assertIn("current_valuation_amount", na["summary"])
        self.assertTrue(any("--force-publish" in c for c in na["then"]))

    def test_assembler_gate_says_inspect_assembler(self):
        na = next_operator_action(self._m(
            event_decision="known_source_halted", status="halted", failed_gate="assembler"))
        self.assertEqual(na["action"], ACT_INSPECT_ASSEMBLER)
        self.assertIn("show-gate", na["command"])

    def test_projection_gate_says_inspect_projection(self):
        na = next_operator_action(self._m(
            event_decision="known_source_halted", status="halted", failed_gate="projection"))
        self.assertEqual(na["action"], ACT_INSPECT_PROJECTION)

    def test_onboarding_gate_blocking_decisions_says_inspect_onboarding(self):
        # not mapping (registry_gap_count=0, no recs) — a run-context operator
        # decision (reporting_date). Never labelled a mapping fix.
        na = next_operator_action(self._m(
            event_decision="known_source_halted", status="halted", failed_gate="onboarding",
            orchestrator_diagnostics={
                "registry_gap_count": 0, "mapping_recommendations": [],
                "handoff_readiness": {
                    "ready_for_transformation_validation": False,
                    "blocking_decisions": [{"target_field": "reporting_date",
                                            "reason": "required but unmapped"}]}}))
        self.assertEqual(na["action"], ACT_INSPECT_ONBOARDING)
        self.assertIn("reporting_date", na["summary"])
        self.assertIn("show-handoff", na["command"])

    def test_onboarding_gate_metadata_mismatch_says_inspect_onboarding(self):
        na = next_operator_action(self._m(
            event_decision="known_source_halted", status="halted", failed_gate="onboarding",
            orchestrator_diagnostics={
                "registry_gap_count": 0, "mapping_recommendations": [],
                "handoff_readiness": {"ready_for_transformation_validation": False,
                                      "failed_readiness_gates": [], "blocking_decisions": []}}))
        self.assertEqual(na["action"], ACT_INSPECT_ONBOARDING)
        self.assertIn("metadata mismatch", na["summary"].lower())

    def test_no_gate_pinned_says_rerun(self):
        na = next_operator_action(self._m(
            event_decision="known_source_halted", status="halted",
            orchestrator_diagnostics={}))
        self.assertEqual(na["action"], ACT_RERUN)

    def test_processed_needs_nothing(self):
        na = next_operator_action(self._m(
            event_decision="known_source_processed", status="processed"))
        self.assertEqual(na["action"], ACT_NONE)


# --------------------------------------------------------------------------- #
# 2. Router → durable run ledger + actionable manifest
# --------------------------------------------------------------------------- #

class TestRunLedgerAndCli(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = self._tmp.name
        self.out = str(Path(self.root) / "scratch")
        self.storage = Storage(local_root=str(Path(self.root) / "blobstore"))
        self.layout = Layout()
        self.persistence = ProductionPersistence(self.storage, self.layout)
        self.registry = self.persistence.load_registry()
        self.fp = _fp()

    def tearDown(self):
        self._tmp.cleanup()

    def _seed_active(self, pid="direct_001", fp=None):
        self.registry.upsert(SourceRecord(
            client_id="ERE", source_portfolio_id=pid, dataset="funded",
            frequency="monthly", source_portfolio_type="direct", approved_mapping_id="m1",
            mapping_config_path="config/m1.yaml",
            expected_schema_fingerprint=(fp or self.fp.fingerprint), status=STATUS_ACTIVE))

    def _marker(self, blob, inv, *, meta=None):
        return R.handle_blob_event(
            blob, registry=self.registry, out_dir=self.out, container="raw-v2",
            pack_marker="_READY.json", schema_info=self.fp, marker_metadata=meta,
            orchestrator_invoker=inv, persistence=self.persistence, now=_NOW)

    # -- known source, incomplete MI-contract handoff --------------------- #
    def test_known_source_incomplete_handoff_creates_ledger_entry(self):
        self._seed_active()
        inv = DiagnosticInvoker(diagnostics={
            "halt_stage": "direct_001/onboard",
            "halt_reason": "onboarding handoff not ready_for_transformation_validation",
            "blocking_decisions": ["unresolved mapping decisions"],
            "registry_gap_count": 2, "validation_errors": [],
            "run_summary": {"failed_gate": "onboarding"},
            "mapping_recommendations": [
                {"field": "current_valuation_amount", "recommendation": "map from 'Val GBP'",
                 "confidence": 0.62}],
            "run_state_path": "/tmp/orun_halt/run_state.json"})
        m = self._marker(
            "raw-v2/ERE/direct/funded/monthly/direct_001/2025-11-30/_READY.json", inv)
        self.assertEqual(m["status"], "halted")
        # actionable manifest
        na = m["next_action"]
        self.assertEqual(na["action"], ACT_INSPECT_ONBOARDING)
        self.assertEqual(na["run_id"], "orun_halt")
        self.assertEqual(na["source_portfolio_id"], "direct_001")
        # durable ledger entry, listable + resolvable by run_id and pack_key
        halted = self.persistence.list_halted_runs()
        self.assertEqual(len(halted), 1)
        pack_key = halted[0]["pack_key"]
        self.assertEqual(pack_key, "ERE_direct_001_funded_monthly_2025-11-30")
        self.assertIsNotNone(RR.find_by_run_id(self.storage, self.layout, "orun_halt"))
        self.assertIsNotNone(RR.load_run_record(self.storage, self.layout, pack_key))
        # recommendations surfaced for the operator
        recs = OPS.recommendations(self.storage, self.layout, "orun_halt")
        self.assertEqual(recs["mapping_recommendations"][0]["field"], "current_valuation_amount")

    def test_handoff_readiness_persisted_and_show_handoff(self):
        # A not-ready handoff (blocking reporting_date decision, zero gaps, no recs)
        # must surface the FULL readiness payload durably + advise resolve_decisions.
        self._seed_active()
        cov = Path(self.root) / "scratch" / "28a_target_coverage_matrix.csv"
        cov.parent.mkdir(parents=True, exist_ok=True)
        cov.write_text("target_field,coverage_status\nreporting_date,missing_required\n")
        hr = {
            "source_portfolio_id": "direct_001",
            "ready_for_transformation_validation": False,
            "failed_readiness_gates": ["blocking_decision_count=1"],
            "blocking_decision_count": 1,
            "blocking_decisions": [{"target_field": "reporting_date",
                                    "reason": "required but unmapped"}],
            "missing_target_fields": ["reporting_date"],
            "unresolved_fields": ["reporting_date"],
            "registry_gap_count": 0, "issue_count": 2,
            "target_coverage_matrix_path": str(cov),
            "handoff_manifest": {"target_contract_id": "mi_semantics_field_registry",
                                 "ready_for_transformation_validation": False,
                                 "blocking_decision_count": 1},
        }
        inv = DiagnosticInvoker(diagnostics={
            "registry_gap_count": 0, "issue_count": 2, "mapping_recommendations": [],
            "validation_errors": [], "handoff_readiness": hr,
            "run_summary": {"failed_gate": "onboarding"}})
        m = self._marker(
            "raw-v2/ERE/direct/funded/monthly/direct_001/2025-11-30/_READY.json", inv)
        self.assertEqual(m["status"], "halted")
        # advisory: inspect_onboarding naming reporting_date — NOT a mapping fix.
        self.assertEqual(m["next_action"]["action"], ACT_INSPECT_ONBOARDING)
        self.assertIn("reporting_date", m["next_action"]["summary"])

        rec = self.persistence.load_run_record(m["pack_key"])
        self.assertEqual(rec["issue_count"], 2)
        self.assertFalse(rec["handoff_readiness"]["ready_for_transformation_validation"])
        self.assertEqual(rec["handoff_readiness"]["blocking_decisions"][0]["target_field"],
                         "reporting_date")
        self.assertEqual(rec["handoff_readiness"]["failed_readiness_gates"],
                         ["blocking_decision_count=1"])
        # durable handoff artifacts persisted to trakt-state (survive scratch cleanup)
        arts = rec["handoff_artifacts"]
        self.assertTrue(self.storage.exists(arts["handoff_manifest_uri"]))
        self.assertTrue(any(self.storage.exists(u)
                            for u in arts["target_coverage_matrix_uris"]))
        # ops show-handoff surfaces the readiness + artifact URIs
        h = OPS.handoff(self.storage, self.layout, "orun_halt")
        self.assertEqual(h["issue_count"], 2)
        self.assertEqual(h["handoff_readiness"]["missing_target_fields"], ["reporting_date"])
        self.assertIn("handoff_manifest_uri", h["handoff_artifacts"])

    def test_validation_halt_advises_inspect_validation(self):
        self._seed_active()
        inv = DiagnosticInvoker(diagnostics={
            "halt_stage": "direct_001/validate",
            "halt_reason": "validation blocked",
            "blocking_decisions": [], "registry_gap_count": 0,
            "validation_errors": ["balance < 0 on 5 rows"],
            "run_summary": {"failed_gate": "validation"},
            "validation_readiness": {
                "ready_for_validation_complete": False, "issue_count": 5,
                "blocking_issue_count": 3, "mandatory_field_failures": ["balance"],
                "numeric_parse_failures": []},
            "run_state_path": "/tmp/x/run_state.json"})
        m = self._marker(
            "raw-v2/ERE/direct/funded/monthly/direct_001/2025-11-30/_READY.json", inv)
        self.assertEqual(m["next_action"]["action"], ACT_INSPECT_VALIDATION)
        self.assertTrue(any("--force-publish" in c for c in m["next_action"]["then"]))
        # validation readiness surfaced via ops show-validation
        v = OPS.validation(self.storage, self.layout, m["orchestrator_run_id"])
        self.assertEqual(v["validation_readiness"]["mandatory_field_failures"], ["balance"])

    # -- new source ------------------------------------------------------- #
    def test_new_source_ledger_and_manifest_action_is_approve(self):
        inv = RecordingInvoker(status="halted")
        m = self._marker(
            "raw-v2/ERE/acquired/funded/ad_hoc/acquired_001/2025-11-30/_READY.json", inv,
            meta={"acquisition_date": "2025-10-31", "seller_name": "BigBank"})
        self.assertEqual(m["event_decision"], "new_source_pending_review")
        self.assertEqual(m["next_action"]["action"], ACT_APPROVE_MAPPING)
        self.assertEqual(m["next_action"]["approval_id"], m["approval_id"])
        # listable in the halted ledger (pending_review is actionable)
        self.assertEqual(len(self.persistence.list_halted_runs()), 1)

    # -- schema drift ----------------------------------------------------- #
    def test_schema_drift_ledger_and_action_is_approve(self):
        self._seed_active(fp="sha256:OLD")
        m = self._marker(
            "raw-v2/ERE/direct/funded/monthly/direct_001/2025-11-30/_READY.json",
            RecordingInvoker())
        self.assertEqual(m["event_decision"], "schema_drift_pending_review")
        self.assertEqual(m["next_action"]["action"], ACT_APPROVE_MAPPING)
        self.assertEqual(len(self.persistence.list_halted_runs()), 1)

    # -- processed leaves nothing in the halted ledger -------------------- #
    def test_processed_pack_not_in_halted_ledger(self):
        self._seed_active()
        m = self._marker(
            "raw-v2/ERE/direct/funded/monthly/direct_001/2025-11-30/_READY.json",
            RecordingInvoker(status="done"))
        self.assertEqual(m["status"], "processed")
        self.assertEqual(m["next_action"]["action"], ACT_NONE)
        self.assertEqual(self.persistence.list_halted_runs(), [])
        # but a durable record still exists (full ledger)
        self.assertIsNotNone(self.persistence.load_run_record(m["pack_key"]))

    # -- approve → edit → promote (mapping version) ----------------------- #
    def test_approve_edit_promote_bumps_mapping_version(self):
        inv = RecordingInvoker(status="halted")
        m = self._marker(
            "raw-v2/ERE/acquired/funded/ad_hoc/acquired_001/2025-11-30/_READY.json", inv)
        approval_id = m["approval_id"]
        # edit a recommendation before approving
        art = OPS.edit_approval(self.storage, self.layout, approval_id,
                                {"suggested_mapping_id": "ere_acq1_v1"})
        self.assertEqual(art["suggested_mapping_id"], "ere_acq1_v1")
        # non-editable field is refused
        with self.assertRaises(ValueError):
            OPS.edit_approval(self.storage, self.layout, approval_id, {"status": "approved"})
        # approve + promote
        APP.approve(self.storage, self.layout, approval_id,
                    mapping_id="ere_acq1_v1", mapping_config_path="config/acq1.yaml")
        rec = APP.promote(self.storage, self.layout, self.registry, approval_id)
        self.assertEqual(rec.status, STATUS_ACTIVE)
        self.assertEqual(rec.mapping_version, 1)
        # promoted mapping version recorded on the approval artifact
        art = APP.show(self.storage, self.layout, approval_id)
        self.assertEqual(art["promoted_mapping"]["mapping_version"], 1)
        # re-promote bumps the version again
        art["status"] = "approved"
        OPS.edit_approval(self.storage, self.layout, approval_id, {})
        self.storage.write_text(self.layout.approval_uri(approval_id),
                                __import__("json").dumps({**art, "status": "approved"}))
        rec2 = APP.promote(self.storage, self.layout, self.registry, approval_id)
        self.assertEqual(rec2.mapping_version, 2)

    # -- rerun re-fires with force_reprocess (+ break-glass) -------------- #
    def test_rerun_refires_pack_with_force_reprocess(self):
        self._seed_active()
        inv = DiagnosticInvoker(diagnostics={"registry_gap_count": 1, "blocking_decisions": ["x"],
                                             "validation_errors": []})
        m = self._marker(
            "raw-v2/ERE/direct/funded/monthly/direct_001/2025-11-30/_READY.json", inv)
        pack_key = m["pack_key"]

        captured = {}

        def fake_reprocessor(blob_path, *, container, input_dir, marker_metadata):
            captured.update(blob_path=blob_path, container=container,
                            input_dir=input_dir, marker_metadata=marker_metadata)
            return {"status": "processed", "next_action": {"action": "none"}}

        out = OPS.rerun(self.persistence, self.registry, pack_key,
                        reprocessor=fake_reprocessor)
        self.assertEqual(out["status"], "processed")
        self.assertTrue(captured["marker_metadata"]["force_reprocess"])
        self.assertNotIn("force_publish", captured["marker_metadata"])
        self.assertEqual(captured["blob_path"],
                         "raw-v2/ERE/direct/funded/monthly/direct_001/2025-11-30/_READY.json")

    def test_rerun_force_publish_is_explicit_breakglass(self):
        self._seed_active()
        inv = DiagnosticInvoker(diagnostics={"validation_errors": ["bad"]})
        m = self._marker(
            "raw-v2/ERE/direct/funded/monthly/direct_001/2025-11-30/_READY.json", inv)

        captured = {}

        def fake_reprocessor(blob_path, *, container, input_dir, marker_metadata):
            captured.update(marker_metadata=marker_metadata)
            return {"status": "processed"}

        OPS.rerun(self.persistence, self.registry, m["pack_key"],
                  force_publish=True, reprocessor=fake_reprocessor)
        self.assertTrue(captured["marker_metadata"]["force_publish"])

    def test_rerun_unknown_pack_raises(self):
        with self.assertRaises(KeyError):
            OPS.rerun(self.persistence, self.registry, "no_such_pack")

    # -- CLI dispatch smoke ---------------------------------------------- #
    def test_cli_list_and_show(self):
        import os
        self._seed_active()
        inv = DiagnosticInvoker(diagnostics={"registry_gap_count": 1, "blocking_decisions": ["x"]})
        self._marker(
            "raw-v2/ERE/direct/funded/monthly/direct_001/2025-11-30/_READY.json", inv)
        # point the CLI at the same emulated blob root
        os.environ["TRAKT_LOCAL_BLOB_ROOT"] = str(Path(self.root) / "blobstore")
        try:
            self.assertEqual(OPS.main(["list-halted"]), 0)
            self.assertEqual(OPS.main(["show", "orun_halt"]), 0)
            self.assertEqual(OPS.main(["show-recommendations", "orun_halt"]), 0)
            self.assertEqual(OPS.main(["show-handoff", "orun_halt"]), 0)
            self.assertEqual(OPS.main(["show-handoff", "does_not_exist"]), 1)
            self.assertEqual(OPS.main(["show", "does_not_exist"]), 1)
        finally:
            os.environ.pop("TRAKT_LOCAL_BLOB_ROOT", None)


# --------------------------------------------------------------------------- #
# 3. Generic gate framework + LLM advisory + debug-storage
# --------------------------------------------------------------------------- #

class TestGateFrameworkAndLLM(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = self._tmp.name
        self.out = str(Path(self.root) / "scratch")
        self.storage = Storage(local_root=str(Path(self.root) / "blobstore"))
        self.layout = Layout()
        self.persistence = ProductionPersistence(self.storage, self.layout)
        self.registry = self.persistence.load_registry()
        self.fp = _fp()

    def tearDown(self):
        self._tmp.cleanup()

    def _seed_active(self, pid="direct_001"):
        self.registry.upsert(SourceRecord(
            client_id="ERE", source_portfolio_id=pid, dataset="funded",
            frequency="monthly", source_portfolio_type="direct", approved_mapping_id="m1",
            mapping_config_path="config/m1.yaml",
            expected_schema_fingerprint=self.fp.fingerprint, status=STATUS_ACTIVE))

    def _fire(self, inv, *, llm_generator=None, meta=None):
        return R.handle_blob_event(
            "raw-v2/ERE/direct/funded/monthly/direct_001/2025-11-30/_READY.json",
            registry=self.registry, out_dir=self.out, container="raw-v2",
            pack_marker="_READY.json", schema_info=self.fp, marker_metadata=meta,
            orchestrator_invoker=inv, persistence=self.persistence,
            llm_generator=llm_generator, now=_NOW)

    # -- transform gate halted → persisted + inspect_transform ------------- #
    def test_transform_gate_persisted_and_next_action(self):
        self._seed_active()
        issues = Path(self.root) / "scratch" / "transformation_issues.json"
        issues.parent.mkdir(parents=True, exist_ok=True)
        issues.write_text(json.dumps({"rows": [
            {"target_field": "current_interest_rate", "issue_type": "numeric_parse_failed",
             "severity": "error"}]}))
        gates = [
            _gate("onboarding", "done", ready_flag_name="ready_for_transformation_validation",
                  ready_flag_value=True),
            _gate("transform", "halted", ready_flag_name="ready_for_validation",
                  ready_flag_value=False, halt_reason="transformation not ready_for_validation",
                  issue_count=1, blocking_issue_count=1, affected_fields=["current_interest_rate"],
                  source_artifact_paths={"transformation_issues_json": str(issues)},
                  next_recommended_operator_action="inspect_transform",
                  payload={"ready_for_validation": False, "issue_count": 1,
                           "blocking_issue_count": 1, "affected_fields": ["current_interest_rate"]}),
        ]
        diagnostics = {
            "registry_gap_count": 0, "validation_errors": [], "mapping_recommendations": [],
            "transform_readiness": gates[1]["payload"], "gates": gates,
            "run_summary": {"failed_gate": "transform", "failed_gate_status": "halted",
                            "gate_status": {"onboarding": "done", "transform": "halted"},
                            "central_canonical_path": None,
                            "central_canonical_unavailable_reason": "halted at transform",
                            "next_action_key": "inspect_transform"}}
        m = self._fire(DiagnosticInvoker(diagnostics=diagnostics))
        self.assertEqual(m["next_action"]["action"], ACT_INSPECT_TRANSFORM)
        pk = m["pack_key"]
        # per-gate diagnostics persisted durably
        gd = self.persistence.load_gate_diagnostics(pk, "transform")
        self.assertEqual(gd["status"], "halted")
        self.assertEqual(gd["blocking_issue_count"], 1)
        self.assertIn("transform", self.persistence.list_gate_names(pk))
        # gate artifact (issues file) copied durably
        self.assertIn("transformation_issues_json", gd["persisted_artifact_uris"])
        self.assertTrue(self.storage.exists(gd["persisted_artifact_uris"]["transformation_issues_json"]))
        # run record carries the gate status summary
        rec = self.persistence.load_run_record(pk)
        self.assertEqual(rec["failed_gate"], "transform")
        self.assertEqual(rec["gate_status"]["transform"], "halted")
        self.assertIn("halted at transform", rec["central_canonical_unavailable_reason"])
        # ops show-gates / show-gate read the persisted diagnostics
        allg = OPS.gates(self.persistence, pk)
        self.assertEqual(allg["failed_gate"], "transform")
        self.assertIn("transform", allg["gates"])
        one = OPS.gate(self.persistence, pk, "transform")
        self.assertEqual(one["gate_name"], "transform")

    def test_validation_gate_persisted_and_next_action(self):
        self._seed_active()
        gates = [
            _gate("onboarding", "done"), _gate("transform", "done"),
            _gate("validation", "halted", ready_flag_name="ready_for_validation_complete",
                  ready_flag_value=False, issue_count=3, blocking_issue_count=3,
                  next_recommended_operator_action="inspect_validation",
                  payload={"ready_for_validation_complete": False, "issue_count": 3,
                           "blocking_issue_count": 3, "mandatory_field_failures": ["balance"],
                           "numeric_parse_failures": ["rate"]}),
        ]
        diagnostics = {
            "validation_readiness": gates[2]["payload"], "gates": gates,
            "run_summary": {"failed_gate": "validation",
                            "gate_status": {"validation": "halted"},
                            "next_action_key": "inspect_validation"}}
        m = self._fire(DiagnosticInvoker(diagnostics=diagnostics))
        self.assertEqual(m["next_action"]["action"], ACT_INSPECT_VALIDATION)
        pk = m["pack_key"]
        gd = self.persistence.load_gate_diagnostics(pk, "validation")
        self.assertEqual(gd["status"], "halted")
        v = OPS.validation(self.storage, self.layout, pk)
        self.assertEqual(v["validation_readiness"]["mandatory_field_failures"], ["balance"])

    # -- debug-storage ---------------------------------------------------- #
    def test_debug_storage_reports_run_record_uri_and_existence(self):
        self._seed_active()
        gates = [_gate("transform", "halted", next_recommended_operator_action="inspect_transform",
                       payload={"ready_for_validation": False})]
        m = self._fire(DiagnosticInvoker(diagnostics={
            "transform_readiness": gates[0]["payload"], "gates": gates,
            "run_summary": {"failed_gate": "transform"}}))
        pk = m["pack_key"]
        info = OPS.debug_storage(self.persistence, pk)
        self.assertIn(info["selected_backend"], ("filesystem", "azure_blob"))
        self.assertIn("TRAKT_BLOB_CONNECTION_present", info)
        self.assertEqual(info["state_container"], "trakt-state")
        self.assertEqual(info["run_record_uri"], self.layout.run_uri(pk))
        self.assertTrue(info["run_record_exists"])
        self.assertTrue(info["gates_folder_exists"])
        # unknown pack → does not exist
        info2 = OPS.debug_storage(self.persistence, "nope_pack")
        self.assertFalse(info2["run_record_exists"])
        self.assertFalse(info2["gates_folder_exists"])

    # -- LLM policy + fallback (unit) ------------------------------------- #
    def test_llm_disabled_by_default(self):
        p = resolve_llm_policy({})
        self.assertFalse(p["enabled"])
        self.assertFalse(p["available"])

    def test_llm_enabled_key_missing_deterministic_fallback(self):
        p = resolve_llm_policy({"TRAKT_LLM_ENABLED": "true"})
        self.assertTrue(p["enabled"])
        self.assertFalse(p["available"])
        recs, meta = generate_recommendations(
            pack_key="pk", decision="source_onboarding", gates=[], gate_failed=True, policy=p)
        self.assertEqual(recs, [])
        self.assertFalse(meta["llm_invoked"])
        self.assertFalse(meta["llm_available"])
        self.assertTrue(meta["deterministic_fallback_used"])

    def test_llm_clean_known_source_not_invoked(self):
        p = resolve_llm_policy({"TRAKT_LLM_ENABLED": "true", "ANTHROPIC_API_KEY": "k"})
        self.assertFalse(should_generate("deterministic", gate_failed=False, policy=p))
        recs, meta = generate_recommendations(
            pack_key="pk", decision="deterministic", gates=[], gate_failed=False, policy=p)
        self.assertFalse(meta["llm_invoked"])
        self.assertEqual(recs, [])

    def test_llm_new_source_mocked_recs_not_auto_applied(self):
        p = resolve_llm_policy({"TRAKT_LLM_ENABLED": "true", "ANTHROPIC_API_KEY": "k"})
        gen = lambda ctx: [{"target_field": "current_valuation_amount",
                            "recommended_mapping": "Val GBP", "rationale": "collateral MI",
                            "confidence": 0.72}]
        recs, meta = generate_recommendations(
            pack_key="pk", decision="source_onboarding", gates=[], gate_failed=True,
            generator=gen, policy=p)
        self.assertTrue(meta["llm_invoked"])
        self.assertTrue(meta["recommendations_present"])
        self.assertEqual(recs[0]["approval_required"], True)     # never auto-applied
        self.assertEqual(recs[0]["status"], "pending")

    def test_llm_error_falls_back_not_fails(self):
        p = resolve_llm_policy({"TRAKT_LLM_ENABLED": "true", "ANTHROPIC_API_KEY": "k"})
        def boom(ctx):
            raise RuntimeError("rate limited")
        recs, meta = generate_recommendations(
            pack_key="pk", decision="schema_drift", gates=[], gate_failed=True,
            generator=boom, policy=p)
        self.assertEqual(recs, [])
        self.assertIn("rate limited", meta["llm_error"])
        self.assertTrue(meta["deterministic_fallback_used"])

    # -- LLM through the router (advisory, persisted, not applied) --------- #
    def test_router_llm_recs_persisted_and_advisory(self):
        gen = lambda ctx: [{"target_field": "x", "recommended_mapping": "Y", "confidence": 0.6}]
        os.environ["TRAKT_LLM_ENABLED"] = "true"
        os.environ["ANTHROPIC_API_KEY"] = "k"
        try:
            m = R.handle_blob_event(
                "raw-v2/ERE/acquired/funded/ad_hoc/acquired_001/2025-11-30/_READY.json",
                registry=self.registry, out_dir=self.out, container="raw-v2",
                pack_marker="_READY.json", schema_info=self.fp,
                orchestrator_invoker=RecordingInvoker(status="halted"),
                persistence=self.persistence, llm_generator=gen, now=_NOW)
        finally:
            os.environ.pop("TRAKT_LLM_ENABLED", None)
            os.environ.pop("ANTHROPIC_API_KEY", None)
        self.assertTrue(m["llm"]["llm_invoked"])
        self.assertTrue(m["llm"]["recommendations_present"])
        self.assertIsNotNone(m["llm"]["recommendations_artifact_uri"])
        # persisted + advisory-only (operator must approve)
        l = OPS.show_llm(self.persistence, m["pack_key"])
        self.assertTrue(l["recommendations_doc"]["advisory_only"])
        self.assertEqual(l["recommendations"][0]["status"], "pending")
        self.assertEqual(l["recommendations"][0]["approval_required"], True)

    def test_router_clean_known_source_no_llm_call(self):
        self._seed_active()
        called = []
        gen = lambda ctx: (called.append(1), [])[1]
        os.environ["TRAKT_LLM_ENABLED"] = "true"
        os.environ["ANTHROPIC_API_KEY"] = "k"
        try:
            m = self._fire(RecordingInvoker(status="done"), llm_generator=gen)
        finally:
            os.environ.pop("TRAKT_LLM_ENABLED", None)
            os.environ.pop("ANTHROPIC_API_KEY", None)
        self.assertEqual(m["status"], "processed")
        self.assertFalse(m["llm"]["llm_invoked"])
        self.assertEqual(called, [])

    # -- no routing/contract regression ----------------------------------- #
    def test_no_contract_routing_regression(self):
        from engine.orchestrator_agent.orchestrator import onboarding_mode_for_target
        self.assertEqual(onboarding_mode_for_target("mi"), "mi_only")        # MI contract
        self.assertEqual(onboarding_mode_for_target("regime"), "regulatory_mi")  # Annex 2
        self.assertEqual(onboarding_mode_for_target("all"), "regulatory_mi")     # combined


if __name__ == "__main__":
    unittest.main()
