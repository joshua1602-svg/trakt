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
    next_operator_action, ACT_APPROVE, ACT_FIX_DATA, ACT_FIX_MAPPING,
    ACT_RERUN, ACT_INVESTIGATE, ACT_NONE)
from apps.blob_trigger_app.layout import Layout
from apps.blob_trigger_app.persistence import ProductionPersistence
from apps.blob_trigger_app.schema_fingerprint import fingerprint_from_schema
from apps.blob_trigger_app.source_registry import (
    SourceRecord, SourceRegistry, STATUS_ACTIVE)
from apps.blob_trigger_app.storage import Storage

_NOW = "2026-10-01T00:00:00+00:00"
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

    def _m(self, **kw):
        base = {"pack_key": "ERE_direct_001_funded_monthly_2025-11-30",
                "source_portfolio_id": "direct_001", "orchestrator_run_id": "orun_1"}
        base.update(kw)
        return base

    def test_new_source_says_approve_then_promote_then_rerun(self):
        na = next_operator_action(self._m(
            event_decision="new_source_pending_review", status="pending_review",
            approval_id="appr_1"))
        self.assertEqual(na["action"], ACT_APPROVE)
        self.assertIn("approve appr_1", na["command"])
        self.assertIn("--mapping-id", na["command"])
        self.assertTrue(any("promote appr_1" in c for c in na["then"]))
        self.assertTrue(any("rerun ERE_direct_001" in c for c in na["then"]))
        # carries the identifiers an operator needs
        self.assertEqual(na["approval_id"], "appr_1")
        self.assertEqual(na["source_portfolio_id"], "direct_001")

    def test_schema_drift_says_approve(self):
        na = next_operator_action(self._m(
            event_decision="schema_drift_pending_review", status="pending_review",
            approval_id="appr_2"))
        self.assertEqual(na["action"], ACT_APPROVE)

    def test_incomplete_pack_says_fix_data_supply(self):
        na = next_operator_action(self._m(
            event_decision="incomplete_pack_pending_review", status="pending_review"))
        self.assertEqual(na["action"], ACT_FIX_DATA)

    def test_known_halt_with_validation_errors_says_fix_data_with_breakglass(self):
        na = next_operator_action(self._m(
            event_decision="known_source_halted", status="halted",
            orchestrator_diagnostics={"validation_errors": ["balance < 0 on 3 rows"]}))
        self.assertEqual(na["action"], ACT_FIX_DATA)
        self.assertTrue(any("--force-publish" in c for c in na["then"]))

    def test_known_halt_with_mapping_gaps_says_fix_mapping(self):
        na = next_operator_action(self._m(
            event_decision="known_source_halted", status="halted",
            orchestrator_diagnostics={"registry_gap_count": 3, "blocking_decisions": ["x"]}))
        self.assertEqual(na["action"], ACT_FIX_MAPPING)
        self.assertIn("show-recommendations", na["command"])

    def test_known_halt_clean_says_rerun(self):
        na = next_operator_action(self._m(
            event_decision="known_source_halted", status="halted",
            orchestrator_diagnostics={}))
        self.assertEqual(na["action"], ACT_RERUN)

    def test_failed_says_investigate(self):
        na = next_operator_action(self._m(event_decision="failed", status="failed"))
        self.assertEqual(na["action"], ACT_INVESTIGATE)

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
            "halt_stage": "direct_001/transform",
            "halt_reason": "onboarding handoff not ready_for_transformation_validation",
            "blocking_decisions": ["unresolved mapping decisions"],
            "registry_gap_count": 2, "validation_errors": [],
            "mapping_recommendations": [
                {"field": "current_valuation_amount", "recommendation": "map from 'Val GBP'",
                 "confidence": 0.62}],
            "run_state_path": "/tmp/orun_halt/run_state.json"})
        m = self._marker(
            "raw-v2/ERE/direct/funded/monthly/direct_001/2025-11-30/_READY.json", inv)
        self.assertEqual(m["status"], "halted")
        # actionable manifest
        na = m["next_action"]
        self.assertEqual(na["action"], ACT_FIX_MAPPING)
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

    def test_validation_halt_advises_fix_data_supply(self):
        self._seed_active()
        inv = DiagnosticInvoker(diagnostics={
            "halt_stage": "direct_001/validate",
            "halt_reason": "validation blocked",
            "blocking_decisions": [], "registry_gap_count": 0,
            "validation_errors": ["balance < 0 on 5 rows"],
            "run_state_path": "/tmp/x/run_state.json"})
        m = self._marker(
            "raw-v2/ERE/direct/funded/monthly/direct_001/2025-11-30/_READY.json", inv)
        self.assertEqual(m["next_action"]["action"], ACT_FIX_DATA)
        recs = OPS.recommendations(self.storage, self.layout, m["orchestrator_run_id"])
        self.assertEqual(recs["validation_issues"], ["balance < 0 on 5 rows"])

    # -- new source ------------------------------------------------------- #
    def test_new_source_ledger_and_manifest_action_is_approve(self):
        inv = RecordingInvoker(status="halted")
        m = self._marker(
            "raw-v2/ERE/acquired/funded/ad_hoc/acquired_001/2025-11-30/_READY.json", inv,
            meta={"acquisition_date": "2025-10-31", "seller_name": "BigBank"})
        self.assertEqual(m["event_decision"], "new_source_pending_review")
        self.assertEqual(m["next_action"]["action"], ACT_APPROVE)
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
        self.assertEqual(m["next_action"]["action"], ACT_APPROVE)
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
            self.assertEqual(OPS.main(["show", "does_not_exist"]), 1)
        finally:
            os.environ.pop("TRAKT_LOCAL_BLOB_ROOT", None)


if __name__ == "__main__":
    unittest.main()
