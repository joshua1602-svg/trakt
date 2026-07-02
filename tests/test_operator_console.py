#!/usr/bin/env python3
"""tests/test_operator_console.py

The standalone Operator console: the approval service (queue / detail / approve /
reject / choose-alternative / auto-approval audit) and the FAIL-CLOSED server-side
auth gate. Proves approving from the console promotes the source to ACTIVE exactly
like the CLI, and that the API refuses without a valid operator token.

Run: python -m unittest tests.test_operator_console
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from apps.blob_trigger_app import router as R
from apps.blob_trigger_app.layout import Layout
from apps.blob_trigger_app.persistence import ProductionPersistence
from apps.blob_trigger_app.repin import repin_source
from apps.blob_trigger_app.schema_fingerprint import fingerprint_pack
from apps.blob_trigger_app.source_registry import SourceRegistry, STATUS_ACTIVE
from apps.blob_trigger_app.storage import Storage
from mi_agent_operator.service import OperatorService

_LOAN = ["loan_id", "balance", "rate", "origination_date", "maturity_date"]


class _Inv:
    def __call__(self, **kw):
        return {"run_id": "orun", "status": "done", "central_canonical_path": None, "blockers": []}


def _stub_assembler(**kw):
    return {}


def _write_pack(d, cols, name="PipelineExtract.csv"):
    d.mkdir(parents=True, exist_ok=True)
    (d / name).write_text(",".join(cols) + "\n" + ",".join(["x"] * len(cols)) + "\n")
    return [str(d / name)]


def _ctx(td):
    storage = Storage(Path(td))
    layout = Layout()
    persistence = ProductionPersistence(storage, layout)
    registry = SourceRegistry("blob://trakt-state/registry/source_registry.yaml", storage=storage)
    return storage, layout, persistence, registry


def _route(registry, persistence, td, marker, pack, meta=None):
    role_schemas = R.role_schemas_for_pack(registry, marker, "raw-v2")
    schema = fingerprint_pack(pack, role_schemas=role_schemas)
    return R.handle_blob_event(
        marker, registry=registry, out_dir=td, container="raw-v2",
        pack_marker="_READY.json", schema_info=schema,
        input_dir_override=str(Path(pack[0]).parent),
        pack_files=[Path(p).name for p in pack], orchestrator_invoker=_Inv(),
        assembler_refresher=_stub_assembler, persistence=persistence,
        marker_metadata=meta, now="2026-01-01T00:00:00+00:00")


class TestOperatorService(unittest.TestCase):

    def test_queue_approve_promotes_to_active(self):
        with tempfile.TemporaryDirectory() as td:
            storage, layout, persistence, registry = _ctx(td)
            pack = _write_pack(Path(td) / "wk", _LOAN)
            marker = "raw-v2/ERE/direct/pipeline/weekly/direct_001/2026-W02/_READY.json"
            m = _route(registry, persistence, td, marker, pack)
            self.assertEqual(m["status"], "pending_review")

            svc = OperatorService(storage, layout, persistence, registry)
            q = svc.queue()
            self.assertEqual(len(q), 1)
            self.assertEqual(q[0]["kind"], "new_source")
            aid = q[0]["approval_id"]

            detail = svc.item(aid)
            self.assertIsNotNone(detail)
            self.assertIn("PipelineExtract.csv", detail["detected_files"])

            res = svc.approve(aid, decided_by="alice")
            self.assertEqual(res["status"], "promoted")
            rec = registry.lookup("ERE", "direct_001", "pipeline", "weekly")
            self.assertEqual(rec.status, STATUS_ACTIVE)
            self.assertTrue(rec.has_approved_mapping)
            self.assertEqual(svc.queue(), [])          # cleared from the queue

    def test_reject_sets_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            storage, layout, persistence, registry = _ctx(td)
            pack = _write_pack(Path(td) / "wk", _LOAN)
            marker = "raw-v2/ERE/direct/pipeline/weekly/direct_001/2026-W02/_READY.json"
            aid = _route(registry, persistence, td, marker, pack)["approval_id"]
            svc = OperatorService(storage, layout, persistence, registry)
            out = svc.reject(aid, reason="wrong portfolio id", decided_by="bob")
            self.assertEqual(out["status"], "rejected")
            self.assertEqual(svc.queue(), [])          # no longer pending

    def test_edit_choose_alternative_then_approve_uses_it(self):
        with tempfile.TemporaryDirectory() as td:
            storage, layout, persistence, registry = _ctx(td)
            pack = _write_pack(Path(td) / "wk", _LOAN)
            marker = "raw-v2/ERE/direct/pipeline/weekly/direct_001/2026-W02/_READY.json"
            aid = _route(registry, persistence, td, marker, pack)["approval_id"]
            svc = OperatorService(storage, layout, persistence, registry)
            svc.edit(aid, {"suggested_mapping_config_path": "config/client/mappings/alt.yaml"})
            svc.approve(aid, decided_by="alice")
            rec = registry.lookup("ERE", "direct_001", "pipeline", "weekly")
            self.assertEqual(rec.mapping_config_path, "config/client/mappings/alt.yaml")

    def test_edit_rejects_non_editable_field(self):
        with tempfile.TemporaryDirectory() as td:
            storage, layout, persistence, registry = _ctx(td)
            pack = _write_pack(Path(td) / "wk", _LOAN)
            marker = "raw-v2/ERE/direct/pipeline/weekly/direct_001/2026-W02/_READY.json"
            aid = _route(registry, persistence, td, marker, pack)["approval_id"]
            svc = OperatorService(storage, layout, persistence, registry)
            with self.assertRaises(ValueError):
                svc.edit(aid, {"schema_fingerprint": "sha256:tampered"})

    def test_audit_lists_auto_approvals(self):
        with tempfile.TemporaryDirectory() as td:
            storage, layout, persistence, registry = _ctx(td)
            old = _write_pack(Path(td) / "wk1", _LOAN)
            repin_source(registry, client_id="ERE", source_portfolio_id="direct_001",
                         dataset="pipeline", frequency="weekly", data_files=old,
                         source_book_type="direct", regime_required=False)
            new = _write_pack(Path(td) / "wk2",
                              ["Loan ID", "Balance", "Rate", "Origination Date", "Maturity Date"])
            marker = "raw-v2/ERE/direct/pipeline/weekly/direct_001/2026-W03/_READY.json"
            m = _route(registry, persistence, td, marker, new)
            self.assertTrue(m.get("auto_approved"))
            svc = OperatorService(storage, layout, persistence, registry)
            audit = svc.audit()
            self.assertEqual(len(audit), 1)
            self.assertEqual(audit[0]["new_fingerprint"], m["schema_fingerprint"])
            self.assertEqual(svc.queue(), [])          # auto-approvals never queue


class TestOperatorAuth(unittest.TestCase):

    def setUp(self):
        from fastapi.testclient import TestClient
        import mi_agent_operator.operator_app as app_mod
        self._saved = {k: os.environ.get(k) for k in
                       ("TRAKT_OPERATOR_TOKEN", "TRAKT_STORAGE_BACKEND", "TRAKT_LOCAL_BLOB_ROOT")}
        self._tmp = tempfile.TemporaryDirectory()
        os.environ["TRAKT_STORAGE_BACKEND"] = "file"
        os.environ["TRAKT_LOCAL_BLOB_ROOT"] = self._tmp.name
        self.client = TestClient(app_mod.app)

    def tearDown(self):
        for k, v in self._saved.items():
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)
        self._tmp.cleanup()

    def test_503_when_token_not_configured(self):
        os.environ.pop("TRAKT_OPERATOR_TOKEN", None)
        self.assertEqual(self.client.get("/api/queue").status_code, 503)

    def test_401_on_wrong_token(self):
        os.environ["TRAKT_OPERATOR_TOKEN"] = "secret"
        r = self.client.get("/api/queue", headers={"X-Operator-Token": "nope"})
        self.assertEqual(r.status_code, 401)

    def test_200_with_valid_token(self):
        os.environ["TRAKT_OPERATOR_TOKEN"] = "secret"
        r = self.client.get("/api/queue", headers={"X-Operator-Token": "secret"})
        self.assertEqual(r.status_code, 200)
        self.assertIn("items", r.json())

    def test_ui_served_without_token(self):
        r = self.client.get("/")
        self.assertEqual(r.status_code, 200)
        self.assertIn("Operator Console", r.text)


if __name__ == "__main__":
    unittest.main()
