"""OCC-owned input readiness: batches, registration, recognition, readiness,
immutable run manifest, auto/manual start — and proof that the _READY.json
sentinel neither exists in the path nor triggers execution."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from operations_control.engine import OpsError

from .conftest import (
    OP_A,
    OP_B,
    make_client_config,
    make_engine,
    wait_for,
)


def _tape(dirpath: Path, name: str = "loan_tape_2026.csv",
          body: str = "loan_id,balance\nL1,100\n") -> Path:
    dirpath.mkdir(parents=True, exist_ok=True)
    p = dirpath / name
    p.write_text(body, encoding="utf-8")
    return p


def _mk(store, source_registry, tmp_path, **kw):
    return make_engine(store, source_registry,
                       client_config=make_client_config(tmp_path, valid=True),
                       **kw)


def _batch(engine, *, workflow_type="mi", auto=False, period="2026-06-30",
           portfolio="pf1"):
    return engine.create_batch(client_id="client_a", portfolio_id=portfolio,
                               reporting_date=period,
                               workflow_type=workflow_type,
                               created_by="test",
                               auto_start_when_ready=auto)


class TestBatchLifecycle:
    def test_create_is_idempotent_and_dates_isolated(self, store,
                                                     source_registry, tmp_path):
        engine = _mk(store, source_registry, tmp_path)
        a = _batch(engine)
        b = _batch(engine)
        assert a["batch_id"] == b["batch_id"]
        c = _batch(engine, period="2026-07-31")
        assert c["batch_id"] != a["batch_id"]

    def test_register_required_file_reaches_ready(self, store, source_registry,
                                                  tmp_path):
        engine = _mk(store, source_registry, tmp_path)
        batch = _batch(engine)
        f = _tape(tmp_path / "in")
        batch = engine.register_batch_file(client_id="client_a",
                                           batch_id=batch["batch_id"],
                                           source_path=str(f),
                                           received_by="alice")
        assert batch["status"] == "ready"
        assert batch["received_input_roles"] == ["loan_extract"]
        assert batch["effective_config_hash"].startswith("sha256:")
        # Optional roles absent does not block; audit trail intact.
        assert store.verify_audit_chain("client_a")
        events = {a["event"] for a in store.list_audit("client_a")}
        assert {"input_batch_created", "file_registered", "file_classified",
                "readiness_evaluated", "batch_ready"} <= events

    def test_missing_required_role_is_incomplete_with_plain_sentence(
            self, store, source_registry, tmp_path):
        engine = _mk(store, source_registry, tmp_path)
        batch = _batch(engine)
        f = _tape(tmp_path / "in", name="collateral_2026.csv",
                  body="prop_id,value\nP1,1\n")
        batch = engine.register_batch_file(client_id="client_a",
                                           batch_id=batch["batch_id"],
                                           source_path=str(f),
                                           received_by="alice")
        assert batch["status"] == "incomplete"
        assert "Primary loan tape" in batch["status_reason"]
        from operations_control import language
        assert language.is_operator_safe(batch["status_reason"])

    def test_duplicate_and_replacement_files(self, store, source_registry,
                                             tmp_path):
        engine = _mk(store, source_registry, tmp_path)
        batch = _batch(engine)
        f = _tape(tmp_path / "in")
        engine.register_batch_file(client_id="client_a",
                                   batch_id=batch["batch_id"],
                                   source_path=str(f), received_by="a")
        # Exact duplicate content → ignored, no second record.
        batch = engine.register_batch_file(client_id="client_a",
                                           batch_id=batch["batch_id"],
                                           source_path=str(f),
                                           received_by="a")
        current = [x for x in batch["files"]
                   if x["superseded_status"] == "current"]
        assert len(current) == 1
        # Replacement (same name, new content) supersedes the original.
        _tape(tmp_path / "in", body="loan_id,balance\nL1,999\n")
        batch = engine.register_batch_file(client_id="client_a",
                                           batch_id=batch["batch_id"],
                                           source_path=str(f),
                                           received_by="a")
        current = [x for x in batch["files"]
                   if x["superseded_status"] == "current"]
        superseded = [x for x in batch["files"]
                      if x["superseded_status"] == "superseded"]
        assert len(current) == 1 and len(superseded) == 1

    def test_unknown_file_does_not_satisfy_required_role(
            self, store, source_registry, tmp_path):
        engine = _mk(store, source_registry, tmp_path)
        batch = _batch(engine)
        f = _tape(tmp_path / "in", name="mystery.csv", body="a,b\n1,2\n")
        batch = engine.register_batch_file(client_id="client_a",
                                           batch_id=batch["batch_id"],
                                           source_path=str(f),
                                           received_by="a")
        assert batch["status"] == "incomplete"
        assert "loan_extract" in batch["missing_input_roles"]


class TestAmbiguityAndOverride:
    def test_role_conflict_blocks_with_file_role_decision(
            self, store, source_registry, tmp_path):
        engine = _mk(store, source_registry, tmp_path)
        batch = _batch(engine)
        d = tmp_path / "in"
        _tape(d, name="loan_tape_a.csv")
        _tape(d, name="loan_report_b.csv", body="loan_id,rate\nL1,0.05\n")
        batch = engine.register_batch_file(client_id="client_a",
                                           batch_id=batch["batch_id"],
                                           source_path=str(d),
                                           received_by="a")
        assert batch["status"] == "review_required"
        decs = [x for x in store.open_decisions("client_a",
                                                batch["batch_id"])
                if x["kind"] == "file_role"]
        assert decs, "ambiguous required role must raise a decision"
        from operations_control import language
        assert language.is_operator_safe(
            decs[0]["question"], allow=(decs[0]["source_file"],))

        # Operator identifies one file → batch becomes ready.
        out = engine.resolve_decision(
            client_id="client_a", decision_id=decs[0]["decision_id"],
            action="amend", actor="alice", value="property_extract",
            scope="file")
        assert out["decision"]["status"] == "approved"
        fresh = engine.intake.load_batch("client_a", batch["batch_id"])
        # One of the two files was overridden; remaining conflict resolved on
        # reassessment (other decisions may remain open if still ambiguous).
        assert fresh["status"] in ("ready", "review_required")
        overridden = [f for f in fresh["files"]
                      if f["recognition_status"] == "overridden"]
        assert overridden and overridden[0]["recognition_basis"] == \
            "operator_override"
        events = {a["event"] for a in store.list_audit("client_a")}
        assert "file_classification_overridden" in events


class TestConfigurationGate:
    def test_annex2_batch_blocked_on_configuration(self, store,
                                                   source_registry, tmp_path):
        engine = make_engine(store, source_registry,
                             client_config=make_client_config(tmp_path,
                                                              valid=False))
        batch = _batch(engine, workflow_type="mi_annex2")
        f = _tape(tmp_path / "in")
        batch = engine.register_batch_file(client_id="client_a",
                                           batch_id=batch["batch_id"],
                                           source_path=str(f),
                                           received_by="a")
        assert batch["status"] == "configuration_required"
        with pytest.raises(OpsError) as e:
            engine.start_batch(client_id="client_a",
                               batch_id=batch["batch_id"], actor="a")
        assert e.value.code == "OPS_BATCH_NOT_READY"


class TestStartAndManifest:
    def test_manual_start_runs_workflow_and_completes_batch(
            self, store, source_registry, tmp_path):
        engine = _mk(store, source_registry, tmp_path)
        batch = _batch(engine)
        f = _tape(tmp_path / "in")
        engine.register_batch_file(client_id="client_a",
                                   batch_id=batch["batch_id"],
                                   source_path=str(f), received_by="a")
        started = engine.start_batch(client_id="client_a",
                                     batch_id=batch["batch_id"], actor="a")
        assert started["status"] == "running" and started["workflow_id"]
        # Duplicate start → suppressed, same workflow.
        again = engine.start_batch(client_id="client_a",
                                   batch_id=batch["batch_id"], actor="a")
        assert again["workflow_id"] == started["workflow_id"]
        final = wait_for(lambda: (
            (lambda b: b if b and b["status"] == "completed" else None)(
                engine.intake.load_batch("client_a", batch["batch_id"]))))
        assert final["completed_at"]
        run = store.load_workflow("client_a", started["workflow_id"])
        assert run.batch_id == batch["batch_id"]

        # Manifest: immutable, hashed, idempotent run id.
        m = engine.intake.load_manifest("client_a", batch["batch_id"])
        assert m["run_id"].startswith("run_")
        assert all(x["sha256"].startswith("sha256:")
                   for x in m["input_files"])
        assert m["effective_configuration"]["content_hash"].startswith(
            "sha256:")
        events = {a["event"] for a in store.list_audit("client_a")}
        assert {"run_manifest_created", "onboarding_started"} <= events

    def test_auto_start_when_ready(self, store, source_registry, tmp_path):
        engine = _mk(store, source_registry, tmp_path)
        batch = _batch(engine, auto=True)
        f = _tape(tmp_path / "in")
        batch = engine.register_batch_file(client_id="client_a",
                                           batch_id=batch["batch_id"],
                                           source_path=str(f),
                                           received_by="blob-trigger")
        assert batch["status"] == "running" and batch["workflow_id"]

    def test_late_file_after_start_creates_new_version(
            self, store, source_registry, tmp_path):
        engine = _mk(store, source_registry, tmp_path)
        batch = _batch(engine)
        f = _tape(tmp_path / "in")
        engine.register_batch_file(client_id="client_a",
                                   batch_id=batch["batch_id"],
                                   source_path=str(f), received_by="a")
        engine.start_batch(client_id="client_a", batch_id=batch["batch_id"],
                           actor="a")
        late = _tape(tmp_path / "late", name="loan_tape_2026.csv",
                     body="loan_id,balance\nL9,5\n")
        started = engine.intake.load_batch("client_a", batch["batch_id"])
        newer = engine.intake.register_file(started, late,
                                            received_by_or_source="a")
        assert newer["batch_id"] != batch["batch_id"]
        assert newer["batch_id"].endswith("_v2")
        # The active batch's files were not mutated.
        active = engine.intake.load_batch("client_a", batch["batch_id"])
        assert all(x["sha256"] != newer["files"][0]["sha256"]
                   for x in active["files"])

    def test_manifest_conflict_requires_new_version(self, store,
                                                    source_registry, tmp_path):
        engine = _mk(store, source_registry, tmp_path)
        batch = _batch(engine)
        f = _tape(tmp_path / "in")
        engine.register_batch_file(client_id="client_a",
                                   batch_id=batch["batch_id"],
                                   source_path=str(f), received_by="a")
        fresh = engine.intake.load_batch("client_a", batch["batch_id"])
        m1 = engine.intake.ensure_manifest(
            fresh, effective_config={"effective_config_id": "e",
                                     "content_hash": "sha256:x"},
            approved_decisions=[], expected_outputs=["platform_canonical"])
        m2 = engine.intake.ensure_manifest(
            fresh, effective_config={"effective_config_id": "e",
                                     "content_hash": "sha256:x"},
            approved_decisions=[], expected_outputs=["platform_canonical"])
        assert m1["idempotency_key"] == m2["idempotency_key"]
        with pytest.raises(ValueError):
            engine.intake.ensure_manifest(
                fresh, effective_config={"effective_config_id": "e",
                                         "content_hash": "sha256:CHANGED"},
                approved_decisions=[], expected_outputs=[])

    def test_restart_reconstruction(self, store, storage, source_registry,
                                    tmp_path):
        from operations_control.stores import OpsLayout, OpsStore
        engine = _mk(store, source_registry, tmp_path)
        batch = _batch(engine)
        f = _tape(tmp_path / "in")
        engine.register_batch_file(client_id="client_a",
                                   batch_id=batch["batch_id"],
                                   source_path=str(f), received_by="a")
        fresh_store = OpsStore(storage, OpsLayout("operations-control"))
        engine2 = _mk(fresh_store, source_registry, tmp_path)
        again = engine2.intake.load_batch("client_a", batch["batch_id"])
        assert again is not None and again["status"] == "ready"


class TestTenancy:
    def test_cross_tenant_batch_access_denied(self, store, source_registry,
                                              tmp_path):
        from fastapi.testclient import TestClient
        from operations_control.api import app as app_module
        engine = _mk(store, source_registry, tmp_path)
        batch = _batch(engine)
        app_module.set_engine(engine)
        client = TestClient(app_module.app, raise_server_exceptions=False)
        try:
            r = client.get(f"/ops/batches/{batch['batch_id']}?client=client_a",
                           headers={"X-Operator-Token": OP_B})
            assert r.status_code == 404
            r = client.post(f"/ops/batches/{batch['batch_id']}/start"
                            "?client=client_a",
                            headers={"X-Operator-Token": OP_B})
            assert r.status_code == 404
            r = client.get(f"/ops/batches/{batch['batch_id']}?client=client_a",
                           headers={"X-Operator-Token": OP_A})
            assert r.status_code == 200
        finally:
            app_module.set_engine(None)


class TestSentinelRemoval:
    """_READY.json must never trigger execution anywhere in the new path."""

    def _bridge_engine(self, store, source_registry, tmp_path, monkeypatch):
        from apps.blob_trigger_app import occ_intake
        engine = _mk(store, source_registry, tmp_path)
        monkeypatch.setattr(occ_intake, "_engine", lambda: engine)
        return engine, occ_intake

    @staticmethod
    def _download_from(local_dir: Path):
        def _dl(container: str, path: str, dest_dir: Path) -> Path:
            name = path.rsplit("/", 1)[-1]
            dest = dest_dir / name
            dest.write_bytes((local_dir / name).read_bytes())
            return dest
        return _dl

    def test_sentinel_arrival_is_ignored_and_audited(
            self, store, source_registry, tmp_path, monkeypatch):
        engine, occ_intake = self._bridge_engine(store, source_registry,
                                                 tmp_path, monkeypatch)
        result = occ_intake.handle_arrival(
            "raw-v2",
            "client_a/direct/funded/monthly/pf1/2026-06-30/_READY.json",
            download=self._download_from(tmp_path))
        assert result["registered"] is False
        assert result["reason"] == "legacy_sentinel_ignored"
        # No workflow started, nothing ran.
        assert store.list_workflows("client_a") == []
        events = [a for a in store.list_audit("client_a")
                  if a["event"] == "legacy_sentinel_ignored"]
        assert events

    def test_source_files_alone_start_the_workflow(
            self, store, source_registry, tmp_path, monkeypatch):
        engine, occ_intake = self._bridge_engine(store, source_registry,
                                                 tmp_path, monkeypatch)
        src = tmp_path / "blobs"
        _tape(src, name="loan_tape_2026.csv")
        result = occ_intake.handle_arrival(
            "raw-v2",
            "client_a/direct/funded/monthly/pf1/2026-06-30/loan_tape_2026.csv",
            download=self._download_from(src))
        assert result["registered"] is True
        assert result["status"] == "running"
        assert result["workflow_id"]
        final = wait_for(lambda: (
            (lambda b: b if b and b["status"] == "completed" else None)(
                engine.intake.load_batch("client_a", result["batch_id"]))))
        assert final["workflow_id"] == result["workflow_id"]

    def test_no_production_sentinel_consumer_remains(self):
        """The deployed entrypoint must not gate on the sentinel."""
        src = Path("function_app.py").read_text(encoding="utf-8")
        assert "PACK_MARKER" not in src
        assert "handle_blob_event" not in src
        assert "occ_intake" in src