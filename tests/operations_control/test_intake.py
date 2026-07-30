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

class TestDatasetRouting:
    """Which workflow an automated arrival gets, and which pack it joins.

    The business rules these encode:
      * the FUNDED book — direct or acquired — is regime-reportable when the
        registry says so; acquired is not a second-class book;
      * an acquired book is registered at the frequency of its first delivery and
        reported monthly thereafter, so frequency must not decide scope;
      * PIPELINE is strictly an MI view and is never regime-reportable;
      * a pipeline delivery is its own input pack, not part of the funded one.

    Before this, the registry lookup was hardcoded to ("funded", "monthly"), so an
    acquired funded book registered ad hoc silently lost its Annex 2 delivery.
    """

    def _bridge(self, store, source_registry, tmp_path, monkeypatch):
        from apps.blob_trigger_app import occ_intake
        engine = _mk(store, source_registry, tmp_path)
        monkeypatch.setattr(occ_intake, "_engine", lambda: engine)
        return engine, occ_intake

    @staticmethod
    def _record(portfolio, dataset, frequency, regime):
        from apps.blob_trigger_app.source_registry import SourceRecord
        return SourceRecord(client_id="client_a", source_portfolio_id=portfolio,
                            dataset=dataset, frequency=frequency,
                            regime_required=regime)

    # -- funded: direct and acquired are treated alike -------------------- #

    @pytest.mark.parametrize("book,portfolio", [("direct", "direct_001"),
                                                ("acquired", "acquired_001")])
    def test_regime_required_funded_book_gets_annex2(
            self, store, source_registry, tmp_path, monkeypatch, book, portfolio):
        engine, occ_intake = self._bridge(store, source_registry, tmp_path,
                                          monkeypatch)
        source_registry.upsert(self._record(portfolio, "funded", "monthly", True))
        assert occ_intake._outcome_for(engine, "client_a", portfolio,
                                       "funded") == "mi_annex2"

    def test_acquired_registered_ad_hoc_keeps_annex2_when_reported_monthly(
            self, store, source_registry, tmp_path, monkeypatch):
        """THE REGRESSION. Registered on its first (ad hoc) delivery, reported
        monthly afterwards — scope must not depend on the frequency."""
        engine, occ_intake = self._bridge(store, source_registry, tmp_path,
                                          monkeypatch)
        source_registry.upsert(
            self._record("acquired_001", "funded", "ad_hoc", True))
        # The monthly delivery that follows must still be Annex 2.
        assert occ_intake._outcome_for(engine, "client_a", "acquired_001",
                                       "funded") == "mi_annex2"

    def test_funded_book_not_flagged_is_mi(self, store, source_registry,
                                           tmp_path, monkeypatch):
        engine, occ_intake = self._bridge(store, source_registry, tmp_path,
                                          monkeypatch)
        source_registry.upsert(
            self._record("direct_001", "funded", "monthly", False))
        assert occ_intake._outcome_for(engine, "client_a", "direct_001",
                                       "funded") == "mi"

    def test_unknown_portfolio_falls_back_to_mi(self, store, source_registry,
                                                tmp_path, monkeypatch):
        engine, occ_intake = self._bridge(store, source_registry, tmp_path,
                                          monkeypatch)
        assert occ_intake._outcome_for(engine, "client_a", "unknown_001",
                                       "funded") == "mi"

    # -- pipeline is strictly MI ------------------------------------------ #

    def test_pipeline_is_mi_even_when_the_registry_flags_regime(
            self, store, source_registry, tmp_path, monkeypatch):
        """No registry flag may pull a pipeline view into regime reporting."""
        engine, occ_intake = self._bridge(store, source_registry, tmp_path,
                                          monkeypatch)
        source_registry.upsert(
            self._record("direct_001", "pipeline", "weekly", True))
        source_registry.upsert(
            self._record("direct_001", "funded", "monthly", True))
        assert occ_intake._outcome_for(engine, "client_a", "direct_001",
                                       "pipeline") == "mi"

    def test_forecast_is_also_mi_only(self, store, source_registry, tmp_path,
                                      monkeypatch):
        engine, occ_intake = self._bridge(store, source_registry, tmp_path,
                                          monkeypatch)
        source_registry.upsert(
            self._record("direct_001", "funded", "monthly", True))
        assert occ_intake._outcome_for(engine, "client_a", "direct_001",
                                       "forecast") == "mi"

    # -- pack identity ------------------------------------------------------ #

    def test_funded_pack_id_is_unchanged_by_the_new_parameter(
            self, store, source_registry, tmp_path):
        """No existing pack may be re-keyed: packs part-way through collection
        must not be stranded by this change."""
        engine = _mk(store, source_registry, tmp_path)
        args = dict(client_id="client_a", portfolio_id="pf1",
                    reporting_date="2026-06-30", workflow_type="mi")
        before = engine.intake.deterministic_batch_id(**args)
        assert engine.intake.deterministic_batch_id(**args, dataset="") == before
        assert engine.intake.deterministic_batch_id(
            **args, dataset="funded") == before

    def test_pipeline_gets_its_own_pack(self, store, source_registry, tmp_path):
        engine = _mk(store, source_registry, tmp_path)
        args = dict(client_id="client_a", portfolio_id="pf1",
                    reporting_date="2026-06-30", workflow_type="mi")
        funded = engine.intake.deterministic_batch_id(**args, dataset="funded")
        pipeline = engine.intake.deterministic_batch_id(**args,
                                                        dataset="pipeline")
        assert pipeline != funded

    def test_pipeline_and_funded_same_period_do_not_share_a_pack(
            self, store, source_registry, tmp_path):
        """Same portfolio, same period, both resolving to MI: previously one pack,
        so a pipeline file was assessed as part of the funded delivery."""
        engine = _mk(store, source_registry, tmp_path)
        common = dict(client_id="client_a", portfolio_id="pf1",
                      reporting_date="2026-06-30", workflow_type="mi",
                      created_by="test")
        funded = engine.create_batch(**common, dataset="funded")
        pipeline = engine.create_batch(**common, dataset="pipeline")
        assert funded["batch_id"] != pipeline["batch_id"]
        assert funded["dataset"] == "funded"
        assert pipeline["dataset"] == "pipeline"

    def test_batch_created_without_a_dataset_records_the_funded_default(
            self, store, source_registry, tmp_path):
        engine = _mk(store, source_registry, tmp_path)
        batch = _batch(engine)
        assert batch["dataset"] == "funded"

    # -- end to end through the real trigger path --------------------------- #

    @staticmethod
    def _download_from(local_dir: Path):
        def _dl(container: str, path: str, dest_dir: Path) -> Path:
            name = path.rsplit("/", 1)[-1]
            dest = dest_dir / name
            dest.write_bytes((local_dir / name).read_bytes())
            return dest
        return _dl

    @pytest.mark.parametrize("blob_path,portfolio,expect_dataset", [
        ("client_a/direct/funded/monthly/pf1/2026-06-30/loan_tape_2026.csv",
         "pf1", "funded"),
        ("client_a/acquired/funded/ad_hoc/pf1/2026-06-30/loan_tape_2026.csv",
         "pf1", "funded"),
        ("client_a/direct/pipeline/weekly/pf1/2026-06-30/loan_tape_2026.csv",
         "pf1", "pipeline"),
    ])
    def test_all_three_categories_register(
            self, store, source_registry, tmp_path, monkeypatch,
            blob_path, portfolio, expect_dataset):
        """Pipeline, Funded–Direct and Funded–Acquired all reach OCC intake.
        Only Funded–Direct–monthly was covered before."""
        engine, occ_intake = self._bridge(store, source_registry, tmp_path,
                                          monkeypatch)
        src = tmp_path / "blobs"
        _tape(src, name="loan_tape_2026.csv")
        result = occ_intake.handle_arrival(
            "raw-v2", blob_path, download=self._download_from(src))
        assert result["registered"] is True
        batch = engine.intake.load_batch("client_a", result["batch_id"])
        assert batch["dataset"] == expect_dataset

    def test_pipeline_and_funded_arrivals_land_in_different_packs(
            self, store, source_registry, tmp_path, monkeypatch):
        engine, occ_intake = self._bridge(store, source_registry, tmp_path,
                                          monkeypatch)
        src = tmp_path / "blobs"
        _tape(src, name="loan_tape_2026.csv")
        funded = occ_intake.handle_arrival(
            "raw-v2",
            "client_a/direct/funded/monthly/pf1/2026-06-30/loan_tape_2026.csv",
            download=self._download_from(src))
        pipeline = occ_intake.handle_arrival(
            "raw-v2",
            "client_a/direct/pipeline/weekly/pf1/2026-06-30/loan_tape_2026.csv",
            download=self._download_from(src))
        assert funded["batch_id"] != pipeline["batch_id"]

    def test_late_pipeline_file_stays_in_the_pipeline_pack(
            self, store, source_registry, tmp_path, monkeypatch):
        """A file arriving after the run started opens a successor pack. That
        successor must stay in the pipeline key space, not fall back to funded."""
        engine, occ_intake = self._bridge(store, source_registry, tmp_path,
                                          monkeypatch)
        src = tmp_path / "blobs"
        _tape(src, name="loan_tape_2026.csv")
        first = occ_intake.handle_arrival(
            "raw-v2",
            "client_a/direct/pipeline/weekly/pf1/2026-06-30/loan_tape_2026.csv",
            download=self._download_from(src))
        batch = engine.intake.load_batch("client_a", first["batch_id"])
        assert batch["dataset"] == "pipeline"

        # Force the started-batch path, then register a late file directly.
        batch["status"] = "running"
        engine.intake.save_batch(batch)
        _tape(src, name="late_addition.csv")
        successor = engine.intake.register_file(
            batch, src / "late_addition.csv", received_by_or_source="test")
        assert successor["batch_id"] != batch["batch_id"]
        assert successor["dataset"] == "pipeline"

        # And it must not collide with the funded pack for the same period.
        funded = engine.intake.deterministic_batch_id(
            client_id="client_a", portfolio_id="pf1",
            reporting_date="2026-06-30",
            workflow_type=batch["workflow_type"], dataset="funded")
        assert not successor["batch_id"].startswith(funded)


class TestManualDatasetSelection:
    """Operators choose the book when they open a pack by hand.

    Automated arrivals get the dataset from the blob path. The OCC is the other
    door into intake, and it defaulted every manual pack to the funded book —
    so a pipeline delivery opened by hand joined the funded pack. The selection
    closes that, and the engine refuses combinations that would route a delivery
    into the wrong reporting.
    """

    def _api(self, store, source_registry, tmp_path):
        from fastapi.testclient import TestClient
        from operations_control.api import app as app_module
        engine = _mk(store, source_registry, tmp_path)
        app_module.set_engine(engine)
        return engine, app_module, TestClient(app_module.app,
                                              raise_server_exceptions=False)

    @staticmethod
    def _body(**kw):
        body = {"client_id": "client_a", "portfolio_id": "pf1",
                "reporting_date": "2026-06-30", "workflow_type": "mi"}
        body.update(kw)
        return body

    # -- engine-level rules ------------------------------------------------- #

    def test_unknown_dataset_is_refused(self, store, source_registry, tmp_path):
        engine = _mk(store, source_registry, tmp_path)
        with pytest.raises(OpsError) as exc:
            engine.create_batch(client_id="client_a", portfolio_id="pf1",
                                reporting_date="2026-06-30", workflow_type="mi",
                                created_by="test", dataset="nonsense")
        assert exc.value.code == "OPS_BAD_DATASET"

    def test_pipeline_cannot_carry_a_regime_delivery(self, store,
                                                     source_registry, tmp_path):
        """The rule the blob trigger applies, enforced at the other door too."""
        engine = _mk(store, source_registry, tmp_path)
        with pytest.raises(OpsError) as exc:
            engine.create_batch(client_id="client_a", portfolio_id="pf1",
                                reporting_date="2026-06-30",
                                workflow_type="mi_annex2",
                                created_by="test", dataset="pipeline")
        assert exc.value.code == "OPS_DATASET_NOT_REGIME_CAPABLE"

    def test_funded_may_carry_a_regime_delivery(self, store, source_registry,
                                                tmp_path):
        engine = _mk(store, source_registry, tmp_path)
        batch = engine.create_batch(client_id="client_a", portfolio_id="pf1",
                                    reporting_date="2026-06-30",
                                    workflow_type="mi_annex2",
                                    created_by="test", dataset="funded")
        assert batch["dataset"] == "funded"

    # -- API ---------------------------------------------------------------- #

    def test_api_accepts_a_dataset_and_records_it(self, store, source_registry,
                                                  tmp_path):
        engine, app_module, client = self._api(store, source_registry, tmp_path)
        try:
            r = client.post("/ops/batches", json=self._body(dataset="pipeline"),
                            headers={"X-Operator-Token": OP_A})
            assert r.status_code == 201, r.text
            batch = r.json()["batch"]
            assert batch["dataset"] == "pipeline"
            assert batch["dataset_label"] == "Pipeline"
        finally:
            app_module.set_engine(None)

    def test_api_without_a_dataset_still_works_and_means_funded(
            self, store, source_registry, tmp_path):
        """An older client that does not send the field must behave as before."""
        engine, app_module, client = self._api(store, source_registry, tmp_path)
        try:
            r = client.post("/ops/batches", json=self._body(),
                            headers={"X-Operator-Token": OP_A})
            assert r.status_code == 201, r.text
            assert r.json()["batch"]["dataset"] == "funded"
        finally:
            app_module.set_engine(None)

    def test_api_refuses_pipeline_with_regime_and_explains_plainly(
            self, store, source_registry, tmp_path):
        engine, app_module, client = self._api(store, source_registry, tmp_path)
        try:
            r = client.post("/ops/batches",
                            json=self._body(workflow_type="mi_annex2",
                                            dataset="pipeline"),
                            headers={"X-Operator-Token": OP_A})
            assert r.status_code == 400
            body = r.json()
            assert body["ok"] is False
            assert body["errorCode"] == "OPS_DATASET_NOT_REGIME_CAPABLE"
            # Operator-facing: a plain sentence, no identifiers, no stack trace.
            assert "funded book" in body["message"]
            assert "pipeline" not in body["message"].lower()
        finally:
            app_module.set_engine(None)

    def test_api_refuses_an_unknown_dataset(self, store, source_registry,
                                            tmp_path):
        engine, app_module, client = self._api(store, source_registry, tmp_path)
        try:
            r = client.post("/ops/batches", json=self._body(dataset="nonsense"),
                            headers={"X-Operator-Token": OP_A})
            assert r.status_code == 400
            assert r.json()["errorCode"] == "OPS_BAD_DATASET"
        finally:
            app_module.set_engine(None)

    def test_manual_pipeline_and_funded_packs_are_separate(
            self, store, source_registry, tmp_path):
        engine, app_module, client = self._api(store, source_registry, tmp_path)
        try:
            funded = client.post("/ops/batches", json=self._body(dataset="funded"),
                                 headers={"X-Operator-Token": OP_A}).json()["batch"]
            pipeline = client.post("/ops/batches",
                                   json=self._body(dataset="pipeline"),
                                   headers={"X-Operator-Token": OP_A}).json()["batch"]
            assert funded["batch_id"] != pipeline["batch_id"]
        finally:
            app_module.set_engine(None)

    def test_a_pack_created_before_this_change_reads_as_funded(
            self, store, source_registry, tmp_path):
        """Packs already on disk have no dataset field; they must not read blank."""
        from operations_control.api import presenters
        presented = presenters.present_batch(
            {"batch_id": "b0", "status": "receiving"}, {})
        assert presented["dataset"] == "funded"
        assert presented["dataset_label"] == "Funded book"
