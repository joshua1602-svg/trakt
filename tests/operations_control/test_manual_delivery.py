"""The manual delivery route: an operator uploads files, and the SAME governed
intake path the blob trigger uses takes over.

What is proven here:

  * the destination is derived server-side from controlled fields, and a path
    the production parser would reject is never written to;
  * the browser cannot name a location — the deprecated server-path route is
    fail-closed and administrator-only;
  * every source file is placed and registered BEFORE the internal run manifest
    (the governed replacement for the ``_READY.json`` sentinel) exists;
  * ``_READY.json`` itself is refused outright;
  * re-uploading the same pack does not create a second batch or a second run;
  * one client's operator cannot upload into another client's pack.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from operations_control import manual_intake
from operations_control.api import app as app_module
from operations_control.engine import OpsError

from .conftest import OP_A, OP_ADMIN, OP_B, make_client_config, make_engine


TAPE = b"loan_id,balance\nL1,100\n"
COLLATERAL = b"prop_id,value\nP1,1\n"


def _mk(store, source_registry, tmp_path):
    return make_engine(store, source_registry,
                       client_config=make_client_config(tmp_path, valid=True))


def _batch(engine, *, client="client_a", portfolio="direct_001",
           period="2026-06-30", auto=True, workflow_type="mi"):
    return engine.create_batch(client_id=client, portfolio_id=portfolio,
                               reporting_date=period,
                               workflow_type=workflow_type,
                               created_by="alice", auto_start_when_ready=auto)


def _h(token: str):
    return {"X-Operator-Token": token}


# --------------------------------------------------------------------------- #
# Destination derivation
# --------------------------------------------------------------------------- #

class TestDerivedDestination:
    def test_prefix_follows_the_production_convention(self):
        assert manual_intake.derive_raw_prefix(
            client_id="ERE", portfolio_id="direct_001",
            reporting_period="2025-10-31", container="raw-v2"
        ) == "raw-v2/ERE/direct/funded/monthly/direct_001/2025-10-31"

    def test_book_type_follows_the_portfolio_not_the_caller(self):
        assert manual_intake.derive_raw_prefix(
            client_id="ERE", portfolio_id="acquired_002",
            reporting_period="2025-10", dataset="pipeline",
            frequency="weekly", container="raw-v2"
        ) == "raw-v2/ERE/acquired/pipeline/weekly/acquired_002/2025-10"

    @pytest.mark.parametrize("field,value", [
        ("client_id", "../secrets"),
        ("client_id", "a/b"),
        ("portfolio_id", ".."),
        ("reporting_period", "not-a-period"),
        ("reporting_period", "../../etc"),
    ])
    def test_a_segment_that_is_not_an_identifier_is_refused(self, field, value):
        kwargs = {"client_id": "ERE", "portfolio_id": "direct_001",
                  "reporting_period": "2025-10-31", "container": "raw-v2"}
        kwargs[field] = value
        with pytest.raises(manual_intake.ManualIntakeError):
            manual_intake.derive_raw_prefix(**kwargs)

    def test_an_unknown_book_or_frequency_is_refused(self):
        with pytest.raises(manual_intake.ManualIntakeError):
            manual_intake.derive_raw_prefix(
                client_id="ERE", portfolio_id="direct_001",
                reporting_period="2025-10-31", dataset="anything")
        with pytest.raises(manual_intake.ManualIntakeError):
            manual_intake.derive_raw_prefix(
                client_id="ERE", portfolio_id="direct_001",
                reporting_period="2025-10-31", frequency="hourly")

    @pytest.mark.parametrize("name,leaf", [
        ("loan_tape.csv", "loan_tape.csv"),
        ("../../etc/passwd.csv", "passwd.csv"),
        ("C:\\Windows\\tape.xlsx", "tape.xlsx"),
        ("sub/dir/tape.xlsx", "tape.xlsx"),
    ])
    def test_only_the_leaf_of_a_filename_survives(self, name, leaf):
        assert manual_intake.sanitise_filename(name) == leaf

    @pytest.mark.parametrize("name", [
        "_READY.json", "", "..", "tape.exe", "tape", "tape\n.csv",
    ])
    def test_a_name_trakt_cannot_accept_is_refused(self, name):
        with pytest.raises(manual_intake.ManualIntakeError):
            manual_intake.sanitise_filename(name)


# --------------------------------------------------------------------------- #
# Upload through the engine
# --------------------------------------------------------------------------- #

class TestUploadOrdering:
    def test_files_are_placed_and_registered_before_the_manifest_exists(
            self, store, source_registry, tmp_path, monkeypatch):
        engine = _mk(store, source_registry, tmp_path)
        batch = _batch(engine)

        seen: list[str] = []
        real_write = store.storage.write_bytes
        real_manifest = engine.intake.ensure_manifest

        def spy_write(uri, data):
            seen.append(f"placed:{uri.rsplit('/', 1)[-1]}")
            return real_write(uri, data)

        def spy_manifest(*a, **kw):
            seen.append("manifest")
            return real_manifest(*a, **kw)

        monkeypatch.setattr(store.storage, "write_bytes", spy_write)
        monkeypatch.setattr(engine.intake, "ensure_manifest", spy_manifest)

        engine.upload_batch_files(
            client_id="client_a", batch_id=batch["batch_id"],
            uploads=[("loan_tape_2026.csv", TAPE),
                     ("collateral_2026.csv", COLLATERAL)],
            received_by="alice")

        placed = [i for i, s in enumerate(seen) if s.startswith("placed:")]
        assert "manifest" in seen, "the run manifest was never written"
        assert max(placed) < seen.index("manifest"), (
            "a source file was placed after the readiness manifest — the "
            "manifest must be the last thing written")
        assert "placed:loan_tape_2026.csv" in seen
        assert "placed:collateral_2026.csv" in seen

    def test_the_pack_reaches_a_real_workflow_through_the_existing_intake(
            self, store, source_registry, tmp_path):
        engine = _mk(store, source_registry, tmp_path)
        batch = _batch(engine)
        result = engine.upload_batch_files(
            client_id="client_a", batch_id=batch["batch_id"],
            uploads=[("loan_tape_2026.csv", TAPE)], received_by="alice")

        assert result["workflow_id"], "no workflow was opened for a ready pack"
        run = store.load_workflow("client_a", result["workflow_id"])
        assert run is not None and run.batch_id == batch["batch_id"]
        # The manifest the intake path writes is the governed readiness record.
        manifest = engine.intake.load_manifest("client_a", batch["batch_id"])
        assert manifest["input_files"][0]["original_filename"] == \
            "loan_tape_2026.csv"

    def test_the_source_location_is_recorded_not_supplied(
            self, store, source_registry, tmp_path):
        engine = _mk(store, source_registry, tmp_path)
        batch = _batch(engine, auto=False)
        result = engine.upload_batch_files(
            client_id="client_a", batch_id=batch["batch_id"],
            uploads=[("loan_tape_2026.csv", TAPE)], received_by="alice")
        assert result["source_prefix"].endswith(
            "client_a/direct/funded/monthly/direct_001/2026-06-30")
        assert result["files"][0]["source_uri"].endswith("loan_tape_2026.csv")

    def test_a_refused_name_places_nothing_at_all(self, store, source_registry,
                                                  tmp_path):
        engine = _mk(store, source_registry, tmp_path)
        batch = _batch(engine)
        with pytest.raises(OpsError) as exc:
            engine.upload_batch_files(
                client_id="client_a", batch_id=batch["batch_id"],
                uploads=[("loan_tape_2026.csv", TAPE), ("_READY.json", b"{}")],
                received_by="alice")
        assert exc.value.code == "OPS_UPLOAD_REFUSED"
        reloaded = engine.intake.load_batch("client_a", batch["batch_id"])
        assert reloaded["files"] == [], (
            "a rejected upload must not leave half a pack behind")

    def test_an_empty_upload_is_refused(self, store, source_registry, tmp_path):
        engine = _mk(store, source_registry, tmp_path)
        batch = _batch(engine)
        with pytest.raises(OpsError):
            engine.upload_batch_files(client_id="client_a",
                                      batch_id=batch["batch_id"],
                                      uploads=[], received_by="alice")


class TestDuplicates:
    def test_the_same_context_reuses_one_pack(self, store, source_registry,
                                              tmp_path):
        engine = _mk(store, source_registry, tmp_path)
        assert _batch(engine)["batch_id"] == _batch(engine)["batch_id"]

    def test_identical_content_is_not_registered_twice(self, store,
                                                       source_registry,
                                                       tmp_path):
        engine = _mk(store, source_registry, tmp_path)
        batch = _batch(engine, auto=False)
        engine.upload_batch_files(
            client_id="client_a", batch_id=batch["batch_id"],
            uploads=[("loan_tape_2026.csv", TAPE)], received_by="alice")
        result = engine.upload_batch_files(
            client_id="client_a", batch_id=batch["batch_id"],
            uploads=[("loan_tape_2026.csv", TAPE)], received_by="alice")
        assert len(result["files"]) == 1, "the same file was taken twice"

    def test_a_started_pack_refuses_further_uploads(self, store,
                                                    source_registry, tmp_path):
        engine = _mk(store, source_registry, tmp_path)
        batch = _batch(engine)
        started = engine.upload_batch_files(
            client_id="client_a", batch_id=batch["batch_id"],
            uploads=[("loan_tape_2026.csv", TAPE)], received_by="alice")
        assert started["workflow_id"]
        with pytest.raises(OpsError) as exc:
            engine.upload_batch_files(
                client_id="client_a", batch_id=batch["batch_id"],
                uploads=[("loan_tape_2026.csv", COLLATERAL)],
                received_by="alice")
        assert exc.value.code == "OPS_BATCH_ALREADY_STARTED"


class TestBothDoorsAgreeOnTheDelivery:
    """A manual delivery is filed where an automated one would be, so the
    trigger sees it too. If the two routes would call it different deliveries
    the pack splits in half — so the upload is refused before anything is
    written, not reconciled afterwards."""

    @staticmethod
    def _register(source_registry, portfolio, *, regime):
        from apps.blob_trigger_app.source_registry import SourceRecord
        source_registry.upsert(SourceRecord(
            client_id="client_a", source_portfolio_id=portfolio,
            dataset="funded", frequency="monthly", regime_required=regime))

    def test_the_two_routes_derive_the_same_identity(self, store,
                                                     source_registry, tmp_path):
        engine = _mk(store, source_registry, tmp_path)
        batch = _batch(engine)
        prefix = manual_intake.derive_raw_prefix(
            client_id="client_a", portfolio_id="direct_001",
            reporting_period="2026-06-30", container="raw-v2")
        automated = engine.automated_identity(prefix, "raw-v2")
        assert automated["batch_id"] == batch["batch_id"]
        assert automated["client_id"] == "client_a"
        assert automated["portfolio_id"] == "direct_001"
        assert automated["reporting_period"] == "2026-06-30"
        assert automated["dataset"] == "funded"
        assert automated["workflow_type"] == batch["workflow_type"]

    def test_the_same_agreement_holds_for_a_pipeline_delivery(
            self, store, source_registry, tmp_path):
        engine = _mk(store, source_registry, tmp_path)
        batch = engine.create_batch(
            client_id="client_a", portfolio_id="direct_001",
            reporting_date="2026-06-30", workflow_type="mi",
            created_by="alice", auto_start_when_ready=True, dataset="pipeline")
        prefix = manual_intake.derive_raw_prefix(
            client_id="client_a", portfolio_id="direct_001",
            reporting_period="2026-06-30", dataset="pipeline",
            container="raw-v2")
        assert engine.automated_identity(prefix, "raw-v2")["batch_id"] == \
            batch["batch_id"]

    def test_asking_for_the_annex_on_a_book_that_does_not_report_it_is_refused(
            self, store, source_registry, tmp_path):
        """The automated route would prepare management information for this
        book, so an Annex 2 pack created by hand would never be joined by the
        files the trigger registers."""
        engine = _mk(store, source_registry, tmp_path)
        self._register(source_registry, "direct_001", regime=False)
        batch = _batch(engine, workflow_type="mi_annex2")
        with pytest.raises(OpsError) as exc:
            engine.upload_batch_files(
                client_id="client_a", batch_id=batch["batch_id"],
                uploads=[("loan_tape_2026.csv", TAPE)], received_by="alice")
        assert exc.value.code == "OPS_IDENTITY_DIVERGENCE"
        assert "management information only" in exc.value.message

    def test_a_regime_book_refuses_a_management_information_only_pack(
            self, store, source_registry, tmp_path):
        engine = _mk(store, source_registry, tmp_path)
        self._register(source_registry, "direct_001", regime=True)
        batch = _batch(engine, workflow_type="mi")
        with pytest.raises(OpsError) as exc:
            engine.upload_batch_files(
                client_id="client_a", batch_id=batch["batch_id"],
                uploads=[("loan_tape_2026.csv", TAPE)], received_by="alice")
        assert exc.value.code == "OPS_IDENTITY_DIVERGENCE"
        assert "ESMA Annex 2" in exc.value.message

    def test_a_refusal_writes_nothing_and_is_audited(self, store,
                                                     source_registry, tmp_path):
        engine = _mk(store, source_registry, tmp_path)
        self._register(source_registry, "direct_001", regime=True)
        batch = _batch(engine, workflow_type="mi")
        with pytest.raises(OpsError):
            engine.upload_batch_files(
                client_id="client_a", batch_id=batch["batch_id"],
                uploads=[("loan_tape_2026.csv", TAPE)], received_by="alice")

        # Nothing reached storage, nothing reached the pack, and the refusal is
        # on the record.
        assert store.storage.list("blob://raw-v2") == []
        reloaded = engine.intake.load_batch("client_a", batch["batch_id"])
        assert reloaded["files"] == []
        assert reloaded["source_prefix"] == ""
        events = [a for a in store.list_audit("client_a")
                  if a["event"] == "manual_delivery_refused"]
        assert events and events[0]["detail"]["reason"] == "identity_divergence"
        assert store.verify_audit_chain("client_a")

    def test_a_matching_choice_still_goes_through(self, store, source_registry,
                                                  tmp_path):
        engine = _mk(store, source_registry, tmp_path)
        self._register(source_registry, "direct_001", regime=True)
        batch = _batch(engine, workflow_type="mi_annex2")
        result = engine.upload_batch_files(
            client_id="client_a", batch_id=batch["batch_id"],
            uploads=[("loan_tape_2026.csv", TAPE)], received_by="alice")
        assert result["files"], "an agreed delivery must not be refused"

    def test_the_operator_message_never_leaks_a_location(self, store,
                                                         source_registry,
                                                         tmp_path):
        from operations_control.language import is_operator_safe
        engine = _mk(store, source_registry, tmp_path)
        self._register(source_registry, "direct_001", regime=True)
        batch = _batch(engine, workflow_type="mi")
        with pytest.raises(OpsError) as exc:
            engine.upload_batch_files(
                client_id="client_a", batch_id=batch["batch_id"],
                uploads=[("loan_tape_2026.csv", TAPE)], received_by="alice")
        assert is_operator_safe(exc.value.message)


class TestTheStorageEventTheUploadRaises:
    """A manual delivery is filed where an automated one would be, so the
    storage event fires and offers Trakt the very files it has just processed.
    That echo must change nothing: no second copy of the delivery, and above all
    no second run of it."""

    @staticmethod
    def _replay_arrival(engine, monkeypatch, blob_path, local: Path):
        """Drive the real automated handler for a blob the operator uploaded."""
        from apps.blob_trigger_app import occ_intake
        monkeypatch.setattr(occ_intake, "_engine", lambda: engine)

        def download(_container, _blob_path, dest_dir):
            dest = Path(dest_dir) / Path(blob_path).name
            dest.write_bytes(local.read_bytes())
            return dest

        return occ_intake.handle_arrival("raw-v2", blob_path, download=download)

    def test_the_echo_does_not_start_a_second_run(self, store, source_registry,
                                                  tmp_path, monkeypatch):
        engine = _mk(store, source_registry, tmp_path)
        batch = _batch(engine)
        uploaded = engine.upload_batch_files(
            client_id="client_a", batch_id=batch["batch_id"],
            uploads=[("loan_tape_2026.csv", TAPE)], received_by="alice")
        workflow_id = uploaded["workflow_id"]
        assert workflow_id

        local = tmp_path / "echo.csv"
        local.write_bytes(TAPE)
        result = self._replay_arrival(
            engine, monkeypatch,
            "client_a/direct/funded/monthly/direct_001/2026-06-30/"
            "loan_tape_2026.csv", local)

        # The pack that ran is untouched, and no second workflow exists.
        assert result["batch_id"].startswith(batch["batch_id"])
        runs = {b.get("workflow_id") for b in
                engine.intake.list_batches("client_a") if b.get("workflow_id")}
        assert runs == {workflow_id}, (
            "the storage event raised by the operator's own upload started a "
            "second run of the same delivery")

    def test_the_echo_does_not_add_a_second_copy_of_the_delivery(
            self, store, source_registry, tmp_path, monkeypatch):
        engine = _mk(store, source_registry, tmp_path)
        batch = _batch(engine, auto=False)
        engine.upload_batch_files(
            client_id="client_a", batch_id=batch["batch_id"],
            uploads=[("loan_tape_2026.csv", TAPE)], received_by="alice")

        local = tmp_path / "echo.csv"
        local.write_bytes(TAPE)
        self._replay_arrival(
            engine, monkeypatch,
            "client_a/direct/funded/monthly/direct_001/2026-06-30/"
            "loan_tape_2026.csv", local)

        packs = engine.intake.pack_family("client_a", batch["batch_id"])
        registered = [f for p in packs for f in p["files"]]
        assert len(registered) == 1, "the same file was taken twice"
        entries = [a for a in store.list_audit("client_a")
                   if a["event"] == "file_registered"
                   and a["detail"].get("duplicate_status") == "duplicate_ignored"]
        assert entries, "the echo was not recorded as a duplicate"

    def test_content_already_in_an_earlier_version_of_the_pack_is_a_duplicate(
            self, store, source_registry, tmp_path):
        """The check spans pack VERSIONS, not one document — otherwise an echo
        arriving after the pack started would open a successor for itself."""
        engine = _mk(store, source_registry, tmp_path)
        batch = _batch(engine)
        engine.upload_batch_files(
            client_id="client_a", batch_id=batch["batch_id"],
            uploads=[("loan_tape_2026.csv", TAPE)], received_by="alice")

        started = engine.intake.load_batch("client_a", batch["batch_id"])
        assert started["status"] == "running"
        echo = tmp_path / "echo.csv"
        echo.write_bytes(TAPE)
        same = engine.intake.register_file(started, echo,
                                           received_by_or_source="blob-trigger")
        assert same["batch_id"] == batch["batch_id"], (
            "an echo of a file the pack already holds opened a successor pack")
        assert engine.intake.load_batch("client_a",
                                        f"{batch['batch_id']}_v2") is None

    def test_a_genuinely_late_file_still_opens_a_successor(self, store,
                                                           source_registry,
                                                           tmp_path):
        engine = _mk(store, source_registry, tmp_path)
        batch = _batch(engine)
        engine.upload_batch_files(
            client_id="client_a", batch_id=batch["batch_id"],
            uploads=[("loan_tape_2026.csv", TAPE)], received_by="alice")

        started = engine.intake.load_batch("client_a", batch["batch_id"])
        late = tmp_path / "late.csv"
        late.write_bytes(b"loan_id,balance\nL9,5\n")
        successor = engine.intake.register_file(started, late,
                                                received_by_or_source="alice")
        assert successor["batch_id"] == f"{batch['batch_id']}_v2"

    def test_pack_versions_are_recognised_as_one_pack(self, store,
                                                      source_registry,
                                                      tmp_path):
        engine = _mk(store, source_registry, tmp_path)
        assert engine.intake.base_batch_id("batch_abc123_v4") == "batch_abc123"
        assert engine.intake.base_batch_id("batch_abc123") == "batch_abc123"


# --------------------------------------------------------------------------- #
# The HTTP surface
# --------------------------------------------------------------------------- #

@pytest.fixture()
def api(store, source_registry, tmp_path):
    engine = _mk(store, source_registry, tmp_path)
    app_module.set_engine(engine)
    client = TestClient(app_module.app, raise_server_exceptions=False)
    yield {"client": client, "engine": engine}
    app_module.set_engine(None)


class TestUploadEndpoint:
    def test_an_operator_uploads_and_lands_on_a_workflow(self, api):
        batch = _batch(api["engine"])
        r = api["client"].post(
            f"/ops/batches/{batch['batch_id']}/upload?client=client_a",
            headers=_h(OP_A),
            files=[("files", ("loan_tape_2026.csv", TAPE, "text/csv"))])
        assert r.status_code == 200, r.text
        assert r.json()["batch"]["workflow_id"]

    def test_another_clients_operator_cannot_upload(self, api):
        batch = _batch(api["engine"])
        r = api["client"].post(
            f"/ops/batches/{batch['batch_id']}/upload?client=client_a",
            headers=_h(OP_B),
            files=[("files", ("loan_tape_2026.csv", TAPE, "text/csv"))])
        assert r.status_code == 404
        assert api["engine"].intake.load_batch(
            "client_a", batch["batch_id"])["files"] == []

    def test_an_unauthenticated_upload_is_refused(self, api):
        batch = _batch(api["engine"])
        r = api["client"].post(
            f"/ops/batches/{batch['batch_id']}/upload?client=client_a",
            files=[("files", ("loan_tape_2026.csv", TAPE, "text/csv"))])
        assert r.status_code == 401

    def test_a_traversing_filename_lands_on_its_leaf_only(self, api):
        batch = _batch(api["engine"], auto=False)
        r = api["client"].post(
            f"/ops/batches/{batch['batch_id']}/upload?client=client_a",
            headers=_h(OP_A),
            files=[("files",
                    ("../../loan_tape_2026.csv", TAPE, "text/csv"))])
        assert r.status_code == 200, r.text
        names = [f["filename"] for f in r.json()["batch"]["files"]]
        assert names == ["loan_tape_2026.csv"]


class TestServerPathRouteIsFailClosed:
    """The old free-text location route. A browser must not be able to point
    Trakt at a location on the server."""

    def test_an_operator_cannot_name_a_location(self, api, tmp_path):
        batch = _batch(api["engine"])
        f = tmp_path / "elsewhere.csv"
        f.write_bytes(TAPE)
        r = api["client"].post(
            f"/ops/batches/{batch['batch_id']}/files?client=client_a",
            headers=_h(OP_A), json={"path": str(f)})
        assert r.status_code == 403
        assert r.json()["detail"]["errorCode"] == "OPS_ADMIN_REQUIRED"

    def test_an_administrator_is_refused_outside_the_allow_list(self, api,
                                                               tmp_path):
        batch = _batch(api["engine"])
        f = tmp_path / "elsewhere.csv"
        f.write_bytes(TAPE)
        r = api["client"].post(
            f"/ops/batches/{batch['batch_id']}/files?client=client_a",
            headers=_h(OP_ADMIN), json={"path": str(f)})
        assert r.status_code == 403
        assert r.json()["detail"]["errorCode"] == "OPS_PATH_NOT_ALLOWED"

    def test_traversal_out_of_an_allow_listed_root_is_refused(
            self, api, tmp_path, monkeypatch):
        allowed = tmp_path / "incoming"
        allowed.mkdir()
        outside = tmp_path / "secret.csv"
        outside.write_bytes(TAPE)
        monkeypatch.setenv("TRAKT_OPS_SERVER_PATH_ROOTS", str(allowed))
        batch = _batch(api["engine"])
        r = api["client"].post(
            f"/ops/batches/{batch['batch_id']}/files?client=client_a",
            headers=_h(OP_ADMIN),
            json={"path": str(allowed / ".." / "secret.csv")})
        assert r.status_code == 403

    def test_an_allow_listed_root_still_works_for_server_tooling(
            self, api, tmp_path, monkeypatch):
        allowed = tmp_path / "incoming"
        allowed.mkdir()
        (allowed / "loan_tape_2026.csv").write_bytes(TAPE)
        monkeypatch.setenv("TRAKT_OPS_SERVER_PATH_ROOTS", str(allowed))
        batch = _batch(api["engine"], auto=False)
        r = api["client"].post(
            f"/ops/batches/{batch['batch_id']}/files?client=client_a",
            headers=_h(OP_ADMIN),
            json={"path": str(allowed / "loan_tape_2026.csv")})
        assert r.status_code == 200, r.text
        assert r.json()["batch"]["files"]
