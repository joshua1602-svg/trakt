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
