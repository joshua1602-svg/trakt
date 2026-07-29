"""The administrator configuration API the OCC React admin area consumes.

Covers the read models added for the administrator area (overview, catalogue,
compatibility, comparison, impact, audit) and the safety rules around the
existing governed lifecycle: an invalid draft cannot be activated, an
incompatible version cannot be activated, an operator cannot reach any of it,
and superseded versions are never lost.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from operations_control.api import app as app_module
from operations_control.configuration import admin_views
from operations_control.configuration.packages import (
    ConfigPackageStore,
    LAYER_ASSET,
    LAYER_REGIME,
    LAYER_SYSTEM,
)

from .conftest import (
    OP_A,
    OP_ADMIN,
    make_engine,
    register_and_create,
    start_and_wait,
)

ADMIN = {"X-Operator-Token": OP_ADMIN}
OPERATOR = {"X-Operator-Token": OP_A}


@pytest.fixture()
def api(store, source_registry):
    engine = make_engine(store, source_registry)
    app_module.set_engine(engine)
    client = TestClient(app_module.app, raise_server_exceptions=False)
    try:
        yield client, engine
    finally:
        app_module.set_engine(None)


# --------------------------------------------------------------------------- #
# Access control
# --------------------------------------------------------------------------- #

class TestAccess:
    def test_me_reports_the_role(self, api):
        client, _ = api
        admin = client.get("/ops/me", headers=ADMIN).json()["principal"]
        assert admin["is_admin"] is True
        assert admin["role"] == "admin"
        operator = client.get("/ops/me", headers=OPERATOR).json()["principal"]
        assert operator["is_admin"] is False

    @pytest.mark.parametrize("path", [
        "/ops/admin/config",
        "/ops/admin/config/catalogue",
        "/ops/admin/config/audit",
        "/ops/admin/config/system/1",
        "/ops/admin/config/system/impact",
        "/ops/admin/config/system/compare?from_version=1&to_version=1",
    ])
    def test_operator_is_refused_every_admin_read(self, api, path):
        client, _ = api
        assert client.get(path, headers=OPERATOR).status_code == 403

    def test_operator_cannot_change_configuration(self, api):
        client, engine = api
        pkgs = ConfigPackageStore(engine.store)
        pkgs.ensure_seeded(LAYER_SYSTEM, by="admin")
        assert client.post("/ops/admin/config/system/draft", headers=OPERATOR,
                           json={"edits": {}}).status_code == 403
        assert client.post("/ops/admin/config/system/1/validate",
                           headers=OPERATOR).status_code == 403
        assert client.post("/ops/admin/config/system/1/activate",
                           headers=OPERATOR).status_code == 403
        assert client.post("/ops/admin/config/system/rollback",
                           headers=OPERATOR,
                           json={"to_version": 1}).status_code == 403
        assert len(pkgs.list_versions(LAYER_SYSTEM)) == 1

    def test_unauthenticated_is_refused(self, api):
        client, _ = api
        assert client.get("/ops/admin/config").status_code == 401


# --------------------------------------------------------------------------- #
# Overview
# --------------------------------------------------------------------------- #

class TestOverview:
    def test_overview_shows_active_versions_for_all_three_layers(self, api):
        client, _ = api
        body = client.get("/ops/admin/config", headers=ADMIN).json()
        assert body["ok"] is True
        for layer in (LAYER_SYSTEM, LAYER_REGIME, LAYER_ASSET):
            info = body["layers"][layer]
            assert info["active_version"] == 1
            assert info["status"] == "active"
            assert info["package_hash"].startswith("sha256:")
            assert info["file_count"] > 0
            assert info["draft"] is None
            assert info["validation_summary"]["ok"] is True

    def test_overview_carries_the_catalogues_and_compatibility(self, api):
        client, _ = api
        body = client.get("/ops/admin/config", headers=ADMIN).json()
        assets = {a["id"]: a for a in body["assets"]}
        regimes = {r["id"]: r for r in body["regimes"]}
        assert "equity_release" in assets
        assert assets["equity_release"]["label"] == "Equity Release"
        assert [r["id"] for r in assets["equity_release"]["supported_regimes"]] \
            == ["ESMA_Annex2"]
        assert regimes["ESMA_Annex2"]["regulator"] == "ESMA"
        assert regimes["ESMA_Annex2"]["annex"] == "Annex 2"
        assert regimes["ESMA_Annex2"]["field_universe"]["count"] > 0
        assert regimes["ESMA_Annex2"]["schema"]["file"].endswith(".xsd")
        row = body["compatibility"]["rows"][0]
        assert row["asset"] == "equity_release"
        assert row["cells"][0]["supported"] is True
        assert row["cells"][0]["label"] == "Supported"

    def test_overview_surfaces_a_draft_awaiting_action(self, api):
        client, engine = api
        pkgs = ConfigPackageStore(engine.store)
        pkgs.ensure_seeded(LAYER_SYSTEM, by="admin")
        pkgs.create_draft(LAYER_SYSTEM, by="admin", notes="tidy up",
                          edits={"config/system/run_context_policy.yaml":
                                 "version: 9\n"})
        body = client.get("/ops/admin/config", headers=ADMIN).json()
        draft = body["layers"][LAYER_SYSTEM]["draft"]
        assert draft["version"] == 2
        assert draft["based_on_version"] == 1
        assert draft["notes"] == "tidy up"
        assert draft["validated"] is False
        assert [f["label"] for f in draft["changed_files"]] == ["Run policy"]
        assert body["attention"]["drafts_awaiting_action"][0]["version"] == 2
        assert body["attention"]["packages_requiring_review"][0]["layer"] \
            == LAYER_SYSTEM

    def test_overview_surfaces_validation_failures(self, api):
        client, engine = api
        pkgs = ConfigPackageStore(engine.store)
        pkgs.ensure_seeded(LAYER_SYSTEM, by="admin")
        draft = pkgs.create_draft(
            LAYER_SYSTEM, by="admin",
            edits={"config/system/run_context_policy.yaml": "not: [valid: yaml"})
        pkgs.validate_version(LAYER_SYSTEM, draft["version"])
        body = client.get("/ops/admin/config", headers=ADMIN).json()
        failure = body["attention"]["validation_failures"][0]
        assert failure["layer"] == LAYER_SYSTEM
        assert failure["version"] == draft["version"]
        summary = body["layers"][LAYER_SYSTEM]["draft"]["validation_summary"]
        assert summary["state"] == "failed"
        assert summary["headline"] == "Validation failed"
        assert summary["problems"][0]["file"] == "Run policy"


# --------------------------------------------------------------------------- #
# Version inspection
# --------------------------------------------------------------------------- #

class TestVersionInspection:
    def test_version_lists_files_with_plain_names_and_hashes(self, api):
        client, _ = api
        body = client.get("/ops/admin/config/system/1", headers=ADMIN).json()
        version = body["version"]
        assert version["version"] == 1
        assert version["package_hash"].startswith("sha256:")
        labels = {f["label"] for f in version["file_list"]}
        assert "Field registry" in labels
        for entry in version["file_list"]:
            assert entry["sha256"].startswith("sha256:")
            assert entry["size"] > 0
        assert version["activation_blockers"] == []
        # Contents are not shipped unless a specific file is asked for.
        assert "file_content" not in version

    def test_a_single_file_can_be_read(self, api):
        client, _ = api
        body = client.get(
            "/ops/admin/config/system/1"
            "?file=config/system/fields_registry.yaml",
            headers=ADMIN).json()
        assert body["version"]["file_label"] == "Field registry"
        assert "fields" in body["version"]["file_content"]

    def test_unknown_version_is_not_found(self, api):
        client, _ = api
        assert client.get("/ops/admin/config/system/99",
                          headers=ADMIN).status_code == 404


# --------------------------------------------------------------------------- #
# Lifecycle
# --------------------------------------------------------------------------- #

class TestLifecycle:
    def test_draft_validate_activate_keeps_the_previous_version(self, api):
        client, engine = api
        client.get("/ops/admin/config", headers=ADMIN)   # seed
        draft = client.post("/ops/admin/config/system/draft", headers=ADMIN,
                            json={"edits": {
                                "config/system/run_context_policy.yaml":
                                "version: 7\n"}, "notes": "raise the version"})
        assert draft.status_code == 200
        version = draft.json()["version"]
        assert draft.json()["status"] == "draft"

        validated = client.post(
            f"/ops/admin/config/system/{version}/validate", headers=ADMIN)
        assert validated.json()["validation"]["ok"] is True
        assert validated.json()["validation_summary"]["headline"] \
            == "Checks passed"

        activated = client.post(
            f"/ops/admin/config/system/{version}/activate", headers=ADMIN)
        assert activated.json()["active_version"] == version

        pkgs = ConfigPackageStore(engine.store)
        assert pkgs.active_version(LAYER_SYSTEM)["version"] == version
        superseded = pkgs.get_version(LAYER_SYSTEM, 1)
        assert superseded is not None
        assert superseded["status"] == "superseded"
        assert superseded["files"], "the superseded version keeps its contents"

    def test_an_unvalidated_draft_cannot_be_activated(self, api):
        client, engine = api
        client.get("/ops/admin/config", headers=ADMIN)
        version = client.post("/ops/admin/config/system/draft", headers=ADMIN,
                              json={"edits": {}}).json()["version"]
        response = client.post(
            f"/ops/admin/config/system/{version}/activate", headers=ADMIN)
        assert response.status_code == 409
        assert response.json()["detail"]["errorCode"] == "OPS_CONFIG_NOT_READY"
        assert ConfigPackageStore(
            engine.store).active_version(LAYER_SYSTEM)["version"] == 1

    def test_a_failing_draft_cannot_be_activated(self, api):
        client, engine = api
        client.get("/ops/admin/config", headers=ADMIN)
        version = client.post(
            "/ops/admin/config/system/draft", headers=ADMIN,
            json={"edits": {"config/system/run_context_policy.yaml":
                            "not: [valid: yaml"}}).json()["version"]
        validated = client.post(
            f"/ops/admin/config/system/{version}/validate", headers=ADMIN)
        summary = validated.json()["validation_summary"]
        assert summary["ok"] is False
        assert "cannot be activated" in summary["detail"]
        assert summary["problems"][0]["summary"].startswith("Run policy")
        # The raw problem stays behind the technical field.
        assert ".yaml" in summary["problems"][0]["technical"]
        assert ".yaml" not in summary["problems"][0]["summary"]
        assert client.post(f"/ops/admin/config/system/{version}/activate",
                           headers=ADMIN).status_code == 409

    def test_a_version_missing_a_governed_file_is_refused(self, api):
        client, engine = api
        pkgs = ConfigPackageStore(engine.store)
        pkgs.ensure_seeded(LAYER_REGIME, by="admin")
        draft = pkgs.create_draft(LAYER_REGIME, by="admin")
        # A draft that no longer carries the Annex 2 package at all: Equity
        # Release still reports under it, so activation must be refused.
        draft["files"] = {}
        from operations_control.stores import _write_json
        _write_json(engine.store.storage,
                    pkgs._version_uri(LAYER_REGIME, draft["version"]), draft)
        pkgs.validate_version(LAYER_REGIME, draft["version"])
        response = client.post(
            f"/ops/admin/config/regime/{draft['version']}/activate",
            headers=ADMIN)
        assert response.status_code == 409
        detail = response.json()["detail"]
        assert detail["errorCode"] == "OPS_CONFIG_INCOMPATIBLE"
        kinds = {b["kind"] for b in detail["blockers"]}
        assert "missing_file" in kinds
        assert "incompatible_dependency" in kinds
        assert pkgs.active_version(LAYER_REGIME)["version"] == 1

    def test_rollback_reactivates_the_prior_version(self, api):
        client, engine = api
        client.get("/ops/admin/config", headers=ADMIN)
        version = client.post(
            "/ops/admin/config/system/draft", headers=ADMIN,
            json={"edits": {"config/system/run_context_policy.yaml":
                            "version: 4\n"}}).json()["version"]
        client.post(f"/ops/admin/config/system/{version}/validate",
                    headers=ADMIN)
        client.post(f"/ops/admin/config/system/{version}/activate",
                    headers=ADMIN)
        rolled = client.post("/ops/admin/config/system/rollback", headers=ADMIN,
                             json={"to_version": 1})
        assert rolled.json()["active_version"] == 1
        pkgs = ConfigPackageStore(engine.store)
        assert pkgs.active_version(LAYER_SYSTEM)["version"] == 1
        # The version rolled back FROM is kept, not deleted.
        assert pkgs.get_version(LAYER_SYSTEM, version) is not None


# --------------------------------------------------------------------------- #
# Comparison
# --------------------------------------------------------------------------- #

class TestComparison:
    def test_compare_groups_changes_and_returns_a_structured_diff(self, api):
        client, engine = api
        pkgs = ConfigPackageStore(engine.store)
        pkgs.ensure_seeded(LAYER_SYSTEM, by="admin")
        pkgs.create_draft(LAYER_SYSTEM, by="admin",
                          edits={"config/system/run_context_policy.yaml":
                                 "version: 5\n"})
        body = client.get("/ops/admin/config/system/compare"
                          "?from_version=1&to_version=2", headers=ADMIN).json()
        comparison = body["comparison"]
        assert comparison["summary"]["changed"] == 1
        assert comparison["summary"]["added"] == 0
        assert comparison["summary"]["removed"] == 0
        changed = [f for f in comparison["files"] if f["change"] == "changed"]
        assert changed[0]["label"] == "Run policy"
        kinds = {line["kind"] for line in changed[0]["lines"]}
        assert "added" in kinds
        assert comparison["from"]["status"] == "active"
        assert comparison["to"]["status"] == "draft"

    def test_compare_with_an_unknown_version_is_not_found(self, api):
        client, _ = api
        assert client.get("/ops/admin/config/system/compare"
                          "?from_version=1&to_version=44",
                          headers=ADMIN).status_code == 404


# --------------------------------------------------------------------------- #
# Impact
# --------------------------------------------------------------------------- #

class TestImpact:
    def test_impact_counts_pinned_workflows(self, store, source_registry,
                                            delivery_dir):
        engine = make_engine(store, source_registry)
        run = register_and_create(engine, delivery_dir)
        start_and_wait(engine, run)
        app_module.set_engine(engine)
        client = TestClient(app_module.app, raise_server_exceptions=False)
        try:
            body = client.get("/ops/admin/config/system/impact",
                              headers=ADMIN).json()["impact"]
        finally:
            app_module.set_engine(None)
        assert body["version"] == 1
        assert body["clients"] == 1
        assert body["portfolios"] == 1
        assert body["pinned_workflows"] == 1
        assert body["future_workflows_affected"] is True
        assert body["workflows"][0]["pinned_version"] == "v1"
        assert any("already started stay on the version" in line
                   for line in body["explanation"])

    def test_impact_is_empty_when_nothing_is_pinned(self, api):
        client, _ = api
        body = client.get("/ops/admin/config/asset/impact",
                          headers=ADMIN).json()["impact"]
        assert body["pinned_workflows"] == 0
        assert body["running_pinned_workflows"] == 0


# --------------------------------------------------------------------------- #
# Audit
# --------------------------------------------------------------------------- #

class TestAudit:
    def test_every_lifecycle_step_is_recorded_in_plain_english(self, api):
        client, _ = api
        client.get("/ops/admin/config", headers=ADMIN)
        version = client.post(
            "/ops/admin/config/system/draft", headers=ADMIN,
            json={"edits": {"config/system/run_context_policy.yaml":
                            "version: 6\n"}, "notes": "bump"}).json()["version"]
        client.post(f"/ops/admin/config/system/{version}/validate",
                    headers=ADMIN)
        client.post(f"/ops/admin/config/system/{version}/activate",
                    headers=ADMIN)
        client.post("/ops/admin/config/system/rollback", headers=ADMIN,
                    json={"to_version": 1})

        body = client.get("/ops/admin/config/audit", headers=ADMIN).json()
        assert body["chain_intact"] is True
        events = [e["event"] for e in body["entries"]]
        assert events == ["config_rolled_back", "config_activated",
                          "config_validated", "config_draft_created"]
        activation = next(e for e in body["entries"]
                          if e["event"] == "config_activated")
        assert activation["actor"] == "Administrator"
        assert "Future workflows will use this version" \
            in activation["description"]
        assert "Existing workflows remain pinned" in activation["description"]
        rollback = body["entries"][0]
        assert "rolled back to v1" in rollback["description"]
        assert "kept" in rollback["description"]

    def test_a_failed_validation_reads_as_a_failure(self, api):
        client, _ = api
        client.get("/ops/admin/config", headers=ADMIN)
        version = client.post(
            "/ops/admin/config/system/draft", headers=ADMIN,
            json={"edits": {"config/system/run_context_policy.yaml":
                            "not: [valid: yaml"}}).json()["version"]
        client.post(f"/ops/admin/config/system/{version}/validate",
                    headers=ADMIN)
        entries = client.get("/ops/admin/config/audit",
                             headers=ADMIN).json()["entries"]
        failure = next(e for e in entries if e["event"] == "config_validated")
        assert failure["outcome"] == "failed"
        assert "did not pass" in failure["description"]


# --------------------------------------------------------------------------- #
# Read-model unit checks
# --------------------------------------------------------------------------- #

class TestReadModels:
    def test_no_repository_path_leaks_into_a_plain_name(self):
        for rel, label in admin_views.FILE_LABELS.items():
            assert "/" not in label
            assert ".yaml" not in label
            assert admin_views.friendly_file_name(rel) == label
        assert admin_views.friendly_file_name(
            "config/system/unknown_thing.yaml") == "Unknown thing"

    def test_compatibility_only_reports_declared_relationships(self, store):
        pkgs = ConfigPackageStore(store)
        assets = admin_views.describe_assets(pkgs)
        regimes = admin_views.describe_regimes(pkgs, pkgs.repo)
        matrix = admin_views.compatibility_matrix(assets, regimes)
        for row in matrix["rows"]:
            for cell in row["cells"]:
                assert cell["label"] in ("Supported", "Not supported")

    def test_a_draft_regulatory_schema_is_reported_as_a_warning(self, store):
        pkgs = ConfigPackageStore(store)
        assets = admin_views.describe_assets(pkgs)
        regimes = admin_views.describe_regimes(pkgs, pkgs.repo)
        issues = admin_views.compatibility_issues(assets, regimes)
        assert any("draft regulatory schema" in i["message"] for i in issues)
