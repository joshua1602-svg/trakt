"""Tenancy, isolation, and the API boundary.

The OCC Agent is mounted inside the existing Operations Control API, behind the
existing operator authentication and tenancy binding. These tests assert that
the boundary is genuinely the existing one, that one tenant cannot reach
another's case, and that the feature flag governs the routes as well as the tab.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

from operations_control.occ_agent import api as agent_api
from operations_control.occ_agent import policy as _policy
from operations_control.occ_agent.scenarios import run_scenario
from operations_control.occ_agent.service import OccAgentService
from operations_control.occ_agent.store import CaseNotFound

from .conftest import ACTOR, TENANT_A, TENANT_B


# --------------------------------------------------------------------------- #
# Tenancy at the store
# --------------------------------------------------------------------------- #

def test_one_tenant_cannot_load_another_tenants_case(service):
    case = service.create_case(tenant=TENANT_A, initiating_user=ACTOR,
                               instruction="Onboard Northstar Lending.")
    with pytest.raises(CaseNotFound):
        service.load_case(TENANT_B, case.case_id)


def test_a_tenants_case_list_shows_only_its_own(service):
    a = service.create_case(tenant=TENANT_A, initiating_user=ACTOR,
                            instruction="Onboard Northstar Lending.")
    b = service.create_case(tenant=TENANT_B, initiating_user="Bob",
                            instruction="Onboard Harbour Point Capital.")
    assert [row["case_id"] for row in service.list_cases(TENANT_A)] == [a.case_id]
    assert [row["case_id"] for row in service.list_cases(TENANT_B)] == [b.case_id]


def test_synthetic_files_are_isolated_by_tenant_and_case(service, agent_env):
    a = service.create_case(tenant=TENANT_A, initiating_user=ACTOR,
                            instruction="Onboard Northstar Lending.")
    b = service.create_case(tenant=TENANT_B, initiating_user="Bob",
                            instruction="Onboard Harbour Point Capital.")
    service.artefacts.register(a, filename="a.csv", data=b"x,y\n1,2\n",
                               provided_by=ACTOR)
    service.artefacts.register(b, filename="b.csv", data=b"x,y\n3,4\n",
                               provided_by="Bob")
    dir_a = service.store.artefact_dir(TENANT_A, a.case_id)
    dir_b = service.store.artefact_dir(TENANT_B, b.case_id)
    assert dir_a != dir_b
    assert TENANT_A in str(dir_a) and TENANT_B in str(dir_b)
    assert {p.name for p in dir_a.iterdir()} == {"a.csv"}
    assert {p.name for p in dir_b.iterdir()} == {"b.csv"}


def test_a_case_cannot_reach_another_cases_files(service):
    a = service.create_case(tenant=TENANT_A, initiating_user=ACTOR,
                            instruction="Onboard Northstar Lending.")
    b = service.create_case(tenant=TENANT_A, initiating_user=ACTOR,
                            instruction="Onboard Harbour Point Capital.")
    service.artefacts.register(a, filename="secret.csv", data=b"x,y\n1,2\n",
                               provided_by=ACTOR)
    # Case B's own artefact list is empty, and its run sees no files.
    assert service.load_case(TENANT_A, b.case_id).received_artefacts == []
    assert service._artefact_paths(service.load_case(TENANT_A, b.case_id)) == []


def test_a_document_moved_between_tenant_folders_is_refused(service,
                                                            synthetic_store):
    """Defence in depth: the case's own tenant must match where it was found."""
    case = service.create_case(tenant=TENANT_A, initiating_user=ACTOR,
                               instruction="Onboard Northstar Lending.")
    doc = json.loads(Path(synthetic_store.storage._local_path(
        synthetic_store.case_uri(TENANT_A, case.case_id))).read_text(
            encoding="utf-8"))
    # Plant tenant A's document in tenant B's folder.
    from operations_control.stores import _write_json
    _write_json(synthetic_store.storage,
                synthetic_store.case_uri(TENANT_B, case.case_id), doc)
    with pytest.raises(CaseNotFound):
        synthetic_store.load(TENANT_B, case.case_id)


def test_the_audit_trail_is_tamper_evident(service):
    run = run_scenario(service, "scenario_a_clean", tenant=TENANT_A, actor=ACTOR)
    assert service.store.verify_audit_chain(TENANT_A, run.case.case_id) is True
    events = service.store.list_audit(TENANT_A, run.case.case_id)
    assert len(events) > 5
    for event in events:
        assert event["runtime_mode"] == _policy.RUNTIME_MODE_SYNTHETIC
        assert event["execution_classification"]
        assert event["actor_type"] in ("human", "agent", "system")
        assert "event_id" in event and "at" in event and "action" in event


def test_the_audit_trail_records_no_hidden_reasoning(service):
    run = run_scenario(service, "scenario_a_clean", tenant=TENANT_A, actor=ACTOR)
    for event in service.store.list_audit(TENANT_A, run.case.case_id):
        basis = event.get("decision_basis", "")
        # A concise rationale, not a transcript.
        assert len(basis) < 200, basis
        assert "thinking" not in basis.lower()
        assert "chain of thought" not in basis.lower()


def test_a_tampered_audit_record_is_detected(service, synthetic_store):
    case = service.create_case(tenant=TENANT_A, initiating_user=ACTOR,
                               instruction="Onboard Northstar Lending.")
    uri = synthetic_store.audit_uri(TENANT_A, case.case_id, 1)
    path = Path(synthetic_store.storage._local_path(uri))
    doc = json.loads(path.read_text(encoding="utf-8"))
    doc["action"] = "something_else"
    path.write_text(json.dumps(doc), encoding="utf-8")
    assert synthetic_store.verify_audit_chain(TENANT_A, case.case_id) is False


# --------------------------------------------------------------------------- #
# The feature flag
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("value,expected", [
    ("true", True), ("TRUE", True), ("1", True), ("yes", True), ("on", True),
    ("enabled", True), ("false", False), ("0", False), ("", False),
    ("maybe", False), ("  ", False),
])
def test_the_feature_flag_fails_closed(monkeypatch, value, expected):
    monkeypatch.setenv(_policy.FEATURE_FLAG_ENV, value)
    assert _policy.feature_enabled() is expected


def test_the_feature_flag_is_off_when_unset(monkeypatch):
    monkeypatch.delenv(_policy.FEATURE_FLAG_ENV, raising=False)
    assert _policy.feature_enabled() is False


# --------------------------------------------------------------------------- #
# The API boundary
# --------------------------------------------------------------------------- #

def route_paths(app) -> set:
    """Every path the application serves.

    Read from the generated schema rather than from ``app.routes``: an included
    router is held as a lazy wrapper there on this FastAPI version, so walking
    the list would under-report. The schema is what the application actually
    serves, which is the question these tests are asking.
    """
    app.openapi_schema = None              # force a fresh generation
    return set(app.openapi().get("paths") or {})


@pytest.fixture()
def api_client(agent_env, monkeypatch):
    """The EXISTING Operations Control API, with the agent routes mounted."""
    from fastapi.testclient import TestClient
    import operations_control.api.app as app_module
    importlib.reload(app_module)
    agent_api.configure(OccAgentService(
        __import__("apps.blob_trigger_app.storage",
                   fromlist=["Storage"]).Storage(agent_env["blob_root"]),
        container="operations-control-synthetic",
        sandbox=agent_env["sandbox"]))
    return TestClient(app_module.app)


def test_the_routes_are_mounted_inside_the_existing_api(api_client):
    """Same application, same auth — not a second service."""
    paths = route_paths(api_client.app)
    assert "/ops/dashboard" in paths            # the live OCC is still there
    assert "/ops/agent/cases" in paths          # and the tab's routes are too


def test_an_unauthenticated_request_is_refused(api_client):
    response = api_client.get("/ops/agent/cases")
    assert response.status_code == 401


def test_a_bad_token_is_refused(api_client):
    response = api_client.get("/ops/agent/cases",
                              headers={"X-Operator-Token": "nope"})
    assert response.status_code == 401


def test_a_case_is_created_and_read_back_over_the_api(api_client):
    headers = {"X-Operator-Token": "tok-a"}
    created = api_client.post(
        "/ops/agent/cases", headers=headers,
        json={"instruction": "Onboard Northstar Lending. UK equity release. "
                             "Monthly portfolio MI. Portfolio id direct_101."})
    assert created.status_code == 200
    case_id = created.json()["case"]["case_id"]
    fetched = api_client.get(f"/ops/agent/cases/{case_id}", headers=headers)
    assert fetched.status_code == 200
    assert fetched.json()["case"]["runtime_mode"] == "synthetic"
    assert fetched.json()["policy"]["allow_live_blob_write"] is False


def test_one_operator_cannot_read_another_tenants_case(api_client):
    created = api_client.post("/ops/agent/cases",
                              headers={"X-Operator-Token": "tok-a"},
                              json={"instruction": "Onboard Northstar Lending."})
    case_id = created.json()["case"]["case_id"]
    # Bob is bound to a different client and must not see it — 404, not 403.
    denied = api_client.get(f"/ops/agent/cases/{case_id}",
                            headers={"X-Operator-Token": "tok-b"})
    assert denied.status_code == 404


def test_an_operator_cannot_name_a_tenant_outside_their_binding(api_client):
    denied = api_client.get(f"/ops/agent/cases?tenant={TENANT_B}",
                            headers={"X-Operator-Token": "tok-a"})
    assert denied.status_code == 404


def test_the_whole_scenario_runs_over_the_api(api_client):
    headers = {"X-Operator-Token": "tok-a"}
    response = api_client.post("/ops/agent/scenarios/run", headers=headers,
                               json={"fixture_id": "scenario_a_clean"})
    assert response.status_code == 200
    body = response.json()
    assert body["case"]["state"] == "READY_FOR_EXECUTION"
    assert body["readiness"]["ready"] is True

    case_id = body["case"]["case_id"]
    package = api_client.get(f"/ops/agent/cases/{case_id}/readiness",
                             headers=headers).json()
    assert package["package"]["execution_manifest"]["execution_performed"] is False


def test_an_illegal_transition_is_refused_by_the_api(api_client):
    headers = {"X-Operator-Token": "tok-a"}
    created = api_client.post("/ops/agent/cases", headers=headers,
                              json={"instruction": "Onboard Northstar Lending."})
    case_id = created.json()["case"]["case_id"]
    # Readiness cannot be approved from the very first state.
    refused = api_client.post(f"/ops/agent/cases/{case_id}/readiness/approve",
                              headers=headers, json={})
    assert refused.status_code == 409
    assert refused.json()["errorCode"] == "OCC_AGENT_ACTION_NOT_ALLOWED"


def test_the_routes_answer_not_found_when_the_flag_is_off(api_client, monkeypatch):
    monkeypatch.setenv(_policy.FEATURE_FLAG_ENV, "false")
    response = api_client.get("/ops/agent/cases",
                              headers={"X-Operator-Token": "tok-a"})
    assert response.status_code == 404
    assert response.json()["errorCode"] == "OCC_AGENT_DISABLED"


def test_the_routes_are_not_mounted_at_all_without_the_flag(agent_env,
                                                            monkeypatch):
    """With the flag unset the live API is exactly what it was."""
    import operations_control.api.app as app_module
    monkeypatch.delenv(_policy.FEATURE_FLAG_ENV, raising=False)
    importlib.reload(app_module)
    paths = route_paths(app_module.app)
    assert not any(path.startswith("/ops/agent") for path in paths)
    assert "/ops/dashboard" in paths            # and the live OCC is untouched


def test_the_live_occ_routes_are_unchanged_by_the_mount(agent_env, monkeypatch):
    import operations_control.api.app as app_module

    monkeypatch.delenv(_policy.FEATURE_FLAG_ENV, raising=False)
    importlib.reload(app_module)
    without = route_paths(app_module.app)

    monkeypatch.setenv(_policy.FEATURE_FLAG_ENV, "true")
    importlib.reload(app_module)
    with_agent = route_paths(app_module.app)

    added = with_agent - without
    assert added, "the agent routes were not mounted"
    assert all(path.startswith("/ops/agent") for path in added)
    assert without - with_agent == set(), "a live route disappeared"


def test_a_failure_to_mount_never_stops_the_api(agent_env, monkeypatch):
    """A pre-scale capability must not be able to take the OCC API down."""
    import operations_control.api.app as app_module
    monkeypatch.setenv(_policy.FEATURE_FLAG_ENV, "true")
    monkeypatch.setattr(
        "operations_control.occ_agent.service.OccAgentService.__init__",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    importlib.reload(app_module)
    assert app_module.mount_occ_agent(app_module.app) is False
    assert "/ops/dashboard" in route_paths(app_module.app)


def test_the_browser_cannot_choose_a_storage_location(api_client):
    """Upload takes files only; the destination is derived server-side."""
    headers = {"X-Operator-Token": "tok-a"}
    created = api_client.post(
        "/ops/agent/cases", headers=headers,
        json={"instruction": "Onboard Northstar Lending. UK equity release. "
                             "Monthly portfolio MI. Portfolio id direct_101. "
                             "First reporting date 2026-06-30."})
    case_id = created.json()["case"]["case_id"]
    api_client.post(f"/ops/agent/cases/{case_id}/requirements/confirm",
                    headers=headers, json={})
    api_client.post(f"/ops/agent/cases/{case_id}/pack/generate", headers=headers,
                    json={})
    api_client.post(f"/ops/agent/cases/{case_id}/pack/approve", headers=headers,
                    json={})
    response = api_client.post(
        f"/ops/agent/cases/{case_id}/artefacts", headers=headers,
        files=[("files", ("loan_extract.csv", b"Loan Identifier\nL1\n",
                          "text/csv"))])
    assert response.status_code == 200
    artefact = response.json()["case"]["received_artefacts"][0]
    # The intended location was DERIVED, from the case's confirmed identity.
    assert artefact["intended_live_uri"].startswith(
        "blob://raw-v2/northstar_lending/direct/funded/monthly/direct_101/"
        "2026-06-30/")
    assert artefact["execution_status"] == "simulated_only"


def test_the_meta_route_exposes_the_lifecycle_and_the_scenarios(api_client):
    body = api_client.get("/ops/agent/meta",
                          headers={"X-Operator-Token": "tok-a"}).json()
    assert body["runtime_mode"] == "synthetic"
    assert len(body["scenarios"]) == 5
    assert len(body["lifecycle"]) == 21
    assert body["policy"]["allow_publish"] is False
