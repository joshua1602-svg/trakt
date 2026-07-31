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
from operations_control.occ_agent import states as _states
from operations_control.occ_agent.scenarios import run_scenario
from operations_control.occ_agent.service import OccAgentService
from operations_control.occ_agent.store import RunNotFound

from .conftest import ACTOR, TENANT_A, TENANT_B


# --------------------------------------------------------------------------- #
# Tenancy at the store
# --------------------------------------------------------------------------- #

def test_one_tenant_cannot_load_another_tenants_case(service):
    agent_case = service.create_case(tenant=TENANT_A, initiating_user=ACTOR,
                                     instruction="Onboard Northstar Lending.")
    with pytest.raises(RunNotFound):
        service.load(TENANT_B, agent_case.case_ref)


def test_a_tenants_case_list_shows_only_its_own(service):
    a = service.create_case(tenant=TENANT_A, initiating_user=ACTOR,
                            instruction="Onboard Northstar Lending.")
    b = service.create_case(tenant=TENANT_B, initiating_user="Bob",
                            instruction="Onboard Harbour Point Capital.")
    assert [row["case_ref"] for row in service.list_cases(TENANT_A)] == \
        [a.case_ref]
    assert [row["case_ref"] for row in service.list_cases(TENANT_B)] == \
        [b.case_ref]


def test_synthetic_files_are_isolated_by_tenant_and_case(service, agent_env):
    a = service.create_case(tenant=TENANT_A, initiating_user=ACTOR,
                            instruction="Onboard Northstar Lending.")
    b = service.create_case(tenant=TENANT_B, initiating_user="Bob",
                            instruction="Onboard Harbour Point Capital.")
    service.artefacts.register(a.run, service.facts(a), filename="a.csv",
                               data=b"x,y\n1,2\n", provided_by=ACTOR)
    service.artefacts.register(b.run, service.facts(b), filename="b.csv",
                               data=b"x,y\n3,4\n", provided_by="Bob")
    dir_a = service.store.artefact_dir(TENANT_A, a.case_ref)
    dir_b = service.store.artefact_dir(TENANT_B, b.case_ref)
    assert dir_a != dir_b
    assert TENANT_A in str(dir_a) and TENANT_B in str(dir_b)
    assert {p.name for p in dir_a.iterdir()} == {"a.csv"}
    assert {p.name for p in dir_b.iterdir()} == {"b.csv"}


def test_a_case_cannot_reach_another_cases_files(service):
    a = service.create_case(tenant=TENANT_A, initiating_user=ACTOR,
                            instruction="Onboard Northstar Lending.")
    b = service.create_case(tenant=TENANT_A, initiating_user=ACTOR,
                            instruction="Onboard Harbour Point Capital.")
    service.artefacts.register(a.run, service.facts(a), filename="secret.csv",
                               data=b"x,y\n1,2\n", provided_by=ACTOR)
    reloaded = service.load(TENANT_A, b.case_ref)
    assert reloaded.run.received_artefacts == []
    assert service._artefact_paths(reloaded.run) == []


def test_a_document_moved_between_tenant_folders_is_refused(service,
                                                            synthetic_store):
    """Defence in depth: the run's own tenant must match where it was found."""
    agent_case = service.create_case(tenant=TENANT_A, initiating_user=ACTOR,
                                     instruction="Onboard Northstar Lending.")
    doc = json.loads(Path(synthetic_store.storage._local_path(
        synthetic_store.run_uri(TENANT_A, agent_case.case_ref))).read_text(
            encoding="utf-8"))
    # Plant tenant A's document in tenant B's folder.
    from operations_control.stores import _write_json
    _write_json(synthetic_store.storage,
                synthetic_store.run_uri(TENANT_B, agent_case.case_ref), doc)
    with pytest.raises(RunNotFound):
        synthetic_store.load(TENANT_B, agent_case.case_ref)


def test_the_audit_trail_is_tamper_evident(service):
    run = run_scenario(service, "scenario_a_clean", tenant=TENANT_A, actor=ACTOR)
    ref = run.case.case_ref
    assert service.store.verify_audit_chain(TENANT_A, ref) is True
    events = service.store.list_audit(TENANT_A, ref)
    assert len(events) > 5
    for event in events:
        assert event["runtime_mode"] == _policy.RUNTIME_MODE_SYNTHETIC
        assert event["execution_classification"]
        assert event["actor_type"] in ("human", "agent", "system")
        assert "event_id" in event and "at" in event and "action" in event


def test_the_audit_trail_records_no_hidden_reasoning(service):
    run = run_scenario(service, "scenario_a_clean", tenant=TENANT_A, actor=ACTOR)
    for event in service.store.list_audit(TENANT_A, run.case.case_ref):
        basis = event.get("decision_basis", "")
        # A concise rationale, not a transcript.
        assert len(basis) < 200, basis
        assert "thinking" not in basis.lower()
        assert "chain of thought" not in basis.lower()


def test_the_onboarding_keeps_its_own_history_alongside(service):
    """Two records, each with its own trail. Neither restates the other."""
    run = run_scenario(service, "scenario_a_clean", tenant=TENANT_A, actor=ACTOR)
    case_events = {e["event"] for e in run.case.case.events}
    assert "opened" in case_events
    assert "status_approved" in case_events
    run_actions = {e["action"] for e in
                   service.store.list_audit(TENANT_A, run.case.case_ref)}
    assert "ready_for_execution" in run_actions
    assert case_events.isdisjoint(run_actions)


def test_a_tampered_audit_record_is_detected(service, synthetic_store):
    agent_case = service.create_case(tenant=TENANT_A, initiating_user=ACTOR,
                                     instruction="Onboard Northstar Lending.")
    uri = synthetic_store.audit_uri(TENANT_A, agent_case.case_ref, 1)
    path = Path(synthetic_store.storage._local_path(uri))
    doc = json.loads(path.read_text(encoding="utf-8"))
    doc["action"] = "something_else"
    path.write_text(json.dumps(doc), encoding="utf-8")
    assert synthetic_store.verify_audit_chain(
        TENANT_A, agent_case.case_ref) is False


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
    from apps.blob_trigger_app.storage import Storage
    import operations_control.api.app as app_module
    importlib.reload(app_module)
    agent_api.configure(OccAgentService(
        Storage(agent_env["blob_root"]),
        container="operations-control-synthetic",
        sandbox=agent_env["sandbox"]))
    return TestClient(app_module.app)


def test_the_routes_are_mounted_inside_the_existing_api(api_client):
    """Same application, same auth — not a second service."""
    paths = route_paths(api_client.app)
    assert "/ops/dashboard" in paths            # the live OCC is still there
    assert "/ops/onboarding/home" in paths      # Client Onboarding too
    assert "/ops/agent/cases" in paths          # and the tab's routes


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
    case_ref = created.json()["case_ref"]
    fetched = api_client.get(f"/ops/agent/cases/{case_ref}", headers=headers)
    assert fetched.status_code == 200
    body = fetched.json()
    assert body["run"]["runtime_mode"] == "synthetic"
    assert body["policy"]["allow_live_blob_write"] is False
    assert body["policy"]["allow_activate_configuration"] is False
    assert body["configuration_written"] is False
    # The onboarding case is presented alongside, in its own shape.
    assert body["onboarding"]["client_name"] == "Northstar Lending"
    assert body["onboarding"]["status"] == "draft"


def test_the_status_links_to_the_onboarding_screens(api_client):
    headers = {"X-Operator-Token": "tok-a"}
    created = api_client.post("/ops/agent/cases", headers=headers,
                              json={"instruction": "Onboard Northstar Lending."})
    body = created.json()
    links = {link["to"] for link in body["occ_links"]}
    assert f"/onboarding/{body['case_ref']}" in links


def test_one_operator_cannot_read_another_tenants_case(api_client):
    created = api_client.post("/ops/agent/cases",
                              headers={"X-Operator-Token": "tok-a"},
                              json={"instruction": "Onboard Northstar Lending."})
    case_ref = created.json()["case_ref"]
    # Bob is bound to a different client and must not see it — 404, not 403.
    denied = api_client.get(f"/ops/agent/cases/{case_ref}",
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
    assert body["scenario"]["state"] == "READY_FOR_EXECUTION"
    assert body["scenario"]["onboarding_status"] == "approved"
    assert body["run"]["state"] == "READY_FOR_EXECUTION"
    assert body["readiness"]["ready"] is True

    case_ref = body["case_ref"]
    package = api_client.get(f"/ops/agent/cases/{case_ref}/readiness",
                             headers=headers).json()
    manifest = package["package"]["execution_manifest"]
    assert manifest["execution_performed"] is False
    assert manifest["configuration_activated"] is False


def test_the_preview_route_creates_nothing(api_client):
    headers = {"X-Operator-Token": "tok-a"}
    ran = api_client.post("/ops/agent/scenarios/run", headers=headers,
                          json={"fixture_id": "scenario_a_clean"}).json()
    body = api_client.get(f"/ops/agent/cases/{ran['case_ref']}/preview",
                          headers=headers).json()
    assert body["written"] is False
    assert body["execution_status"] == "not_activated"
    assert body["preview"]["artefacts"]
    assert body["preview"]["current_version"] == 0


def test_there_is_no_activation_route(api_client):
    paths = {p for p in route_paths(api_client.app) if p.startswith("/ops/agent")}
    assert not any("activate" in p for p in paths)


def test_an_illegal_transition_is_refused_by_the_api(api_client):
    headers = {"X-Operator-Token": "tok-a"}
    created = api_client.post("/ops/agent/cases", headers=headers,
                              json={"instruction": "Onboard Northstar Lending."})
    case_ref = created.json()["case_ref"]
    # Readiness cannot be approved from the very first state.
    refused = api_client.post(f"/ops/agent/cases/{case_ref}/readiness/approve",
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
    assert "/ops/onboarding/home" in paths      # including Client Onboarding


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
    case_ref = created.json()["case_ref"]
    response = api_client.post(
        f"/ops/agent/cases/{case_ref}/artefacts", headers=headers,
        files=[("files", ("loan_extract.csv", b"Loan Identifier\nL1\n",
                          "text/csv"))])
    assert response.status_code == 200
    artefact = response.json()["run"]["received_artefacts"][0]
    # The intended location was DERIVED, from the onboarding case's identity.
    assert artefact["intended_live_uri"].startswith(
        "blob://raw-v2/NORTHSTAR/direct/funded/monthly/direct_101/2026-06-30/")
    assert artefact["execution_status"] == "simulated_only"


def test_the_meta_route_exposes_the_lifecycle_and_the_scenarios(api_client):
    body = api_client.get("/ops/agent/meta",
                          headers={"X-Operator-Token": "tok-a"}).json()
    assert body["runtime_mode"] == "synthetic"
    assert len(body["scenarios"]) == 5
    assert len(body["lifecycle"]) == len(_states.STATE_ORDER)
    assert body["policy"]["allow_publish"] is False
    # The wizard's own reference data, so the tab never restates the catalogue.
    assert body["onboarding_reference"]["steps"]
    assert body["onboarding_reference"]["catalogue"]["sections"]
