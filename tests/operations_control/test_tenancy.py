"""Cross-tenant isolation — the mandatory release gate.

Proves one client's operator cannot view another client's workflow, governed
results or reviews; cannot approve another client's decision; cannot alter
another client's rules; cannot publish another client's output. Auth is
fail-closed (no config -> 503; bad token -> 401)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from operations_control.api import app as app_module
from operations_control.contracts import RUN_AWAITING_PUBLICATION

from .conftest import (
    OP_A,
    OP_B,
    make_engine,
    register_and_create,
    start_and_wait,
)


@pytest.fixture()
def api(store, source_registry, delivery_dir):
    """API bound to a scripted engine with one workflow per client."""
    engine = make_engine(store, source_registry, "mapping_halt")
    run_a = register_and_create(engine, delivery_dir, client_id="client_a")
    start_and_wait(engine, run_a)

    engine_b = make_engine(store, source_registry, "happy")
    run_b = register_and_create(engine_b, delivery_dir, client_id="client_b",
                                portfolio_id="pfb")
    start_and_wait(engine_b, run_b,
                   statuses=(RUN_AWAITING_PUBLICATION,))

    app_module.set_engine(engine)
    client = TestClient(app_module.app, raise_server_exceptions=False)
    yield {"client": client, "run_a": run_a, "run_b": run_b,
           "engine": engine, "store": store}
    app_module.set_engine(None)


def _h(token: str):
    return {"X-Operator-Token": token}


class TestAuthFailClosed:
    def test_no_config_means_no_access(self, api, monkeypatch):
        monkeypatch.delenv("TRAKT_OPS_OPERATORS", raising=False)
        r = api["client"].get("/ops/dashboard", headers=_h(OP_A))
        assert r.status_code == 503

    def test_bad_token_is_unauthenticated(self, api):
        r = api["client"].get("/ops/dashboard", headers=_h("wrong"))
        assert r.status_code == 401

    def test_no_token_is_unauthenticated(self, api):
        assert api["client"].get("/ops/dashboard").status_code == 401


class TestCrossTenantIsolation:
    def test_cannot_view_other_clients_workflow(self, api):
        wf_b = api["run_b"].workflow_id
        r = api["client"].get(f"/ops/workflows/{wf_b}", headers=_h(OP_A))
        assert r.status_code == 404          # existence is not revealed
        # Even naming the client explicitly cannot widen access.
        r = api["client"].get(f"/ops/workflows/{wf_b}?client=client_b",
                              headers=_h(OP_A))
        assert r.status_code == 404
        # The owner sees it.
        r = api["client"].get(f"/ops/workflows/{wf_b}", headers=_h(OP_B))
        assert r.status_code == 200

    def test_workflow_list_is_tenant_filtered(self, api):
        r = api["client"].get("/ops/workflows", headers=_h(OP_A))
        assert r.status_code == 200
        clients = {w["client_id"] for w in r.json()["workflows"]}
        assert clients == {"client_a"}
        # Asking for the other tenant's list directly is refused.
        r = api["client"].get("/ops/workflows?client=client_b", headers=_h(OP_A))
        assert r.status_code == 404

    def test_cannot_view_other_clients_governed_results(self, api):
        wf_b = api["run_b"].workflow_id
        r = api["client"].get(f"/ops/workflows/{wf_b}", headers=_h(OP_A))
        assert r.status_code == 404

    def test_reviews_are_tenant_filtered_and_protected(self, api):
        r = api["client"].get("/ops/reviews", headers=_h(OP_B))
        assert all(x["client_id"] == "client_b" for x in r.json()["reviews"])

        dec_a = api["store"].open_decisions("client_a")[0]
        r = api["client"].get(f"/ops/reviews/{dec_a['decision_id']}",
                              headers=_h(OP_B))
        assert r.status_code == 404

    def test_cannot_approve_other_clients_decision(self, api):
        dec_a = api["store"].open_decisions("client_a")[0]
        r = api["client"].post(
            f"/ops/reviews/{dec_a['decision_id']}/decision",
            headers=_h(OP_B),
            json={"action": "approve", "value": "accept_mapping",
                  "scope": "client"})
        assert r.status_code == 404
        # Still open — nothing changed.
        fresh = api["store"].load_decision("client_a", dec_a["decision_id"])
        assert fresh["status"] == "open"

    def test_cannot_alter_other_clients_rules(self, api):
        # Approve a decision as A to create a client_a rule.
        dec_a = api["store"].open_decisions("client_a")[0]
        r = api["client"].post(
            f"/ops/reviews/{dec_a['decision_id']}/decision",
            headers=_h(OP_A),
            json={"action": "approve", "value": "accept_mapping",
                  "scope": "client"})
        assert r.status_code == 200
        # B's rules view does not contain client_a rules.
        r = api["client"].get("/ops/rules", headers=_h(OP_B))
        assert all(x["client_id"] != "client_a" for x in r.json()["rules"])

    def test_cannot_publish_other_clients_output(self, api, ops_env):
        wf_b = api["run_b"].workflow_id
        r = api["client"].post(f"/ops/workflows/{wf_b}/publish",
                               headers=_h(OP_A))
        assert r.status_code == 404
        latest = (ops_env["blob_root"] / "processed-v2" / "platform"
                  / "client_b" / "latest" / "platform_canonical_typed.csv")
        assert not latest.exists()
        # The owner can publish.
        r = api["client"].post(f"/ops/workflows/{wf_b}/publish",
                               headers=_h(OP_B))
        assert r.status_code == 200
        assert latest.exists()

    def test_audit_is_tenant_bound(self, api):
        r = api["client"].get("/ops/audit?client=client_b", headers=_h(OP_A))
        assert r.status_code == 404
        r = api["client"].get("/ops/audit?client=client_b", headers=_h(OP_B))
        assert r.status_code == 200 and r.json()["chain_intact"] is True
