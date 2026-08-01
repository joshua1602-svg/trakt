"""The concentration-test review surface API: tenant-bound operator review,
administrator-gated approval and activation, and refusals that explain
themselves. Same auth model as the rest of Operations Control — outside the
binding is 404, never 403."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from operations_control.api import app as app_module

from .conftest import OP_A, OP_ADMIN, OP_B, make_engine

ADMIN = {"X-Operator-Token": OP_ADMIN}
ALICE = {"X-Operator-Token": OP_A}
BOB = {"X-Operator-Token": OP_B}

CLAUSE = ("1.1 The aggregate Current Balance of Loans secured on Properties "
          "located in East Anglia must not exceed 25% of the Portfolio.")
NET_WAC = "5.1 The Net WAC of the Portfolio must be at least 3.75%."


@pytest.fixture()
def api(store, source_registry):
    engine = make_engine(store, source_registry)
    app_module.set_engine(engine)
    client = TestClient(app_module.app, raise_server_exceptions=False)
    try:
        yield client
    finally:
        app_module.set_engine(None)


def extract(client, headers=ALICE, text=CLAUSE, tenant="client_a"):
    r = client.post(f"/ops/concentration/{tenant}/extract",
                    json={"source_reference": "covenant.txt", "text": text},
                    headers=headers)
    assert r.status_code == 200, r.text
    return r.json()["proposals"]


class TestAccess:
    def test_unauthenticated_requests_are_refused(self, api):
        assert api.get("/ops/concentration/client_a/review").status_code == 401

    def test_a_client_outside_the_binding_does_not_exist(self, api):
        assert api.get("/ops/concentration/client_a/review",
                       headers=BOB).status_code == 404

    def test_operators_can_review_but_not_approve(self, api):
        [p] = extract(api)
        r = api.post(
            f"/ops/concentration/client_a/proposals/{p['proposal_id']}/approve",
            json={}, headers=ALICE)
        assert r.status_code == 403
        assert r.json()["detail"]["errorCode"] == "OPS_ADMIN_REQUIRED"

    def test_activation_requires_an_administrator(self, api):
        r = api.post("/ops/concentration/client_a/activate", json={},
                     headers=ALICE)
        assert r.status_code == 403


class TestLifecycleOverHttp:
    def test_extract_review_approve_activate(self, api):
        proposals = extract(api, text=CLAUSE + "\n" + NET_WAC)
        review = api.get("/ops/concentration/client_a/review",
                         headers=ALICE).json()
        assert review["countsByStatus"]["pending_approval"] == 1
        assert review["countsByStatus"]["pending_confirmation"] == 1
        assert review["currentVersion"] is None

        wac = next(p for p in proposals
                   if p["proposed_metric_id"] == "rate_net_wac")
        ea = next(p for p in proposals
                  if p["proposed_metric_id"] == "geo_region_share")

        # Approving the ambiguous one is refused with the reason.
        refused = api.post(
            f"/ops/concentration/client_a/proposals/{wac['proposal_id']}/approve",
            json={}, headers=ADMIN)
        assert refused.status_code == 409
        assert "Unresolved concerns" in refused.json()["detail"]["message"]

        # Answer + edit resolves it.
        api.post(
            f"/ops/concentration/client_a/proposals/{wac['proposal_id']}/answer",
            json={"question": wac["confirmation_questions"][0],
                  "answer": "Servicing only, 50bp."},
            headers=ALICE)
        updated = api.post(
            f"/ops/concentration/client_a/proposals/{wac['proposal_id']}/update",
            json={"parameters": {"deduction_percent": 0.5,
                                 "deduction_basis": "servicing_fee_only"},
                  "resolved_concerns": ["net_wac_definition_uncertain"]},
            headers=ALICE).json()
        assert updated["status"] == "pending_approval"
        assert updated["version"] == 2

        for pid in (wac["proposal_id"], ea["proposal_id"]):
            r = api.post(
                f"/ops/concentration/client_a/proposals/{pid}/approve",
                json={"effective_date": "2026-01-31",
                      "comments": "Reviewed against the schedule."},
                headers=ADMIN)
            assert r.status_code == 200, r.text
            assert r.json()["approval"]["operator"] == "Administrator"

        activated = api.post("/ops/concentration/client_a/activate",
                             json={"reason": "Initial activation"},
                             headers=ADMIN)
        assert activated.status_code == 200
        body = activated.json()
        assert body["version"] == 1
        assert len(body["tests"]) == 2
        assert body["content_hash"]

        versions = api.get("/ops/concentration/client_a/versions",
                           headers=ALICE).json()["versions"]
        assert [v["version"] for v in versions] == [1]
        v1 = api.get("/ops/concentration/client_a/versions/1",
                     headers=ALICE).json()
        assert v1["status"] == "active"

    def test_reject_unsupported_not_applicable_and_clarify(self, api):
        for action, body in (("reject", {"reason": "not in facility"}),
                             ("unsupported", {"reason": "no metric"}),
                             ("not-applicable", {"reason": "repaid"}),
                             ("clarify", {"question": "Which basis?"})):
            [p] = extract(api)
            r = api.post(
                f"/ops/concentration/client_a/proposals/{p['proposal_id']}/{action}",
                json=body, headers=ALICE)
            assert r.status_code == 200, (action, r.text)

    def test_extraction_with_nothing_to_read_is_a_409(self, api):
        r = api.post("/ops/concentration/client_a/extract",
                     json={"source_reference": "x"}, headers=ALICE)
        assert r.status_code == 409

    def test_an_unknown_proposal_is_404(self, api):
        r = api.get("/ops/concentration/client_a/proposals/ctp_missing",
                    headers=ALICE)
        assert r.status_code == 404

    def test_the_library_is_readable_for_review(self, api):
        r = api.get("/ops/concentration/client_a/library", headers=ALICE)
        assert r.status_code == 200
        metrics = r.json()["metrics"]
        assert any(m["metric_id"] == "geo_region_share" for m in metrics)
