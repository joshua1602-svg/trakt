"""The delivery case file: a governed workflow the operator can see, not a
decision form with no context.

What is proven here:

  * every delivery reports the same seven steps, in order, with a state each;
  * completed steps stay present — nothing disappears once it is behind you;
  * the step the delivery is AT is the one the page should open on;
  * the approval step carries its evidence, its question and its scopes;
  * the platform-wide scope is not offered on a delivery, and the backend
    refuses it if one is sent anyway;
  * an approval records how far the operator intended it to reach, and audits it.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from operations_control.api import app as app_module
from operations_control.api import workflow_view
from operations_control.contracts import (
    PUBLICATION_SCOPES,
    RUN_AWAITING_PUBLICATION,
)
from operations_control.engine import OpsError

from .conftest import (
    OP_A,
    make_client_config,
    make_engine,
    register_and_create,
    start_and_wait,
)

EXPECTED_STEPS = [
    "files_received", "delivery_identified", "configuration_selected",
    "data_assessed", "issues_reviewed", "publication_approval", "published",
]

EXPECTED_LABELS = [
    "Files received", "Delivery identified", "Configuration selected",
    "Data assessed", "Issues reviewed", "Publication approval", "Published",
]


@pytest.fixture()
def ready(store, source_registry, delivery_dir, tmp_path):
    """One delivery parked at the publication decision."""
    engine = make_engine(store, source_registry, "happy",
                         client_config=make_client_config(tmp_path))
    run = register_and_create(engine, delivery_dir, client_id="client_a")
    start_and_wait(engine, run, statuses=(RUN_AWAITING_PUBLICATION,))
    app_module.set_engine(engine)
    client = TestClient(app_module.app, raise_server_exceptions=False)
    yield {"client": client, "engine": engine, "run": run}
    app_module.set_engine(None)


def _workflow(api) -> dict:
    r = api["client"].get(f"/ops/workflows/{api['run'].workflow_id}",
                          headers={"X-Operator-Token": OP_A})
    assert r.status_code == 200, r.text
    return r.json()["workflow"]


class TestCaseFileShape:
    def test_every_step_is_present_in_order(self, ready):
        steps = _workflow(ready)["steps"]
        assert [s["key"] for s in steps] == EXPECTED_STEPS
        assert [s["label"] for s in steps] == EXPECTED_LABELS

    def test_every_step_reports_one_of_the_five_states(self, ready):
        allowed = {"complete", "current", "pending", "blocked", "not_applicable"}
        for step in _workflow(ready)["steps"]:
            assert step["status"] in allowed
            assert step["status_label"]

    def test_completed_steps_keep_their_content(self, ready):
        steps = {s["key"]: s for s in _workflow(ready)["steps"]}
        identified = steps["delivery_identified"]
        assert identified["status"] == "complete"
        facts = {f["label"]: f["value"] for f in identified["facts"]}
        assert facts["Client"] == "client_a"
        assert facts["Portfolio"] == "pf1"
        assert facts["Reporting period"] == "2026-06-30"
        assert facts["Book"] == "Funded book"
        assert facts["How this was decided"], "the basis must be stated"

    def test_the_current_step_is_where_the_page_opens(self, ready):
        workflow = _workflow(ready)
        assert workflow["current_step"] == "publication_approval"
        steps = {s["key"]: s for s in workflow["steps"]}
        assert steps["publication_approval"]["status"] == "current"
        assert steps["published"]["status"] == "pending"

    def test_files_received_reports_the_pack_it_ran_on(self, ready):
        step = next(s for s in _workflow(ready)["steps"]
                    if s["key"] == "files_received")
        assert step["status"] == "complete"
        assert [f["filename"] for f in step["files"]] == ["loans.csv"]


class TestApprovalStep:
    def test_the_decision_carries_its_evidence_and_question(self, ready):
        approval = next(s for s in _workflow(ready)["steps"]
                        if s["key"] == "publication_approval")["approval"]
        assert approval["available"] is True
        assert approval["headline"] == "Ready to publish"
        assert any("file" in line for line in approval["evidence_lines"])
        assert any("blocking issue" in line
                   for line in approval["evidence_lines"])
        assert approval["question"] == (
            "Publish this delivery as the latest official version?")

    def test_the_default_scope_is_this_delivery_only(self, ready):
        approval = next(s for s in _workflow(ready)["steps"]
                        if s["key"] == "publication_approval")["approval"]
        assert approval["default_scope"] == "delivery"
        assert approval["scopes"][0]["value"] == "delivery"
        assert approval["scopes"][0]["label"] == "No — this delivery only"

    def test_the_platform_wide_scope_is_not_offered(self, ready):
        approval = next(s for s in _workflow(ready)["steps"]
                        if s["key"] == "publication_approval")["approval"]
        values = [s["value"] for s in approval["scopes"]]
        assert values == ["delivery", "portfolio", "client"]
        assert "global" not in values
        assert not any("All of Trakt" in s["label"] for s in approval["scopes"])

    def test_the_consequence_is_stated_before_confirming(self, ready):
        approval = next(s for s in _workflow(ready)["steps"]
                        if s["key"] == "publication_approval")["approval"]
        assert "latest official version" in approval["consequence"]
        assert approval["scope_consequences"]["delivery"] == (
            "This decision applies only to this delivery.")


class TestApprovalScopeIsGoverned:
    def test_publishing_defaults_to_this_delivery_only(self, ready):
        r = ready["client"].post(
            f"/ops/workflows/{ready['run'].workflow_id}/publish",
            headers={"X-Operator-Token": OP_A}, json={})
        assert r.status_code == 200, r.text
        pub = ready["engine"].store.load_publication("client_a", "2026-06-30")
        assert pub["approval_scope"] == "delivery"

    def test_a_wider_scope_is_recorded_and_audited(self, ready):
        r = ready["client"].post(
            f"/ops/workflows/{ready['run'].workflow_id}/publish",
            headers={"X-Operator-Token": OP_A},
            json={"remember_scope": "portfolio"})
        assert r.status_code == 200, r.text
        pub = ready["engine"].store.load_publication("client_a", "2026-06-30")
        assert pub["approval_scope"] == "portfolio"
        entry = next(a for a in ready["engine"].store.list_audit("client_a")
                     if a["event"] == "publication_approved")
        assert entry["detail"]["approval_scope"] == "portfolio"
        assert ready["engine"].store.verify_audit_chain("client_a")

    def test_a_platform_wide_scope_is_refused_by_the_backend(self, ready):
        r = ready["client"].post(
            f"/ops/workflows/{ready['run'].workflow_id}/publish",
            headers={"X-Operator-Token": OP_A},
            json={"remember_scope": "global"})
        assert r.status_code == 400
        assert r.json()["errorCode"] == "OPS_BAD_SCOPE"
        # Nothing was published as a side effect of the refusal.
        pub = ready["engine"].store.load_publication("client_a", "2026-06-30")
        assert pub["status"] == "prepared"

    def test_only_the_declared_scopes_exist(self):
        assert PUBLICATION_SCOPES == ("delivery", "portfolio", "client")

    def test_the_engine_refuses_an_unknown_scope_directly(self, ready):
        with pytest.raises(OpsError) as exc:
            ready["engine"].approve_publication(
                client_id="client_a", workflow_id=ready["run"].workflow_id,
                actor="alice", remember_scope="everything")
        assert exc.value.code == "OPS_BAD_SCOPE"


class TestPublishedStep:
    def test_the_published_step_records_what_happened(self, ready):
        ready["client"].post(
            f"/ops/workflows/{ready['run'].workflow_id}/publish",
            headers={"X-Operator-Token": OP_A},
            json={"remember_scope": "client"})
        steps = {s["key"]: s for s in _workflow(ready)["steps"]}
        published = steps["published"]
        assert published["status"] == "complete"
        facts = {f["label"]: f["value"] for f in published["facts"]}
        assert facts["Published by"] == "Alice"
        assert facts["Decision applied to"] == "Future deliveries for this client"
        assert facts["Audit reference"].startswith("pub_")
        assert steps["publication_approval"]["status"] == "complete"


class TestDeepLinks:
    def test_a_stage_maps_onto_the_step_that_owns_it(self):
        assert workflow_view.step_for_stage("publication") == \
            "publication_approval"
        assert workflow_view.step_for_stage("received") == "files_received"
        assert workflow_view.step_for_stage("validation") == "data_assessed"
