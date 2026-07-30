"""Withdrawing a delivery: out of active work, never out of the record.

A delivery created by mistake, run as a test, sent twice or simply no longer
wanted has to be removable from the queues without being published and without
losing anything. What is proven here:

  * it leaves every actionable queue — attention, review, ready-to-publish;
  * it keeps its files, its run manifest and its whole audit trail;
  * who withdrew it, when and why are all recorded, and the audit chain holds;
  * it cannot be published, re-run or reopened by an ordinary action;
  * it is still reachable, and says plainly what happened, when opened later.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from operations_control.api import app as app_module
from operations_control.contracts import (
    RUN_AWAITING_PUBLICATION,
    RUN_WITHDRAWN,
    WITHDRAWAL_REASON_CODES,
)
from operations_control.engine import OpsError

from .conftest import (
    OP_A,
    OP_B,
    make_client_config,
    make_engine,
    register_and_create,
    start_and_wait,
)


def _h(token: str):
    return {"X-Operator-Token": token}


@pytest.fixture()
def ready(store, source_registry, delivery_dir, tmp_path):
    """A delivery parked at the publication decision, plus an API bound to it."""
    engine = make_engine(store, source_registry, "happy",
                         client_config=make_client_config(tmp_path))
    run = register_and_create(engine, delivery_dir, client_id="client_a")
    start_and_wait(engine, run, statuses=(RUN_AWAITING_PUBLICATION,))
    app_module.set_engine(engine)
    client = TestClient(app_module.app, raise_server_exceptions=False)
    yield {"client": client, "engine": engine, "run": run, "store": store}
    app_module.set_engine(None)


@pytest.fixture()
def needs_review(store, source_registry, delivery_dir, tmp_path):
    """A delivery stopped at a question, so its decisions are still open."""
    engine = make_engine(store, source_registry, "mapping_halt",
                         client_config=make_client_config(tmp_path))
    run = register_and_create(engine, delivery_dir, client_id="client_a")
    start_and_wait(engine, run)
    app_module.set_engine(engine)
    client = TestClient(app_module.app, raise_server_exceptions=False)
    yield {"client": client, "engine": engine, "run": run, "store": store}
    app_module.set_engine(None)


def _withdraw(api, *, token=OP_A, reason="test_delivery", note=""):
    return api["client"].post(
        f"/ops/workflows/{api['run'].workflow_id}/withdraw",
        headers=_h(token), json={"reason_code": reason, "note": note})


class TestItLeavesActiveWork:
    def test_it_disappears_from_the_attention_queue_and_tiles(self, ready):
        before = ready["client"].get("/ops/dashboard", headers=_h(OP_A)).json()
        assert before["tiles"]["ready_to_publish"] == 1

        assert _withdraw(ready).status_code == 200

        after = ready["client"].get("/ops/dashboard", headers=_h(OP_A)).json()
        assert after["tiles"]["ready_to_publish"] == 0
        assert after["tiles"]["needs_attention"] == 0
        assert after["tiles"]["blocked"] == 0
        assert after["needs_attention"] == []

    def test_its_questions_leave_the_review_queue_with_it(self, needs_review):
        before = needs_review["client"].get("/ops/reviews",
                                            headers=_h(OP_A)).json()["reviews"]
        assert before, "the fixture should have left questions open"

        assert _withdraw(needs_review, reason="uploaded_in_error").status_code == 200

        after = needs_review["client"].get("/ops/reviews",
                                           headers=_h(OP_A)).json()["reviews"]
        assert after == [], "a withdrawn delivery's questions are still queued"

    def test_it_is_not_held_blocked_or_ready(self, ready):
        _withdraw(ready)
        for status in ("held", "blocked", "awaiting_publication",
                       "needs_review"):
            rows = ready["client"].get(f"/ops/workflows?status={status}",
                                       headers=_h(OP_A)).json()["workflows"]
            assert rows == [], f"it still appears as {status}"

    def test_it_remains_reachable(self, ready):
        _withdraw(ready)
        rows = ready["client"].get("/ops/workflows?status=withdrawn",
                                   headers=_h(OP_A)).json()["workflows"]
        assert [r["workflow_id"] for r in rows] == [ready["run"].workflow_id]

        direct = ready["client"].get(
            f"/ops/workflows/{ready['run'].workflow_id}", headers=_h(OP_A))
        assert direct.status_code == 200
        assert direct.json()["workflow"]["status"] == RUN_WITHDRAWN


class TestNothingIsLost:
    def test_the_files_manifest_and_results_stay(self, ready):
        run = ready["run"]
        before = ready["client"].get(f"/ops/workflows/{run.workflow_id}",
                                     headers=_h(OP_A)).json()["workflow"]
        files_before = next(s for s in before["steps"]
                            if s["key"] == "files_received")["files"]
        assert files_before

        _withdraw(ready)

        after = ready["client"].get(f"/ops/workflows/{run.workflow_id}",
                                    headers=_h(OP_A)).json()["workflow"]
        files_after = next(s for s in after["steps"]
                           if s["key"] == "files_received")["files"]
        assert files_after == files_before
        # The assessment that was done is still described.
        assessed = next(s for s in after["steps"] if s["key"] == "data_assessed")
        assert assessed["summary"]
        # And the stored governed results are untouched.
        assert ready["store"].load_result(run.client_id, run.workflow_id,
                                          "understanding") is not None

    def test_who_when_and_why_are_recorded_and_the_chain_holds(self, ready):
        _withdraw(ready, reason="duplicate_delivery", note="sent twice")

        run = ready["store"].load_workflow("client_a",
                                           ready["run"].workflow_id)
        assert run.withdrawn["by"] == "Alice"
        assert run.withdrawn["reason_code"] == "duplicate_delivery"
        assert run.withdrawn["reason_label"] == "Duplicate delivery"
        assert run.withdrawn["note"] == "sent twice"
        assert run.withdrawn["at"]

        entry = next(a for a in ready["store"].list_audit("client_a")
                     if a["event"] == "workflow_withdrawn")
        assert entry["actor"] == "Alice"
        assert entry["detail"]["reason_label"] == "Duplicate delivery"
        assert ready["store"].verify_audit_chain("client_a")

    def test_the_case_file_says_plainly_what_happened(self, ready):
        _withdraw(ready, reason="test_delivery")
        workflow = ready["client"].get(
            f"/ops/workflows/{ready['run'].workflow_id}",
            headers=_h(OP_A)).json()["workflow"]

        assert "withdrew this delivery" in workflow["status_sentence"]
        assert workflow["withdrawn"]["reason_label"] == "Test delivery"
        approval = next(s for s in workflow["steps"]
                        if s["key"] == "publication_approval")
        assert approval["status"] == "not_applicable"
        assert "will not be published" in approval["summary"]
        facts = {f["label"]: f["value"] for f in approval["facts"]}
        assert facts["Withdrawn by"] == "Alice"
        assert facts["Reason"] == "Test delivery"


class TestItCannotBePublished:
    def test_publishing_a_withdrawn_delivery_is_refused(self, ready):
        _withdraw(ready)
        r = ready["client"].post(
            f"/ops/workflows/{ready['run'].workflow_id}/publish",
            headers=_h(OP_A), json={})
        assert r.status_code == 409
        assert r.json()["errorCode"] == "OPS_WORKFLOW_WITHDRAWN"

    def test_the_prepared_publication_is_stood_down(self, ready):
        _withdraw(ready)
        pub = ready["store"].load_publication("client_a", "2026-06-30")
        assert pub["status"] == "withdrawn"
        assert pub["withdrawn_by"] == "Alice"
        history = ready["client"].get("/ops/history",
                                      headers=_h(OP_A)).json()["history"]
        assert all(p["status"] != "published" for p in history)

    def test_it_cannot_be_run_again(self, ready):
        _withdraw(ready)
        r = ready["client"].post(
            f"/ops/workflows/{ready['run'].workflow_id}/rerun",
            headers=_h(OP_A))
        assert r.status_code == 409
        assert r.json()["errorCode"] == "OPS_WORKFLOW_WITHDRAWN"

    def test_reopening_is_not_an_ordinary_transition(self):
        from operations_control.contracts import RUN_TRANSITIONS
        assert RUN_TRANSITIONS[RUN_WITHDRAWN] == ()


class TestTheReasonIsGoverned:
    def test_a_reason_is_required(self, ready):
        r = _withdraw(ready, reason="")
        assert r.status_code == 400
        assert r.json()["errorCode"] == "OPS_BAD_REASON"
        assert ready["store"].load_workflow(
            "client_a", ready["run"].workflow_id).status != RUN_WITHDRAWN

    def test_an_unknown_reason_is_refused(self, ready):
        assert _withdraw(ready, reason="because").status_code == 400

    def test_other_needs_a_note(self, ready):
        r = _withdraw(ready, reason="other")
        assert r.status_code == 400
        assert r.json()["errorCode"] == "OPS_REASON_NOTE_REQUIRED"
        assert _withdraw(ready, reason="other",
                         note="wrong quarter").status_code == 200

    def test_the_offered_reasons_are_the_governed_ones(self, ready):
        offered = ready["client"].get("/ops/withdrawal-reasons",
                                      headers=_h(OP_A)).json()
        assert [r["value"] for r in offered["reasons"]] == \
            list(WITHDRAWAL_REASON_CODES)
        assert offered["note_required_for"] == "other"

    def test_a_published_delivery_cannot_be_withdrawn(self, ready):
        ready["client"].post(
            f"/ops/workflows/{ready['run'].workflow_id}/publish",
            headers=_h(OP_A), json={})
        r = _withdraw(ready)
        assert r.status_code == 409
        assert r.json()["errorCode"] == "OPS_ALREADY_PUBLISHED"

    def test_withdrawing_twice_is_harmless(self, ready):
        assert _withdraw(ready).status_code == 200
        assert _withdraw(ready).status_code == 200

    def test_the_engine_refuses_an_unknown_reason_directly(self, ready):
        run = ready["store"].load_workflow("client_a",
                                           ready["run"].workflow_id)
        with pytest.raises(OpsError) as exc:
            ready["engine"].withdraw(run, actor="alice", reason_code="nope")
        assert exc.value.code == "OPS_BAD_REASON"


class TestTenancy:
    def test_another_clients_operator_cannot_withdraw(self, ready):
        r = _withdraw(ready, token=OP_B)
        assert r.status_code == 404
        assert ready["store"].load_workflow(
            "client_a", ready["run"].workflow_id).status == \
            RUN_AWAITING_PUBLICATION
