"""MI Query live telemetry — the Day-1 calibration feedback loop.

Nine proofs, one per acceptance requirement:

A answered query recorded in full        B refusal recorded with its reason
C error recorded with a safe class       D operator review, answer untouched
E client isolation                       F reproducible data version
G period filtering                       H external-safe calibration export
I MI Query behaviour unchanged

The records are built by the same projection the service uses, from real
``GovernedResult`` values, so a change to the governed envelope shows up here
rather than in production.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from mi_agent_api import query_telemetry as qt
from operations_control.api import app as app_module
from trakt_core.envelope import AuditMetadata, GovernedResult, SnapshotRef
from trakt_core.errors import ErrorCode, TraktError

from .conftest import OP_A, OP_ALL, OP_B, make_engine

SNAPSHOT = SnapshotRef(snapshot_id="snap_2025_11_30", content_hash="sha256:abc123",
                       source_kind="platform_canonical")

#: A realistic analytical envelope: the answer the user saw, the structured
#: interpretation the parser produced, and the route that ran.
ANSWER_PAYLOAD = {
    "ok": True,
    "question": "What is the weighted average LTV for the funded book?",
    "answer": "The weighted average LTV is 42.7% across 1,204 loans "
              "(£184,220,110 balance).",
    # The real parser's shape, as produced by the live engine.
    "spec": {"metric": "current_loan_to_value", "aggregation": "weighted_avg",
             "weight_field": "current_outstanding_balance",
             "filters": {"account_status": "LIVE"}, "intent": "summary",
             "dimension": None, "as_of_date": "2025-11-30",
             "route_id": "portfolio_summary"},
    "artifacts": [{"type": "kpi", "value": 42.7}],
    "metadata": {"engine": "mi_agent", "route": "portfolio_summary",
                 "asOfDate": "2025-11-30", "view": "funded",
                 "lensApplied": True,
                 "parserProvenance": {"parser_used": "deterministic",
                                      "llm_failure": None,
                                      "parser_mode_detail": ""}},
    "warnings": [],
}


def _audit(client="client_a", actor="alice@lender.example", ms=812,
           outcome="success", error_code=None, started=None):
    return AuditMetadata(
        capability="mi_query", request_id="req_1", tenant_id=client,
        actor_id=actor, actor_type="user", channel="react", outcome=outcome,
        started_at=started or datetime.now(timezone.utc).isoformat(),
        duration_ms=ms, correlation_id="corr_1", portfolio_id="direct_001",
        snapshot_id=SNAPSHOT.snapshot_id, error_code=error_code)


def _answered(client="client_a", actor="alice@lender.example", started=None):
    return GovernedResult(
        capability="mi_query", status="success", request_id="req_1",
        correlation_id="corr_1", tenant_id=client, portfolio_id="direct_001",
        snapshot=SNAPSHOT, result=dict(ANSWER_PAYLOAD), warnings=(),
        audit=_audit(client, actor, started=started))


def _refused(code=ErrorCode.UNSUPPORTED_QUESTION, client="client_a"):
    err = TraktError(code, "That question is not supported.", request_id="req_2")
    payload = {"ok": False, "error": "That question is not supported.",
               "question": "Can you predict next quarter's defaults?",
               "answer": "", "spec": {}, "artifacts": [],
               "metadata": {"route": "", "controlledUnsupported": True}}
    return GovernedResult(
        capability="mi_query", status="error", request_id="req_2",
        correlation_id="corr_2", tenant_id=client, portfolio_id="direct_001",
        snapshot=SNAPSHOT, result=payload, warnings=(), error=err,
        audit=_audit(client, ms=91, outcome="error", error_code=code))


def _errored(client="client_a"):
    err = TraktError(ErrorCode.STORAGE_UNAVAILABLE,
                     "The governed data store is currently unavailable.",
                     request_id="req_3")
    payload = {"ok": False, "error": "The governed data store is currently "
                                     "unavailable.",
               "question": "What is the total balance?", "answer": "",
               "spec": {}, "artifacts": [], "metadata": {}}
    return GovernedResult(
        capability="mi_query", status="error", request_id="req_3",
        correlation_id="corr_3", tenant_id=client, portfolio_id="direct_001",
        snapshot=None, result=payload, warnings=(), error=err,
        audit=_audit(client, ms=15, outcome="error",
                     error_code=ErrorCode.STORAGE_UNAVAILABLE))


@pytest.fixture()
def telemetry(store, source_registry, monkeypatch):
    """OCC API bound to a store already holding a few recorded questions."""
    monkeypatch.setenv("TRAKT_MI_QUERY_TELEMETRY", "on")
    engine = make_engine(store, source_registry, "happy")
    store.register_client("client_a")
    store.register_client("client_b")

    recorded = {}
    recorded["answered"] = qt.record(
        store, _answered(), question=ANSWER_PAYLOAD["question"])
    recorded["refused"] = qt.record(
        store, _refused(), question="Can you predict next quarter's defaults?")
    recorded["errored"] = qt.record(
        store, _errored(), question="What is the total balance?")
    recorded["other_client"] = qt.record(
        store, _answered(client="client_b", actor="bob@other.example"),
        question="Bob's private question about his own book")

    app_module.set_engine(engine)
    client = TestClient(app_module.app, raise_server_exceptions=False)
    yield {"client": client, "store": store, "recorded": recorded}


def _h(token=OP_ALL):
    return {"X-Operator-Token": token}


# --------------------------------------------------------------------------- #
# A. An answered query is recorded in full
# --------------------------------------------------------------------------- #

class TestAnsweredQuery:
    def test_every_field_the_operator_needs_is_recorded(self, telemetry):
        rec = telemetry["recorded"]["answered"]
        assert rec is not None
        # who asked
        assert rec["client_id"] == "client_a"
        assert rec["user_id"] == "alice@lender.example"
        assert rec["channel"] == "react"
        # what they asked, and what they saw — verbatim
        assert rec["question"] == ANSWER_PAYLOAD["question"]
        assert rec["answer"] == ANSWER_PAYLOAD["answer"]
        # what Trakt understood
        assert rec["interpretation"]["metric"] == "current_loan_to_value"
        assert rec["interpretation"]["aggregation"] == "weighted_avg"
        assert rec["interpretation"]["filters"] == {"account_status": "LIVE"}
        assert rec["interpretation"]["intent"] == "summary"
        # which capability ran
        assert rec["route"] == "portfolio_summary"
        assert rec["capability"] == "mi_query"
        assert rec["parser"]["parser_used"] == "deterministic"
        # which data
        assert rec["snapshot_id"] == "snap_2025_11_30"
        assert rec["reporting_period"] == "2025-11-30"
        # outcome and cost
        assert rec["outcome"] == "ANSWERED"
        assert rec["duration_ms"] == 812
        # and nobody has judged it yet
        assert rec["review"]["classification"] == "UNREVIEWED"
        assert rec["review"]["reviewer"] is None

    def test_the_record_carries_no_model_reasoning_or_secrets(self, telemetry):
        blob = json.dumps(telemetry["recorded"]["answered"]).lower()
        for forbidden in ("prompt", "reasoning", "chain_of_thought", "token",
                          "authorization", "connection_string", "traceback"):
            assert forbidden not in blob

    def test_an_interpretation_is_never_invented(self):
        """A route that exposes no spec records an empty interpretation."""
        result = _answered()
        result.result["spec"] = {}
        rec = qt.build_record(result, question="q")
        assert rec["interpretation"] == {}


# --------------------------------------------------------------------------- #
# B / C. Refusals and errors are distinguishable and safe
# --------------------------------------------------------------------------- #

class TestRefusalAndError:
    def test_a_refusal_records_its_structured_reason_and_no_answer(
            self, telemetry):
        rec = telemetry["recorded"]["refused"]
        assert rec["outcome"] == "REFUSED"
        assert rec["refusal_reason"] == ErrorCode.UNSUPPORTED_QUESTION
        assert rec["error_code"] is None
        assert rec["answer"] == "", "a refusal must not carry an invented answer"
        assert rec["question"] == "Can you predict next quarter's defaults?"

    def test_an_error_is_not_recorded_as_a_refusal(self, telemetry):
        rec = telemetry["recorded"]["errored"]
        assert rec["outcome"] == "ERROR"
        assert rec["error_code"] == ErrorCode.STORAGE_UNAVAILABLE
        assert rec["error_category"] == "infrastructure"
        assert rec["refusal_reason"] is None
        # the identifiers that let it be traced in the system logs
        assert rec["query_id"] and rec["request_id"] and rec["correlation_id"]

    def test_no_stack_trace_reaches_the_operator_record(self, telemetry):
        assert "Traceback" not in json.dumps(telemetry["recorded"]["errored"])

    @pytest.mark.parametrize("code,expected", [
        (ErrorCode.UNSUPPORTED_QUESTION, "REFUSED"),
        (ErrorCode.AMBIGUOUS_QUESTION, "REFUSED"),
        (ErrorCode.NO_MATCHING_RECORDS, "REFUSED"),
        (ErrorCode.PORTFOLIO_NOT_AUTHORISED, "REFUSED"),
        (ErrorCode.CALCULATION_FAILED, "ERROR"),
        (ErrorCode.STORAGE_UNAVAILABLE, "ERROR"),
    ])
    def test_the_outcome_split_uses_the_existing_error_vocabulary(
            self, code, expected):
        assert qt.outcome_for(_refused(code=code)) == expected


# --------------------------------------------------------------------------- #
# D. Human review — calibration evidence, never a change to the answer
# --------------------------------------------------------------------------- #

class TestOperatorReview:
    def test_an_operator_can_classify_a_response(self, telemetry):
        qid = telemetry["recorded"]["answered"]["query_id"]
        r = telemetry["client"].post(
            f"/ops/mi-queries/{qid}/review", headers=_h(),
            json={"classification": "wrong_interpretation",
                  "note": "read 'funded' as the whole book"})
        assert r.status_code == 200
        assert r.json()["review"]["classification"] == "WRONG_INTERPRETATION"
        assert r.json()["review"]["reviewer"] == "Root Operator"
        assert r.json()["review"]["reviewed_at"]

    def test_the_review_never_changes_what_the_user_was_shown(self, telemetry):
        qid = telemetry["recorded"]["answered"]["query_id"]
        before = telemetry["store"].load_mi_query("client_a", qid)
        telemetry["client"].post(
            f"/ops/mi-queries/{qid}/review", headers=_h(),
            json={"classification": "WRONG_CALCULATION", "note": "n"})
        after = telemetry["store"].load_mi_query("client_a", qid)
        assert after["answer"] == before["answer"]
        assert after["question"] == before["question"]
        assert after["interpretation"] == before["interpretation"]
        assert after["outcome"] == before["outcome"] == "ANSWERED"

    def test_an_invented_classification_is_refused(self, telemetry):
        qid = telemetry["recorded"]["answered"]["query_id"]
        r = telemetry["client"].post(
            f"/ops/mi-queries/{qid}/review", headers=_h(),
            json={"classification": "PRETTY_GOOD"})
        assert r.status_code == 400
        assert "OPS_BAD_CLASSIFICATION" in r.text

    def test_the_review_is_audited(self, telemetry):
        qid = telemetry["recorded"]["answered"]["query_id"]
        telemetry["client"].post(f"/ops/mi-queries/{qid}/review", headers=_h(),
                                 json={"classification": "CORRECT"})
        events = [e for e in telemetry["store"].list_audit("client_a")
                  if e.get("event") == "mi_query_reviewed"]
        assert events and events[-1]["detail"]["query_id"] == qid


# --------------------------------------------------------------------------- #
# E. Client isolation
# --------------------------------------------------------------------------- #

class TestClientIsolation:
    def test_one_clients_operator_never_sees_anothers_questions(self, telemetry):
        r = telemetry["client"].get("/ops/mi-queries?window=72h",
                                    headers=_h(OP_A))
        assert r.status_code == 200
        questions = [q["question"] for q in r.json()["queries"]]
        assert any("weighted average LTV" in q for q in questions)
        assert not any("Bob's private question" in q for q in questions)
        assert {q["client_id"] for q in r.json()["queries"]} == {"client_a"}

    def test_the_other_direction_holds_too(self, telemetry):
        r = telemetry["client"].get("/ops/mi-queries?window=72h",
                                    headers=_h(OP_B))
        assert {q["client_id"] for q in r.json()["queries"]} == {"client_b"}

    def test_a_detail_read_across_the_boundary_is_refused(self, telemetry):
        qid = telemetry["recorded"]["other_client"]["query_id"]
        r = telemetry["client"].get(f"/ops/mi-queries/{qid}?client=client_b",
                                    headers=_h(OP_A))
        assert r.status_code in (403, 404)

    def test_the_export_is_scoped_too(self, telemetry):
        r = telemetry["client"].get(
            "/ops/mi-queries/export/calibration?window=72h&reviewed_only=false",
            headers=_h(OP_A))
        assert not any("Bob's private question" in q["question"]
                       for q in r.json()["queries"])


# --------------------------------------------------------------------------- #
# F / G. Data version, and the launch window
# --------------------------------------------------------------------------- #

class TestVersionAndWindow:
    def test_a_query_names_the_exact_data_it_was_answered_from(self, telemetry):
        qid = telemetry["recorded"]["answered"]["query_id"]
        rec = telemetry["client"].get(f"/ops/mi-queries/{qid}",
                                      headers=_h()).json()["query"]
        assert rec["snapshot_id"] == "snap_2025_11_30"
        assert rec["content_hash"] == "sha256:abc123"
        assert rec["source_kind"] == "platform_canonical"
        assert rec["reporting_period"] == "2025-11-30"
        # enough to reproduce the query offline against the same data
        assert rec["question"] and rec["dataset_view"] and rec["portfolio_id"]

    def test_the_window_excludes_older_questions(self, telemetry):
        old = _answered(started=(datetime.now(timezone.utc)
                                 - timedelta(days=9)).isoformat())
        qt.record(telemetry["store"], old, question="an older question")
        recent = telemetry["client"].get("/ops/mi-queries?window=72h",
                                         headers=_h()).json()
        assert not any(q["question"] == "an older question"
                       for q in recent["queries"])
        everything = telemetry["client"].get("/ops/mi-queries?window=all",
                                             headers=_h()).json()
        assert any(q["question"] == "an older question"
                   for q in everything["queries"])

    def test_the_summary_counts_what_the_window_holds(self, telemetry):
        s = telemetry["client"].get("/ops/mi-queries/summary?window=72h",
                                    headers=_h()).json()["summary"]
        assert s["total_questions"] == 4
        assert s["answered"] == 2 and s["refused"] == 1 and s["errors"] == 1
        assert s["unique_users"] == 2
        assert s["median_latency_ms"] is not None

    def test_a_correctness_rate_is_only_over_reviewed_responses(self, telemetry):
        qid = telemetry["recorded"]["answered"]["query_id"]
        telemetry["client"].post(f"/ops/mi-queries/{qid}/review", headers=_h(),
                                 json={"classification": "CORRECT"})
        s = telemetry["client"].get("/ops/mi-queries/summary?window=72h",
                                    headers=_h()).json()["summary"]
        assert s["reviewed"] == 1 and s["unreviewed"] == 3
        assert s["reviewed_correct"] == 1
        # 100% of ONE reviewed response — the denominator travels with the rate.
        assert s["reviewed_correctness_pct"] == 100.0
        assert s["reviewed"] < s["total_questions"]


# --------------------------------------------------------------------------- #
# H. The external-model-safe calibration export
# --------------------------------------------------------------------------- #

class TestCalibrationExport:
    def test_the_safe_export_carries_no_portfolio_content(self, telemetry):
        qid = telemetry["recorded"]["answered"]["query_id"]
        telemetry["client"].post(f"/ops/mi-queries/{qid}/review", headers=_h(),
                                 json={"classification": "WRONG_INTERPRETATION",
                                       "note": "read funded as the whole book"})
        body = telemetry["client"].get(
            "/ops/mi-queries/export/calibration?window=72h",
            headers=_h()).json()
        assert body["export_kind"] == "external_model_safe"
        blob = json.dumps(body)
        # the answer text and every figure in it must be absent
        assert "42.7%" not in blob
        assert "184,220,110" not in blob
        assert "1,204 loans" not in blob
        assert "sha256:abc123" not in blob        # the content hash, too
        for row in body["queries"]:
            assert "answer" not in row
            assert "artifacts" not in row
            assert "content_hash" not in row

    def test_the_safe_export_carries_what_calibration_needs(self, telemetry):
        qid = telemetry["recorded"]["answered"]["query_id"]
        telemetry["client"].post(f"/ops/mi-queries/{qid}/review", headers=_h(),
                                 json={"classification": "WRONG_INTERPRETATION",
                                       "note": "read funded as the whole book"})
        rows = telemetry["client"].get(
            "/ops/mi-queries/export/calibration?window=72h",
            headers=_h()).json()["queries"]
        row = [r for r in rows if r["query_id"] == qid][0]
        assert row["question"] == ANSWER_PAYLOAD["question"]
        assert row["interpretation"]["metric"] == "current_loan_to_value"
        assert row["route"] == "portfolio_summary"
        assert row["outcome"] == "ANSWERED"
        assert row["quality_classification"] == "WRONG_INTERPRETATION"
        assert row["reviewer_note"] == "read funded as the whole book"

    def test_it_exports_reviewed_questions_by_default(self, telemetry):
        body = telemetry["client"].get(
            "/ops/mi-queries/export/calibration?window=72h",
            headers=_h()).json()
        assert body["count"] == 0, "nothing reviewed yet, nothing to calibrate on"

    def test_the_governed_record_still_holds_the_answer_for_review(self,
                                                                  telemetry):
        """The export is narrow; the OCC record an operator reads is not."""
        qid = telemetry["recorded"]["answered"]["query_id"]
        rec = telemetry["client"].get(f"/ops/mi-queries/{qid}",
                                      headers=_h()).json()["query"]
        assert "42.7%" in rec["answer"]


# --------------------------------------------------------------------------- #
# Refusal analysis — repeated unsupported patterns
# --------------------------------------------------------------------------- #

class TestRefusalAnalysis:
    def test_refusals_can_be_filtered_and_counted_by_reason(self, telemetry):
        for q in ("predict defaults next year", "predict defaults next quarter"):
            qt.record(telemetry["store"], _refused(), question=q)
        rows = telemetry["client"].get(
            "/ops/mi-queries?window=72h&outcome=REFUSED",
            headers=_h()).json()["queries"]
        assert len(rows) == 3
        assert {r["refusal_reason"] for r in rows} == {
            ErrorCode.UNSUPPORTED_QUESTION}
        assert all(r["question"] for r in rows)


# --------------------------------------------------------------------------- #
# I. Behaviour parity
# --------------------------------------------------------------------------- #

class TestBehaviourUnchanged:
    def test_telemetry_never_fails_a_query(self, monkeypatch, store):
        """A broken store must not turn an answered question into a failure."""
        monkeypatch.setenv("TRAKT_MI_QUERY_TELEMETRY", "on")

        class Broken:
            def save_mi_query(self, doc):
                raise RuntimeError("storage is down")

        assert qt.record(Broken(), _answered(), question="q") is None

    def test_nothing_is_recorded_without_a_configured_store(self, monkeypatch):
        """Telemetry writes, so it records only where a store was chosen."""
        monkeypatch.delenv("TRAKT_LOCAL_BLOB_ROOT", raising=False)
        monkeypatch.delenv("TRAKT_STORAGE_BACKEND", raising=False)
        monkeypatch.delenv("TRAKT_MI_QUERY_TELEMETRY", raising=False)

        class Recorder:
            saved = []

            def save_mi_query(self, doc):
                self.saved.append(doc)

        r = Recorder()
        qt.record(r, _answered(), question="q")
        assert r.saved == []

    def test_a_query_without_a_tenant_is_not_recorded(self, monkeypatch, store):
        """The store is client-scoped; a record with no client could not be
        isolated, so it is not written at all."""
        monkeypatch.setenv("TRAKT_MI_QUERY_TELEMETRY", "on")
        result = GovernedResult(
            capability="mi_query", status="success", request_id="req_x",
            correlation_id="corr_x", tenant_id="", portfolio_id=None,
            snapshot=SNAPSHOT, result=dict(ANSWER_PAYLOAD), warnings=())
        assert qt.record(store, result, question="q") is None
