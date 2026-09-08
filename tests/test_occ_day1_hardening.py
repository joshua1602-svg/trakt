"""Day-1 go-live hardening: the Gate 3 boundary, and recovery after a restart.

Three properties, all executed on the real production route (raw file at the
governed blob prefix -> Event Grid intake -> OCC -> the gates), because all
three are about what an operator is offered and what the platform then does,
which a stubbed adapter cannot show.

A. A structurally incomplete canonical cannot be waived. A core field that is
   absent or entirely blank is not an exception with a rationale; it is a report
   that cannot be produced. No acceptance is offered, and the engine refuses one
   even if a decision reaches it from an earlier round.

B. A business-rule finding — a value that is present and correctly typed but
   fails a deterministic rule — may still be accepted for ONE delivery, with a
   mandatory justification, the finding named on the record, and no carry-over
   into the next period.

C. A run interrupted by a worker recycle, which takes the ephemeral staging
   directory with it, can be restarted by the operator from the durable source
   and produces the same governed output as an uninterrupted run.
"""

from __future__ import annotations

import shutil
import time

import pandas as pd
import pytest

from operations_control.engine import OpsEngine, OpsError

from .occ_go_live_harness import (
    ALPHA_MAPPING,
    LEI_A,
    alpha_tape,
    answer_mapping_queue,
    arrive,
    blocking_validation_fields,
    bootstrap,
    central_tape,
    gate_steps,
    onboard,
    settle,
    wait_for_rerun,
)

pytestmark = pytest.mark.slow

STATIC = {"portfolio_id": "direct_001", "source_portfolio_id": "direct_001",
          "source_portfolio_type": "direct"}
EXCEPTION_ARTEFACT = "validation_halt"


def _queue(env, client, wf):
    return [d for d in env["store"].open_decisions(client, wf)
            if d["kind"] != "publication"]


def _exceptions(env, client, wf):
    return [d for d in _queue(env, client, wf)
            if (d.get("subject") or {}).get("artefact") == EXCEPTION_ARTEFACT]


def _mapping_only(env, client, wf, mapping):
    """Answer the mapping queue, never the exception."""
    if [d for d in _queue(env, client, wf)
            if (d.get("subject") or {}).get("artefact") != EXCEPTION_ARTEFACT]:
        answer_mapping_queue(env, client, wf, mapping=mapping, static=STATIC)
        return True
    return False


def _deliver(env, client, period, tape, mapping=None, rows=30):
    r = arrive(env, client_id=client, portfolio_id="direct_001",
               period=period, tape=tape)
    wf = r["workflow_id"]
    run = settle(env, client, wf, timeout=600)
    for d in [x for x in _queue(env, client, wf)
              if "loankey" in x["decision_id"]]:
        env["engine"].resolve_decision(
            client_id=client, decision_id=d["decision_id"], action="approve",
            actor="Operator", value="ACCT_REF", scope="portfolio",
            actor_is_admin=True)
    if _mapping_only(env, client, wf, mapping or ALPHA_MAPPING):
        run = wait_for_rerun(env, client, wf, run.rerun_count, timeout=600)
    return wf, run


# --------------------------------------------------------------------------- #
# A. Structural failures are not waivable
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def missing_core(tmp_path_factory):
    """A tape with no balance column at all — CORE001."""
    mp = pytest.MonkeyPatch()
    tmp = tmp_path_factory.mktemp("nocore")
    env = bootstrap(tmp, mp)
    try:
        onboard(env, client_id="NOCORE", name="No Core Lending",
                portfolio_id="direct_001", lei=LEI_A, products=("mi_reporting",))
        tape = alpha_tape(tmp / "t.csv", "2025-11-30")
        df = pd.read_csv(tape).drop(columns=["BAL_OS"])
        df.to_csv(tape, index=False)
        mapping = {k: v for k, v in ALPHA_MAPPING.items() if v != "BAL_OS"}
        wf, run = _deliver(env, "NOCORE", "2025-11-30", tape, mapping)
        yield {"env": env, "wf": wf, "run": run}
    finally:
        mp.undo()


@pytest.fixture(scope="module")
def blank_core(tmp_path_factory):
    """A tape whose status column is present but empty on every row — CORE002."""
    mp = pytest.MonkeyPatch()
    tmp = tmp_path_factory.mktemp("blankcore")
    env = bootstrap(tmp, mp)
    try:
        onboard(env, client_id="BLANK", name="Blank Core Lending",
                portfolio_id="direct_001", lei=LEI_A, products=("mi_reporting",))
        tape = alpha_tape(tmp / "t.csv", "2025-11-30")
        df = pd.read_csv(tape)
        df["STATUS_CD"] = ""
        df.to_csv(tape, index=False)
        wf, run = _deliver(env, "BLANK", "2025-11-30", tape)
        yield {"env": env, "wf": wf, "run": run}
    finally:
        mp.undo()


class TestStructuralFailureCannotBeWaived:
    @pytest.mark.parametrize("case,field", [("missing_core",
                                             "current_principal_balance"),
                                            ("blank_core", "account_status")])
    def test_gate_three_blocks_and_names_the_field(self, case, field, request):
        ctx = request.getfixturevalue(case)
        run = ctx["run"]
        assert run.stage_status("validation") == "blocked"
        assert field in blocking_validation_fields(run)
        gar = ctx["env"]["store"].load_result(run.client_id, ctx["wf"],
                                              "validation")
        # The operator is told what is missing, not "some checks did not pass".
        assert gar.blockers and any(field.replace("_", " ") in b
                                    for b in gar.blockers)
        assert any(e.get("label") == "What is missing" for e in gar.evidence)

    @pytest.mark.parametrize("case", ["missing_core", "blank_core"])
    def test_no_acceptance_is_offered(self, case, request):
        ctx = request.getfixturevalue(case)
        assert _exceptions(ctx["env"], ctx["run"].client_id, ctx["wf"]) == []
        gar = ctx["env"]["store"].load_result(ctx["run"].client_id, ctx["wf"],
                                              "validation")
        assert gar.decisions_required == []

    @pytest.mark.parametrize("case", ["missing_core", "blank_core"])
    def test_the_engine_refuses_an_exception_that_reaches_it_anyway(
            self, case, request):
        """Belt and braces: a decision raised in an earlier round, before the
        data became structurally incomplete, must not be usable now."""
        ctx = request.getfixturevalue(case)
        env, run, wf = ctx["env"], ctx["run"], ctx["wf"]
        doc = {"decision_id": f"{wf}_validation_exception",
               "client_id": run.client_id, "workflow_id": wf,
               "kind": "validation_exception", "status": "open",
               "stage": "validation", "title": "Review the flagged checks",
               "question": "Accept?", "blocking": True,
               "options": [{"value": "proceed", "label": "Continue"},
                           {"value": "stop", "label": "Stop"}],
               "subject": {"artefact": EXCEPTION_ARTEFACT, "findings": []}}
        env["store"].save_decision(run.client_id, doc)
        with pytest.raises(OpsError) as err:
            env["engine"].resolve_decision(
                client_id=run.client_id, decision_id=doc["decision_id"],
                action="approve", actor="Operator", value="proceed",
                reason="I accept this", scope="file", actor_is_admin=True)
        assert "OPS_STRUCTURAL_FAILURE_NOT_WAIVABLE" in str(err.value)

    @pytest.mark.parametrize("case", ["missing_core", "blank_core"])
    def test_nothing_publishes(self, case, request):
        ctx = request.getfixturevalue(case)
        env, run, wf = ctx["env"], ctx["run"], ctx["wf"]
        assert run.stage_status("publication") != "ready"
        assert run.stage_status("assembly") != "completed"
        with pytest.raises(OpsError):
            env["engine"].approve_publication(client_id=run.client_id,
                                              workflow_id=wf, actor="Operator")
        from apps.blob_trigger_app.layout import Layout
        assert not env["storage"].exists(
            Layout.from_env().platform_latest_uri(run.client_id))


# --------------------------------------------------------------------------- #
# B. A business-rule finding may be accepted, once, with a reason
# --------------------------------------------------------------------------- #

def _negative_valuation_tape(path, period, rows=30):
    """Every core field present and populated; two valuations below zero."""
    tape = alpha_tape(path, period, rows=rows)
    df = pd.read_csv(tape)
    df.loc[3, "PROP_VAL_CURR"] = -1
    df.loc[7, "PROP_VAL_CURR"] = -25000
    df.to_csv(tape, index=False)
    return tape


@pytest.fixture(scope="module")
def business(tmp_path_factory):
    mp = pytest.MonkeyPatch()
    tmp = tmp_path_factory.mktemp("biz")
    env = bootstrap(tmp, mp)
    out = {"env": env}
    try:
        onboard(env, client_id="BIZ", name="Biz Lending",
                portfolio_id="direct_001", lei=LEI_A, products=("mi_reporting",))
        t1 = _negative_valuation_tape(tmp / "p1.csv", "2025-11-30")
        wf1, run1 = _deliver(env, "BIZ", "2025-11-30", t1)
        out["wf1"], out["held"] = wf1, run1
        out["offered"] = _exceptions(env, "BIZ", wf1)
        out["gar_held"] = env["store"].load_result("BIZ", wf1, "validation")

        did = out["offered"][0]["decision_id"]
        out["decision_id"] = did
        try:
            env["engine"].resolve_decision(
                client_id="BIZ", decision_id=did, action="approve",
                actor="Operator", value="proceed", reason="   ",
                scope="file", actor_is_admin=True)
            out["blank_reason"] = None
        except OpsError as e:
            out["blank_reason"] = str(e)
        out["after_blank"] = env["store"].load_decision("BIZ", did)["status"]

        env["engine"].resolve_decision(
            client_id="BIZ", decision_id=did, action="approve",
            actor="Operator", value="proceed",
            reason="valuations restated by the lender; corrected file follows "
                   "next month",
            scope="file", actor_is_admin=True)
        out["accepted"] = wait_for_rerun(env, "BIZ", wf1,
                                         run1.rerun_count, timeout=600)
        out["gar_accepted"] = env["store"].load_result("BIZ", wf1, "validation")
        out["decision"] = env["store"].load_decision("BIZ", did)
        out["rules"] = [r for r in env["engine"].rules.list_current("BIZ")
                        if r.kind == "validation_exception"]
        env["engine"].approve_publication(client_id="BIZ", workflow_id=wf1,
                                          actor="Operator")

        t2 = _negative_valuation_tape(tmp / "p2.csv", "2025-12-31", rows=40)
        wf2, run2 = _deliver(env, "BIZ", "2025-12-31", t2)
        out["wf2"], out["next_period"] = wf2, run2
        out["reasked"] = _exceptions(env, "BIZ", wf2)
        yield out
    finally:
        mp.undo()


class TestBusinessRuleException:
    def test_the_finding_is_named_before_it_can_be_accepted(self, business):
        assert len(business["offered"]) == 1
        d = business["offered"][0]
        table = [e for e in d["evidence"] if e.get("label") == "What did not pass"]
        assert table, "the operator was asked to accept findings it did not show"
        rows = table[0]["data"]
        assert any(r["Field"] == "current_valuation_amount" for r in rows)
        assert any(r["Check"] == "BR-CURR-VAL-NONNEG" for r in rows)
        assert {o["value"] for o in d["options"]} == {"proceed", "stop"}

    def test_a_blank_justification_is_refused(self, business):
        assert "OPS_REASON_REQUIRED" in (business["blank_reason"] or "")
        assert business["after_blank"] == "open", \
            "a refused approval must leave the question open"

    def test_an_answer_outside_the_offered_options_is_refused(self, business):
        env = business["env"]
        doc = dict(env["store"].load_decision("BIZ", business["decision_id"]))
        doc["decision_id"] = "probe_not_permitted"
        doc["status"] = "open"
        env["store"].save_decision("BIZ", doc)
        with pytest.raises(OpsError) as err:
            env["engine"].resolve_decision(
                client_id="BIZ", decision_id="probe_not_permitted",
                action="approve", actor="Operator", value="mark_not_applicable",
                reason="x", scope="file", actor_is_admin=True)
        assert "OPS_VALUE_NOT_PERMITTED" in str(err.value)

    def test_the_accepted_run_publishes(self, business):
        assert business["accepted"].status == "awaiting_publication"
        assert gate_steps(business["accepted"])["validate"] == "done"
        assert central_tape(business["accepted"]) is not None

    def test_the_stage_never_claims_a_clean_pass(self, business):
        gar = business["gar_accepted"]
        assert gar.status == "approved", \
            "an accepted exception is not a completed check"
        assert "All checks passed" not in gar.summary
        assert "accepted" in gar.summary.lower()
        # The finding, the operator and the reason all survive on the record.
        joined = " ".join(gar.warnings)
        assert "BR-CURR-VAL-NONNEG" in joined
        assert "Operator" in joined
        assert "restated by the lender" in joined

    def test_the_decision_record_is_attributable(self, business):
        d = business["decision"]
        assert d["status"] == "approved"
        assert d["resolved_by"] == "Operator"
        assert d["resolved_at"]
        assert "restated by the lender" in (d["resolution_reason"] or "")

    def test_the_exception_is_file_scoped_and_names_what_it_accepted(
            self, business):
        rules = business["rules"]
        assert len(rules) == 1
        r = rules[0]
        assert r.scope == "file"          # never portfolio, client or global
        accepted = r.payload.get("accepted_findings") or []
        assert any(a["rule"] == "BR-CURR-VAL-NONNEG" for a in accepted)
        assert r.payload.get("justification")
        assert r.payload.get("reporting_period") == "2025-11-30"

    def test_the_next_period_asks_again(self, business):
        """Definition of contained: last month's judgement is not this month's."""
        assert len(business["reasked"]) == 1
        assert business["next_period"].status != "awaiting_publication"
        assert business["next_period"].stage_status("publication") != "ready"

    def test_the_next_period_cannot_publish_on_last_months_acceptance(
            self, business):
        with pytest.raises(OpsError):
            business["env"]["engine"].approve_publication(
                client_id="BIZ", workflow_id=business["wf2"], actor="Operator")


# --------------------------------------------------------------------------- #
# C. Restart after a worker recycle that took the staging directory with it
# --------------------------------------------------------------------------- #

class TestRestartAfterInterruption:
    def test_the_operator_can_restart_and_gets_the_same_output(
            self, tmp_path, monkeypatch):
        env = bootstrap(tmp_path, monkeypatch)
        onboard(env, client_id="INT", name="Int Lending",
                portfolio_id="direct_001", lei=LEI_A, products=("mi_reporting",))
        tape = alpha_tape(tmp_path / "t.csv", "2025-11-30")
        r = arrive(env, client_id="INT", portfolio_id="direct_001",
                   period="2025-11-30", tape=tape)
        wf = r["workflow_id"]

        run = None
        for _ in range(1500):
            run = env["store"].load_workflow("INT", wf)
            if run and run.status == "running":
                break
            time.sleep(0.02)
        assert run.status == "running", "the run never started"

        # The worker recycles: the scratch directory goes with it. The raw file
        # and every governed record survive.
        shutil.rmtree(tmp_path / "staging", ignore_errors=True)
        assert (tmp_path / "blob" / "raw-v2" / "INT" / "direct" / "funded"
                / "monthly" / "direct_001" / "2025-11-30"
                / "LoanExtract.csv").exists()

        # A restarted OCC API: a fresh engine over the same durable store.
        engine = OpsEngine(env["store"])
        env["engine"] = engine
        assert wf in engine.recover_on_startup()
        run = env["store"].load_workflow("INT", wf)
        assert run.status == "blocked" and run.interrupted
        assert run.blockers and "Run again" in run.blockers[0]

        engine.rerun(run, actor="Operator")
        for _ in range(3000):
            run = env["store"].load_workflow("INT", wf)
            if run and run.status not in ("running", "queued", "received"):
                break
            time.sleep(0.1)
        # The restart re-staged the delivery from its governed location, so the
        # run got as far as an uninterrupted first contact does.
        assert gate_steps(run)["onboard"] == "done"

        for d in [x for x in _queue(env, "INT", wf)
                  if "loankey" in x["decision_id"]]:
            engine.resolve_decision(
                client_id="INT", decision_id=d["decision_id"], action="approve",
                actor="Operator", value="ACCT_REF", scope="portfolio",
                actor_is_admin=True)
        _mapping_only(env, "INT", wf, ALPHA_MAPPING)
        run = wait_for_rerun(env, "INT", wf, run.rerun_count, timeout=600)

        assert run.status == "awaiting_publication"
        assert [gate_steps(run)[k] for k in
                ("onboard", "transform", "validate", "stamp")] == ["done"] * 4
        tape_out = central_tape(run)
        assert tape_out is not None and len(tape_out) == 30
        assert sorted(tape_out["loan_identifier"].astype(str))[0].startswith(
            ("100000", "ACC100000"))

        pub = engine.approve_publication(client_id="INT", workflow_id=wf,
                                         actor="Operator")
        assert pub["status"] == "published" and pub["version"] == 1
        # One delivery, one publication: the restart did not duplicate it.
        assert len(env["store"].list_publications("INT")) == 1

    def test_a_delivery_whose_source_is_gone_says_so(self, tmp_path,
                                                     monkeypatch):
        """The one case a restart cannot fix is stated plainly rather than
        failing with an internal error."""
        env = bootstrap(tmp_path, monkeypatch)
        onboard(env, client_id="GONE", name="Gone Lending",
                portfolio_id="direct_001", lei=LEI_A, products=("mi_reporting",))
        tape = alpha_tape(tmp_path / "t.csv", "2025-11-30")
        r = arrive(env, client_id="GONE", portfolio_id="direct_001",
                   period="2025-11-30", tape=tape)
        wf = r["workflow_id"]
        settle(env, "GONE", wf, timeout=600)
        # Both the scratch copy and the governed source are gone.
        shutil.rmtree(tmp_path / "staging", ignore_errors=True)
        shutil.rmtree(tmp_path / "blob" / "raw-v2", ignore_errors=True)
        engine = OpsEngine(env["store"])
        run = env["store"].load_workflow("GONE", wf)
        run.status = "blocked"
        env["store"].save_workflow(run)
        engine.rerun(run, actor="Operator")
        for _ in range(1200):
            run = env["store"].load_workflow("GONE", wf)
            if run and run.status not in ("running", "queued", "received"):
                break
            time.sleep(0.1)
        assert run.status == "blocked"
        assert any("no longer available" in b for b in run.blockers), \
            run.blockers
