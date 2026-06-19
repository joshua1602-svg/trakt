"""Phase 8C — end-to-end NLQ → interpreter → runtime smoke harness.

Proves that a natural-language MI question can be interpreted into a governed
MIQuerySpec v2 and then EXECUTED through the existing deterministic runtime
(``run_mi_query``) over the Phase 6B synthetic snapshot setup.

Strict separation of concerns:
  * the LLM (a fake client here) ONLY returns MIQuerySpec-v2 JSON — it never
    computes analytics;
  * ``run_mi_query`` is the single execution engine;
  * ambiguous / invalid / clarification interpretations are NEVER executed.

No external LLM calls, no API keys, no Azure/Streamlit/legacy-analytics imports,
no Annex 2 changes, no new chart types.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from mi_agent.interpreter import (
    InterpreterContext,
    interpret,
    interpret_and_run_mi_query,
)
from mi_agent.interpreter.runtime_bridge import (
    NOT_EXECUTED_CLARIFICATION,
    NOT_EXECUTED_INVALID_SPEC,
    BridgeResult,
)
from mi_agent.mi_runtime import RuntimeResult
from mi_agent.mi_query_validator import load_mi_semantics
from mi_agent.risk_monitor import load_risk_monitor_config
from snapshot.adapters import LocalFsSnapshotStore
from snapshot.model import SnapshotHeader

REPO_ROOT = Path(__file__).resolve().parents[1]
SEMANTICS_PATH = REPO_ROOT / "mi_agent" / "mi_semantics_field_registry.yaml"

CLIENT = "smoke"
REPORTING_DATES = ["2024-01-31", "2024-02-29", "2024-03-31"]


# --------------------------------------------------------------------------- #
# Synthetic snapshot setup (mirrors Phase 6B)
# --------------------------------------------------------------------------- #


def _loan(lid, status, stage, bal, region, broker, portfolio,
          grade=None, ifrs=None, pd_b=None, prob=None):
    return {
        "loan_identifier": lid,
        "funded_status": status,
        "pipeline_stage": stage,
        "current_outstanding_balance": float(bal),
        "geographic_region_obligor": region,
        "broker_channel": broker,
        "portfolio_id": portfolio,
        "internal_risk_grade": grade,
        "ifrs9_stage": ifrs,
        "pd_bucket": pd_b,
        "forecast_funding_probability": prob,
        "origination_date": "2020-01-15",
    }


def _snapshot_frames():
    s1 = pd.DataFrame([
        _loan("F1", "funded", "completed", 100, "North", "Broker A", "PF_001",
              "A", "Stage 1", "<0.25%"),
        _loan("F2", "funded", "completed", 200, "South", "Broker B", "PF_002",
              "B", "Stage 1", "0.25-0.5%"),
        _loan("P1", "pipeline", "OFFER", 50, "North", "Broker A", "PF_001",
              prob=0.5),
        _loan("P2", "pipeline", "KFI", 40, "South", "Broker B", "PF_002",
              prob=0.25),
    ])
    s2 = pd.DataFrame([
        _loan("F1", "funded", "completed", 100, "North", "Broker A", "PF_001",
              "A", "Stage 1", "<0.25%"),
        _loan("F2", "funded", "completed", 220, "South", "Broker B", "PF_002",
              "B", "Stage 1", "0.25-0.5%"),
        _loan("F3", "funded", "completed", 300, "North", "Broker A", "PF_001",
              "C", "Stage 2", "1-2.5%"),
        _loan("P1", "pipeline", "OFFER", 50, "North", "Broker A", "PF_001",
              prob=0.5),
        _loan("P2", "pipeline", "APPLICATION", 40, "South", "Broker B", "PF_002",
              prob=0.45),
    ])
    s3 = pd.DataFrame([
        _loan("F1", "funded", "completed", 100, "North", "Broker A", "PF_001",
              "A", "Stage 1", "<0.25%"),
        _loan("F2", "funded", "completed", 220, "South", "Broker B", "PF_002",
              "C", "Stage 2", "0.5-1%"),
        _loan("F3", "funded", "completed", 300, "North", "Broker A", "PF_001",
              "C", "Stage 2", "1-2.5%"),
        _loan("P1", "pipeline", "OFFER", 50, "North", "Broker A", "PF_001",
              prob=0.5),
    ])
    return [s1, s2, s3]


@pytest.fixture(scope="module")
def semantics():
    return load_mi_semantics(SEMANTICS_PATH)


@pytest.fixture(scope="module")
def risk_cfg():
    return load_risk_monitor_config()


@pytest.fixture
def store(tmp_path):
    s = LocalFsSnapshotStore(root=tmp_path / "smoke_snaps")
    for i, (rd, frame) in enumerate(zip(REPORTING_DATES, _snapshot_frames())):
        s.register_snapshot(
            SnapshotHeader(client_id=CLIENT, route="mi", reporting_date=rd,
                           source_file_id=f"sha256:smoke{i}", cadence="monthly",
                           upload_timestamp=f"{rd}T09:00:00"),
            frame)
    return s


def _ctx() -> InterpreterContext:
    # Anchor the interpreter to the synthetic client + dates.
    return InterpreterContext(snapshot_client_id=CLIENT, route_id="mi",
                              as_of="2024-03-31", prev_period="2024-02-29",
                              range_start="2024-01-01")


# --------------------------------------------------------------------------- #
# Fake Anthropic clients — return spec JSON ONLY (never analytics)
# --------------------------------------------------------------------------- #


class FakeClient:
    """Returns a fixed canned spec completion; records the prompt."""

    def __init__(self, response):
        self.response = response
        self.prompt = None
        self.calls = 0

    def complete_mi_spec_json(self, prompt):
        self.prompt = prompt
        self.calls += 1
        return self.response


# Canned MIQuerySpec-v2 JSON the "model" returns for each NL question. Dates are
# resolved from context anchors (as the constrained prompt requires).
CANNED_SPECS = {
    "show total funded": {
        "execution_mode": "state", "state": "total_funded",
        "temporal_mode": "latest"},
    "show total pipeline": {
        "execution_mode": "state", "state": "total_pipeline",
        "temporal_mode": "latest"},
    "show forecast funded": {
        "execution_mode": "state", "state": "total_forecast_funded",
        "temporal_mode": "latest"},
    "trend funded balance over the last three months": {
        "execution_mode": "temporal", "state": "total_funded",
        "temporal_mode": "trend", "start_date": "2024-01-01",
        "end_date": "2024-03-31", "trend_grain": "monthly"},
    "compare funded balance to last month": {
        "execution_mode": "temporal", "state": "total_funded",
        "temporal_mode": "compare", "baseline_date": "2024-02-29",
        "current_date": "2024-03-31"},
    "show funded balance by portfolio": {
        "execution_mode": "risk", "risk_monitor_mode": "concentration",
        "state": "total_funded", "dimension": "portfolio_id"},
    "show funded balance by region": {
        "execution_mode": "risk", "risk_monitor_mode": "concentration",
        "state": "total_funded", "dimension": "geographic_region_obligor"},
    "show pipeline by stage": {
        "execution_mode": "risk", "risk_monitor_mode": "concentration",
        "state": "total_pipeline", "dimension": "pipeline_stage",
        "as_of_date": "2024-02-29"},
    "show concentration by region": {
        "execution_mode": "risk", "risk_monitor_mode": "concentration",
        "state": "total_funded", "dimension": "geographic_region_obligor"},
    "show risk grade migration": {
        "execution_mode": "risk", "risk_monitor_mode": "migration",
        "dimension": "internal_risk_grade", "baseline_date": "2024-02-29",
        "current_date": "2024-03-31"},
    "show IFRS stage migration": {
        "execution_mode": "risk", "risk_monitor_mode": "migration",
        "dimension": "ifrs9_stage", "baseline_date": "2024-02-29",
        "current_date": "2024-03-31"},
    "show PD bucket migration": {
        "execution_mode": "risk", "risk_monitor_mode": "migration",
        "dimension": "pd_bucket", "baseline_date": "2024-02-29",
        "current_date": "2024-03-31"},
}


def _client_for(question):
    return FakeClient(json.dumps(CANNED_SPECS[question]))


def _run(question, store, semantics, risk_cfg, **kw):
    return interpret_and_run_mi_query(
        question, _ctx(), _client_for(question), store,
        semantics=semantics, risk_config=risk_cfg, **kw)


# --------------------------------------------------------------------------- #
# 1. End-to-end execution for each supported NL question
# --------------------------------------------------------------------------- #


def test_show_total_funded(store, semantics, risk_cfg):
    r = _run("show total funded", store, semantics, risk_cfg)
    assert r.executed and r.ok
    assert r.runtime_result.mode == "state"
    assert r.runtime_result.row_count == 3
    assert float(r.data["current_outstanding_balance"].sum()) == 620.0
    # The LLM produced a spec, not analytics.
    assert isinstance(r.runtime_result, RuntimeResult)


def test_show_total_pipeline(store, semantics, risk_cfg):
    r = _run("show total pipeline", store, semantics, risk_cfg)
    assert r.executed and r.ok
    assert r.runtime_result.row_count == 1
    assert float(r.data["current_outstanding_balance"].sum()) == 50.0


def test_show_forecast_funded(store, semantics, risk_cfg):
    r = _run("show forecast funded", store, semantics, risk_cfg)
    assert r.executed and r.ok
    assert r.runtime_result.metadata["forecast_funded_total"] == pytest.approx(645.0)


def test_trend_funded(store, semantics, risk_cfg):
    r = _run("trend funded balance over the last three months", store, semantics,
             risk_cfg)
    assert r.executed and r.ok
    assert r.runtime_result.mode == "temporal"
    assert list(r.data["balance"]) == [300.0, 620.0, 620.0]
    assert r.chart_instruction == {"chart_type": "line"}


def test_compare_funded(store, semantics, risk_cfg):
    r = _run("compare funded balance to last month", store, semantics, risk_cfg)
    assert r.executed and r.ok
    row = r.data.iloc[0]
    # feb29 funded = mar31 funded = 620 (same loans, balances steady).
    assert row["baseline_balance"] == 620.0 and row["current_balance"] == 620.0
    assert row["balance_change"] == 0.0


def test_funded_by_portfolio(store, semantics, risk_cfg):
    r = _run("show funded balance by portfolio", store, semantics, risk_cfg)
    assert r.executed and r.ok
    by = r.data.set_index("portfolio_id")
    assert by.loc["PF_001", "balance_sum"] == 400.0
    assert by.loc["PF_002", "balance_sum"] == 220.0


def test_funded_by_region(store, semantics, risk_cfg):
    r = _run("show funded balance by region", store, semantics, risk_cfg)
    assert r.executed and r.ok
    by = r.data.set_index("geographic_region_obligor")
    assert by.loc["North", "balance_sum"] == 400.0
    assert by.loc["South", "balance_sum"] == 220.0


def test_pipeline_by_stage(store, semantics, risk_cfg):
    r = _run("show pipeline by stage", store, semantics, risk_cfg)
    assert r.executed and r.ok
    by = r.data.set_index("pipeline_stage")
    assert set(by.index) == {"OFFER", "APPLICATION"}
    assert by.loc["OFFER", "balance_sum"] == 50.0
    assert by.loc["APPLICATION", "balance_sum"] == 40.0


def test_concentration_by_region(store, semantics, risk_cfg):
    r = _run("show concentration by region", store, semantics, risk_cfg)
    assert r.executed and r.ok
    by = r.data.set_index("geographic_region_obligor")
    assert by.loc["North", "balance_sum"] == 400.0
    assert "status" in r.data.columns


def test_risk_grade_migration(store, semantics, risk_cfg):
    r = _run("show risk grade migration", store, semantics, risk_cfg)
    assert r.executed and r.ok
    assert r.runtime_result.mode == "risk"
    deter = r.data[(r.data["from_value"] == "B") & (r.data["to_value"] == "C")]
    assert not deter.empty and deter.iloc[0]["movement_type"] == "deteriorated"


def test_ifrs_migration(store, semantics, risk_cfg):
    r = _run("show IFRS stage migration", store, semantics, risk_cfg)
    assert r.executed and r.ok
    deter = r.data[(r.data["from_value"] == "Stage 1")
                   & (r.data["to_value"] == "Stage 2")]
    assert not deter.empty and deter.iloc[0]["movement_type"] == "deteriorated"


def test_pd_bucket_migration(store, semantics, risk_cfg):
    r = _run("show PD bucket migration", store, semantics, risk_cfg)
    assert r.executed and r.ok
    assert "deteriorated" in set(r.data["movement_type"])


# --------------------------------------------------------------------------- #
# 2. Clarification / invalid execution guards — must NOT execute
# --------------------------------------------------------------------------- #


def _guard(raw, store, semantics, risk_cfg, *, semantics_for_interp=None):
    client = FakeClient(raw)
    return interpret_and_run_mi_query(
        "some question", _ctx(), client, store, semantics=semantics,
        risk_config=risk_cfg)


def test_clarification_object_not_executed(store, semantics, risk_cfg):
    raw = json.dumps({"clarification_required": True,
                      "clarification_question": "Which risk view?"})
    r = _guard(raw, store, semantics, risk_cfg)
    assert not r.executed and r.runtime_result is None
    assert r.clarification_required
    assert NOT_EXECUTED_CLARIFICATION in r.issue_codes()


def test_ambiguous_question_via_deterministic_not_executed(store, semantics,
                                                           risk_cfg):
    # Use the deterministic interpreter directly; "show risk" clarifies.
    r = interpret_and_run_mi_query("show risk", _ctx(), interpret, store,
                                   semantics=semantics, risk_config=risk_cfg)
    assert not r.executed and r.runtime_result is None
    assert r.clarification_required


def test_malformed_output_not_executed(store, semantics, risk_cfg):
    r = _guard("not valid json at all {", store, semantics, risk_cfg)
    assert not r.executed and r.runtime_result is None


def test_invalid_enum_not_executed(store, semantics, risk_cfg):
    raw = json.dumps({"execution_mode": "state", "state": "not_a_state",
                      "temporal_mode": "latest"})
    r = _guard(raw, store, semantics, risk_cfg)
    assert not r.executed and r.runtime_result is None
    assert "invalid_enum_value" in r.issue_codes()


def test_hallucinated_field_not_executed(store, semantics, risk_cfg):
    raw = json.dumps({"execution_mode": "risk", "state": "total_funded",
                      "risk_monitor_mode": "concentration",
                      "dimension": "totally_made_up_field"})
    client = FakeClient(raw)
    r = interpret_and_run_mi_query("show funded by nonsense", _ctx(), client,
                                   store, semantics=semantics,
                                   risk_config=risk_cfg)
    assert not r.executed and r.runtime_result is None
    assert "llm_hallucinated_field" in r.issue_codes()


def test_missing_temporal_dates_not_executed(store, semantics, risk_cfg):
    raw = json.dumps({"execution_mode": "temporal", "state": "total_funded",
                      "temporal_mode": "compare"})
    r = _guard(raw, store, semantics, risk_cfg)
    assert not r.executed and r.runtime_result is None
    assert "temporal_selector_incomplete" in r.issue_codes()


def test_regulatory_route_with_mi_state_not_executed(store, semantics, risk_cfg):
    raw = json.dumps({"route_id": "regulatory_annex2", "state": "total_funded",
                      "temporal_mode": "latest"})
    r = _guard(raw, store, semantics, risk_cfg)
    assert not r.executed and r.runtime_result is None
    assert "invalid_route_for_state" in r.issue_codes()


def test_mna_route_with_pipeline_state_not_executed(store, semantics, risk_cfg):
    raw = json.dumps({"route_id": "mna", "state": "total_pipeline",
                      "temporal_mode": "latest"})
    r = _guard(raw, store, semantics, risk_cfg)
    assert not r.executed and r.runtime_result is None
    assert "invalid_route_for_state" in r.issue_codes()


# --------------------------------------------------------------------------- #
# 3. Bridge contract / separation of concerns
# --------------------------------------------------------------------------- #


def test_bridge_result_shape(store, semantics, risk_cfg):
    r = _run("show total funded", store, semantics, risk_cfg)
    assert isinstance(r, BridgeResult)
    assert r.raw_question == "show total funded"
    assert r.interpretation is not None
    assert r.normalized_spec is not None
    assert r.runtime_result is not None
    assert r.executed is True


def test_llm_only_produces_spec_not_analytics(store, semantics, risk_cfg):
    client = _client_for("show total funded")
    interpret_and_run_mi_query("show total funded", _ctx(), client, store,
                               semantics=semantics, risk_config=risk_cfg)
    # The fake "model" response is a spec object — never a result/dataframe.
    payload = json.loads(client.response)
    assert "current_outstanding_balance" not in payload  # no computed columns
    assert set(payload).issubset({
        "execution_mode", "state", "temporal_mode", "risk_monitor_mode",
        "dimension", "baseline_date", "current_date", "start_date", "end_date",
        "trend_grain", "as_of_date", "route_id", "snapshot_client_id"})


def test_invalid_spec_reports_not_executed_code(store, semantics, risk_cfg):
    # Gate 2: a (non-clarifying) interpreter that nonetheless yields an invalid
    # spec must not be executed. Build such a result directly.
    from mi_agent.interpreter.models import InterpretationResult
    from mi_agent.mi_query_spec import MIQuerySpec
    from mi_agent.mi_spec_validation import validate_query_spec

    spec = MIQuerySpec.from_dict({"route_id": "mi", "state": "total_funded",
                                  "temporal_mode": "compare",
                                  "snapshot_client_id": CLIENT}).normalized()
    vr = validate_query_spec(spec)
    assert not vr.ok  # compare without dates

    def _bad_interpreter(question, context):
        return InterpretationResult(
            raw_question=question, candidate_spec={}, normalized_spec=spec,
            validation_result=vr, issues=list(vr.issues),
            clarification_required=False)

    r = interpret_and_run_mi_query("compare funded", _ctx(), _bad_interpreter,
                                   store, semantics=semantics,
                                   risk_config=risk_cfg)
    assert not r.executed and r.runtime_result is None
    assert NOT_EXECUTED_INVALID_SPEC in r.issue_codes()


# --------------------------------------------------------------------------- #
# 4. Guards — no forbidden imports, no external LLM/SDK
# --------------------------------------------------------------------------- #


def test_no_forbidden_imports_in_bridge():
    text = (REPO_ROOT / "mi_agent" / "interpreter" / "runtime_bridge.py"
            ).read_text(encoding="utf-8")
    for token in ("import anthropic", "import openai", "import streamlit",
                  "import azure", "from azure", "from analytics ",
                  "from analytics."):
        assert token not in text
