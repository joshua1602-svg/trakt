"""Phase 8F — prompt/interpreter hardening (reduce unnecessary clarification).

Live Anthropic over-clarified four controlled MI questions (total pipeline,
funded balance by region, concentration by region, IFRS stage migration). The
deterministic interpreter already handled them; the gap was the constrained
prompt. These tests prove, WITHOUT any live Anthropic call (fake clients only):

  * the hardened prompt now encodes the pipeline / region / migration-date
    defaults;
  * the canonical specs a well-behaved model should now emit validate and execute
    through the governed bridge;
  * genuinely ambiguous questions still clarify and never execute;
  * malformed / invalid output still never executes.

No MI calculation logic changed; validation is not weakened.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from mi_agent.interpreter import (
    InterpreterContext,
    interpret,
    interpret_and_run_mi_query,
)
from mi_agent.interpreter.prompt import build_mi_spec_prompt
from mi_agent.mi_query_validator import load_mi_semantics
from mi_agent.risk_monitor import load_risk_monitor_config
from pathlib import Path
from snapshot.adapters import LocalFsSnapshotStore
from snapshot.model import SnapshotHeader

REPO_ROOT = Path(__file__).resolve().parents[1]
SEMANTICS_PATH = REPO_ROOT / "mi_agent" / "mi_semantics_field_registry.yaml"
CLIENT = "p8f"
REPORTING_DATES = ["2024-01-31", "2024-02-29", "2024-03-31"]


# --------------------------------------------------------------------------- #
# Synthetic snapshots (mirror Phase 6B/8C)
# --------------------------------------------------------------------------- #


def _loan(lid, status, stage, bal, region, broker, portfolio,
          grade=None, ifrs=None, pd_b=None, prob=None):
    return {
        "loan_identifier": lid, "funded_status": status, "pipeline_stage": stage,
        "current_outstanding_balance": float(bal),
        "geographic_region_obligor": region, "broker_channel": broker,
        "portfolio_id": portfolio, "internal_risk_grade": grade,
        "ifrs9_stage": ifrs, "pd_bucket": pd_b,
        "forecast_funding_probability": prob, "origination_date": "2020-01-15",
    }


def _frames():
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
    s = LocalFsSnapshotStore(root=tmp_path / "snaps")
    for i, (rd, frame) in enumerate(zip(REPORTING_DATES, _frames())):
        s.register_snapshot(
            SnapshotHeader(client_id=CLIENT, route="mi", reporting_date=rd,
                           source_file_id=f"sha256:p8f{i}", cadence="monthly",
                           upload_timestamp=f"{rd}T09:00:00"),
            frame)
    return s


def _ctx():
    return InterpreterContext(snapshot_client_id=CLIENT, route_id="mi",
                              as_of="2024-03-31", prev_period="2024-02-29",
                              range_start="2024-01-01")


class FakeClient:
    def __init__(self, response):
        self.response = response

    def complete_mi_spec_json(self, prompt):
        return self.response


# --------------------------------------------------------------------------- #
# 1. Hardened prompt encodes the new defaults
# --------------------------------------------------------------------------- #


def test_prompt_has_default_resolution_section():
    p = build_mi_spec_prompt("show total pipeline", _ctx())
    assert "Default resolution rules" in p


def test_prompt_pipeline_default():
    p = build_mi_spec_prompt("show total pipeline", _ctx())
    assert "total_pipeline" in p
    assert "State defaulting" in p


def test_prompt_region_default_no_collateral_obligor_clarification():
    p = build_mi_spec_prompt("show funded balance by region", _ctx())
    assert "geographic_region_obligor" in p
    assert "Region defaulting" in p
    assert "Do NOT ask collateral-vs-obligor" in p


def test_prompt_migration_date_default():
    p = build_mi_spec_prompt("show IFRS stage migration", _ctx())
    assert "baseline_date=prev_period" in p or "baseline_date = context.prev_period" in p


# --------------------------------------------------------------------------- #
# 2. The canonical specs the model should now emit validate AND execute
#    (fake client returns the design-rule JSON; no live Anthropic)
# --------------------------------------------------------------------------- #


def test_total_pipeline_design_spec_executes(store, semantics, risk_cfg):
    raw = json.dumps({
        "route_id": "mi", "execution_mode": "state", "state": "total_pipeline",
        "aggregation": "balance_sum", "as_of_date": "2024-03-31",
        "snapshot_client_id": CLIENT, "output_type": "table"})
    r = interpret_and_run_mi_query("show total pipeline", _ctx(),
                                   FakeClient(raw), store, semantics=semantics,
                                   risk_config=risk_cfg)
    assert r.executed and r.runtime_result.ok
    assert r.runtime_result.mode == "state"
    assert float(r.data["current_outstanding_balance"].sum()) == 50.0


def test_funded_by_region_design_spec_executes(store, semantics, risk_cfg):
    raw = json.dumps({
        "route_id": "mi", "execution_mode": "risk",
        "risk_monitor_mode": "concentration", "state": "total_funded",
        "dimension": "geographic_region_obligor", "output_type": "table",
        "snapshot_client_id": CLIENT})
    r = interpret_and_run_mi_query("show funded balance by region", _ctx(),
                                   FakeClient(raw), store, semantics=semantics,
                                   risk_config=risk_cfg)
    assert r.executed and r.runtime_result.ok
    by = r.data.set_index("geographic_region_obligor")
    assert by.loc["North", "balance_sum"] == 400.0
    assert by.loc["South", "balance_sum"] == 220.0


def test_concentration_by_region_design_spec_executes(store, semantics, risk_cfg):
    raw = json.dumps({
        "route_id": "mi", "execution_mode": "risk",
        "risk_monitor_mode": "concentration", "state": "total_funded",
        "dimension": "geographic_region_obligor", "output_type": "table",
        "snapshot_client_id": CLIENT})
    r = interpret_and_run_mi_query("show concentration by region", _ctx(),
                                   FakeClient(raw), store, semantics=semantics,
                                   risk_config=risk_cfg)
    assert r.executed and r.runtime_result.ok
    assert "status" in r.data.columns


def test_ifrs_migration_design_spec_executes(store, semantics, risk_cfg):
    raw = json.dumps({
        "route_id": "mi", "execution_mode": "risk",
        "risk_monitor_mode": "migration", "migration_dimension": "ifrs9_stage",
        "baseline_date": "2024-02-29", "current_date": "2024-03-31",
        "snapshot_client_id": CLIENT, "output_type": "table"})
    r = interpret_and_run_mi_query("show IFRS stage migration", _ctx(),
                                   FakeClient(raw), store, semantics=semantics,
                                   risk_config=risk_cfg)
    assert r.executed and r.runtime_result.ok
    deter = r.data[(r.data["from_value"] == "Stage 1")
                   & (r.data["to_value"] == "Stage 2")]
    assert not deter.empty and deter.iloc[0]["movement_type"] == "deteriorated"


# --------------------------------------------------------------------------- #
# 3. The deterministic interpreter already maps these (regression guard)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("q,checks", [
    ("show total pipeline", {"execution_mode": "state", "state": "total_pipeline"}),
    ("show funded balance by region",
     {"execution_mode": "risk", "dimension": "geographic_region_obligor"}),
    ("show concentration by region",
     {"execution_mode": "risk", "dimension": "geographic_region_obligor"}),
    ("show IFRS stage migration",
     {"execution_mode": "risk", "risk_monitor_mode": "migration",
      "dimension": "ifrs9_stage", "baseline_date": "2024-02-29",
      "current_date": "2024-03-31"}),
])
def test_deterministic_does_not_overclarify(q, checks):
    r = interpret(q, _ctx())
    assert not r.clarification_required, q
    assert r.ok, (q, r.issue_codes())
    for k, v in checks.items():
        assert getattr(r.normalized_spec, k) == v, (q, k)


# --------------------------------------------------------------------------- #
# 4. Genuinely ambiguous questions still clarify and never execute
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("q", ["show risk", "show changes", "show stage",
                               "show portfolio", "show rate"])
def test_ambiguous_still_clarifies_and_not_executed(q, store, semantics, risk_cfg):
    r = interpret_and_run_mi_query(q, _ctx(), interpret, store,
                                   semantics=semantics, risk_config=risk_cfg)
    assert r.clarification_required
    assert not r.executed and r.runtime_result is None


# --------------------------------------------------------------------------- #
# 5. Malformed / invalid output still never executes
# --------------------------------------------------------------------------- #


def test_malformed_not_executed(store, semantics, risk_cfg):
    r = interpret_and_run_mi_query("show total pipeline", _ctx(),
                                   FakeClient("not json {"), store,
                                   semantics=semantics, risk_config=risk_cfg)
    assert not r.executed and r.runtime_result is None


def test_invalid_enum_not_executed(store, semantics, risk_cfg):
    raw = json.dumps({"execution_mode": "state", "state": "not_a_state",
                      "temporal_mode": "latest"})
    r = interpret_and_run_mi_query("show total pipeline", _ctx(),
                                   FakeClient(raw), store, semantics=semantics,
                                   risk_config=risk_cfg)
    assert not r.executed
    assert "invalid_enum_value" in r.issue_codes()
