"""Phase 6B — MI runtime smoke pack.

Deterministic, end-to-end proof that the Phase 6 MI runtime (`run_mi_query`)
works over canonical snapshot frames persisted in a ``LocalFsSnapshotStore``
across three reporting dates, plus the existing flat single-CSV path. No
multi-artefact consolidation, no onboarding, no Azure, no Streamlit, no LLM.

Every state/temporal/risk path goes through the snapshot layer — the store is
never bypassed.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from mi_agent.mi_query_spec import MIQuerySpec
from mi_agent.mi_query_validator import load_mi_semantics
from mi_agent.mi_runtime import run_mi_query
from mi_agent.portfolio_reference import load_portfolio_reference_config
from mi_agent.risk_monitor import load_risk_monitor_config
from mi_agent.semantic_resolver import resolve_dimension
from snapshot.adapters import LocalFsSnapshotStore
from snapshot.model import SnapshotHeader

REPO_ROOT = Path(__file__).resolve().parents[1]
SEMANTICS_PATH = REPO_ROOT / "mi_agent" / "mi_semantics_field_registry.yaml"
PORTFOLIO_CFG = REPO_ROOT / "config" / "client" / "portfolio_reference_example.yaml"
FLAT_CSV = REPO_ROOT / "tests" / "fixtures" / "phase6b_flat_canonical.csv"

CLIENT = "smoke"
REPORTING_DATES = ["2024-01-31", "2024-02-29", "2024-03-31"]


# --------------------------------------------------------------------------- #
# Canonical synthetic snapshot frames (small, deterministic)
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
              "C", "Stage 2", "0.5-1%"),   # grade/ifrs/pd all deteriorate vs S1
        _loan("F3", "funded", "completed", 300, "North", "Broker A", "PF_001",
              "C", "Stage 2", "1-2.5%"),
        _loan("P1", "pipeline", "OFFER", 50, "North", "Broker A", "PF_001",
              prob=0.5),                     # P2 exited pipeline
    ])
    return [s1, s2, s3]


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def semantics():
    return load_mi_semantics(SEMANTICS_PATH)


@pytest.fixture(scope="module")
def portfolio_cfg():
    return load_portfolio_reference_config(PORTFOLIO_CFG)


@pytest.fixture(scope="module")
def risk_cfg():
    return load_risk_monitor_config()


@pytest.fixture
def store(tmp_path):
    """Three canonical MI snapshots registered through LocalFsSnapshotStore."""
    s = LocalFsSnapshotStore(root=tmp_path / "smoke_snaps")
    for i, (rd, frame) in enumerate(zip(REPORTING_DATES, _snapshot_frames())):
        s.register_snapshot(
            SnapshotHeader(client_id=CLIENT, route="mi", reporting_date=rd,
                           source_file_id=f"sha256:smoke{i}", cadence="monthly",
                           upload_timestamp=f"{rd}T09:00:00"),
            frame)
    return s


def _spec(**kw):
    kw.setdefault("route_id", "mi")
    kw.setdefault("snapshot_client_id", CLIENT)
    return MIQuerySpec(**kw)


# --------------------------------------------------------------------------- #
# State (latest)
# --------------------------------------------------------------------------- #


def test_total_funded_latest(semantics, store):
    res = run_mi_query(_spec(state="total_funded"), semantics=semantics,
                       store=store)
    assert res.mode == "state" and res.ok
    assert res.row_count == 3
    assert float(res.data["current_outstanding_balance"].sum()) == 620.0


def test_total_pipeline_latest(semantics, store):
    res = run_mi_query(_spec(state="total_pipeline"), semantics=semantics,
                       store=store)
    assert res.ok and res.row_count == 1
    assert float(res.data["current_outstanding_balance"].sum()) == 50.0


def test_total_forecast_funded_latest(semantics, store):
    res = run_mi_query(_spec(state="total_forecast_funded"), semantics=semantics,
                       store=store)
    assert res.ok
    # funded 620 + pipeline P1 50*0.5 = 645.
    assert res.metadata["forecast_funded_total"] == pytest.approx(645.0)


# --------------------------------------------------------------------------- #
# Temporal — compare & trend
# --------------------------------------------------------------------------- #


def test_funded_compare_baseline_current(semantics, store):
    res = run_mi_query(_spec(state="total_funded", temporal_mode="compare",
                             baseline_date="2024-01-31", current_date="2024-03-31"),
                       semantics=semantics, store=store)
    assert res.ok
    row = res.data.iloc[0]
    assert row["baseline_balance"] == 300.0 and row["current_balance"] == 620.0
    assert row["balance_change"] == 320.0
    assert row["new_count"] == 1 and row["retained_count"] == 2


def test_funded_trend(semantics, store):
    res = run_mi_query(_spec(state="total_funded", temporal_mode="trend",
                             start_date="2024-01-01", end_date="2024-12-31"),
                       semantics=semantics, store=store)
    assert res.ok and res.row_count == 3
    assert list(res.data["balance"]) == [300.0, 620.0, 620.0]
    assert res.chart_instruction == {"chart_type": "line"}


def test_pipeline_trend(semantics, store):
    res = run_mi_query(_spec(state="total_pipeline", temporal_mode="trend",
                             start_date="2024-01-01", end_date="2024-12-31"),
                       semantics=semantics, store=store)
    assert res.ok
    assert list(res.data["balance"]) == [90.0, 90.0, 50.0]


def test_forecast_funded_trend(semantics, store):
    res = run_mi_query(_spec(state="total_forecast_funded", temporal_mode="trend",
                             start_date="2024-01-01", end_date="2024-12-31"),
                       semantics=semantics, store=store)
    assert res.ok
    # S1 300+25+10=335 ; S2 620+25+18=663 ; S3 620+25=645.
    assert list(res.data["balance"]) == pytest.approx([335.0, 663.0, 645.0])


# --------------------------------------------------------------------------- #
# Grouped "by dimension" views (point-in-time concentration of a state)
# --------------------------------------------------------------------------- #


def test_funded_by_portfolio(semantics, store, risk_cfg, portfolio_cfg):
    # "portfolio" resolves to the Trakt portfolio reference (portfolio_id).
    dim = resolve_dimension("portfolio", portfolio_config=portfolio_cfg,
                            semantics=semantics).field
    assert dim == "portfolio_id"
    res = run_mi_query(_spec(risk_monitor="concentration", state="total_funded",
                             dimension=dim), semantics=semantics, store=store,
                       risk_config=risk_cfg)
    assert res.ok
    by = res.data.set_index("portfolio_id")
    assert by.loc["PF_001", "balance_sum"] == 400.0   # F1 + F3
    assert by.loc["PF_002", "balance_sum"] == 220.0   # F2


def test_funded_by_region(semantics, store, risk_cfg):
    res = run_mi_query(_spec(risk_monitor="concentration", state="total_funded",
                             dimension="geographic_region_obligor"),
                       semantics=semantics, store=store, risk_config=risk_cfg)
    assert res.ok
    by = res.data.set_index("geographic_region_obligor")
    assert by.loc["North", "balance_sum"] == 400.0
    assert by.loc["South", "balance_sum"] == 220.0


def test_pipeline_by_stage(semantics, store, risk_cfg):
    # Use the S2 cut where the pipeline spans two stages.
    res = run_mi_query(_spec(risk_monitor="concentration", state="total_pipeline",
                             dimension="pipeline_stage", as_of_date="2024-02-29"),
                       semantics=semantics, store=store, risk_config=risk_cfg)
    assert res.ok
    by = res.data.set_index("pipeline_stage")
    assert set(by.index) == {"OFFER", "APPLICATION"}
    assert by.loc["OFFER", "balance_sum"] == 50.0
    assert by.loc["APPLICATION", "balance_sum"] == 40.0


def test_forecast_funded_by_region(semantics, store, risk_cfg):
    res = run_mi_query(_spec(risk_monitor="concentration",
                             state="total_forecast_funded",
                             dimension="geographic_region_obligor"),
                       semantics=semantics, store=store, risk_config=risk_cfg)
    assert res.ok
    by = res.data.set_index("geographic_region_obligor")
    # North = F1(100) + F3(300) + P1 forecast(25) = 425 ; South = F2(220).
    assert by.loc["North", "balance_sum"] == pytest.approx(425.0)
    assert by.loc["South", "balance_sum"] == pytest.approx(220.0)


def test_forecast_funded_by_broker(semantics, store, risk_cfg):
    res = run_mi_query(_spec(risk_monitor="concentration",
                             state="total_forecast_funded",
                             dimension="broker_channel"),
                       semantics=semantics, store=store, risk_config=risk_cfg)
    assert res.ok
    by = res.data.set_index("broker_channel")
    assert by.loc["Broker A", "balance_sum"] == pytest.approx(425.0)


def test_concentration_warning(semantics, store, risk_cfg):
    # North holds ~64.5% of the funded book -> red against amber 0.20/red 0.30.
    res = run_mi_query(_spec(risk_monitor="concentration", state="total_funded",
                             dimension="geographic_region_obligor"),
                       semantics=semantics, store=store, risk_config=risk_cfg)
    by = res.data.set_index("geographic_region_obligor")
    assert by.loc["North", "status"] == "red"
    assert "status" in res.data.columns


# --------------------------------------------------------------------------- #
# Risk migration (baseline/current)
# --------------------------------------------------------------------------- #


def test_risk_grade_migration(semantics, store, risk_cfg):
    res = run_mi_query(_spec(risk_monitor="migration",
                             dimension="internal_risk_grade",
                             baseline_date="2024-01-31", current_date="2024-03-31"),
                       semantics=semantics, store=store, risk_config=risk_cfg)
    assert res.mode == "risk" and res.ok
    deter = res.data[(res.data["from_value"] == "B")
                     & (res.data["to_value"] == "C")]
    assert not deter.empty and deter.iloc[0]["movement_type"] == "deteriorated"


def test_ifrs9_migration(semantics, store, risk_cfg):
    res = run_mi_query(_spec(risk_monitor="migration", dimension="ifrs9_stage",
                             baseline_date="2024-01-31", current_date="2024-03-31"),
                       semantics=semantics, store=store, risk_config=risk_cfg)
    assert res.ok
    deter = res.data[(res.data["from_value"] == "Stage 1")
                     & (res.data["to_value"] == "Stage 2")]
    assert not deter.empty and deter.iloc[0]["movement_type"] == "deteriorated"


def test_pd_bucket_migration(semantics, store, risk_cfg):
    res = run_mi_query(_spec(risk_monitor="migration", dimension="pd_bucket",
                             baseline_date="2024-01-31", current_date="2024-03-31"),
                       semantics=semantics, store=store, risk_config=risk_cfg)
    assert res.ok
    assert "deteriorated" in set(res.data["movement_type"])   # F2 0.25-0.5% -> 0.5-1%


# --------------------------------------------------------------------------- #
# Existing flat single-CSV MI query still works
# --------------------------------------------------------------------------- #


def test_flat_single_csv_still_works(semantics):
    assert FLAT_CSV.exists()
    spec = MIQuerySpec(intent="chart", chart_type="bar",
                       metric="current_outstanding_balance",
                       dimension="collateral_geography", aggregation="sum")
    res = run_mi_query(spec, semantics=semantics, data=str(FLAT_CSV))
    assert res.mode == "flat" and res.ok
    assert res.row_count == 3   # North / South / East
    total = float(res.data[res.data.columns[1]].sum())
    assert total == pytest.approx(1_120_000.0)


def test_flat_uses_governed_chart_factory(semantics):
    spec = MIQuerySpec(intent="chart", chart_type="bar",
                       metric="current_outstanding_balance",
                       dimension="collateral_geography", aggregation="sum")
    res = run_mi_query(spec, semantics=semantics, data=str(FLAT_CSV),
                       build_chart=True)
    assert res.chart_instruction == {"chart_type": "bar"}
    assert res.metadata.get("chart", {}).get("rendered") is True


# --------------------------------------------------------------------------- #
# Snapshot layer is genuinely used (not bypassed)
# --------------------------------------------------------------------------- #


def test_snapshot_store_is_used(semantics, store):
    # Three snapshots are registered and resolvable through the store.
    assert len(store.list_snapshots(CLIENT, route="mi")) == 3
    latest = store.resolve_latest(CLIENT, route="mi")
    assert latest.reporting_date == "2024-03-31"
    # A state query with no store must fail with a structured issue (proving the
    # runtime depends on the snapshot layer for state queries).
    res = run_mi_query(_spec(state="total_funded"), semantics=semantics)
    assert not res.ok
    assert "snapshot_store_required" in res.issue_codes()
