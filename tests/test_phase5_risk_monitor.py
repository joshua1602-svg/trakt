"""Phase 5 — risk monitor foundations tests.

LocalFsSnapshotStore + synthetic multi-snapshot frames covering migration
matrices, per-loan deterioration/improvement flags, concentration early-warning,
trend trajectory, route gating, and structured issues. No Streamlit / legacy
analytics / Azure imports; no Annex 2 / regulatory files modified.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pandas as pd
import pytest

from mi_agent.risk_monitor import (
    concentration_movement,
    funded_concentration,
    load_risk_monitor_config,
    migration_matrix,
    per_loan_movement,
    run_migration,
    run_trajectory,
    validate_risk_monitor_route,
)
from mi_agent.risk_monitor import models as RM
from snapshot.adapters import LocalFsSnapshotStore
from snapshot.model import SnapshotHeader

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def cfg():
    return load_risk_monitor_config()


def _df(rows):
    return pd.DataFrame(rows)


def _loan(lid, region, grade, ifrs, pd_b, bal, status="funded"):
    return {"loan_identifier": lid, "loan_id": lid, "geographic_region_obligor": region,
            "internal_risk_grade": grade, "ifrs9_stage": ifrs, "pd_bucket": pd_b,
            "funded_status": status, "current_outstanding_balance": float(bal)}


@pytest.fixture
def store3(tmp_path):
    store = LocalFsSnapshotStore(root=tmp_path / "s")
    s1 = _df([
        _loan("L1", "N", "A", "Stage 1", "<0.25%", 100),
        _loan("L2", "N", "B", "Stage 1", "0.25-0.5%", 100),
        _loan("L3", "S", "C", "Stage 2", "1-2.5%", 100),
    ])
    s2 = _df([
        _loan("L1", "N", "A", "Stage 1", "<0.25%", 100),
        _loan("L2", "N", "B", "Stage 2", "0.5-1%", 100),   # ifrs + pd deteriorate
        _loan("L3", "S", "C", "Stage 2", "1-2.5%", 200),
        _loan("L4", "S", "D", "Stage 1", "<0.25%", 100),   # new
    ])
    s3 = _df([
        _loan("L1", "N", "A", "Stage 1", "<0.25%", 100),
        _loan("L3", "S", "B", "Stage 1", "0.5-1%", 300),   # grade + ifrs improve
        _loan("L4", "S", "D", "Stage 1", "<0.25%", 300),
    ])                                                      # L2 exited
    for i, (rd, df) in enumerate([("2024-01-31", s1), ("2024-02-29", s2),
                                  ("2024-03-31", s3)]):
        h = SnapshotHeader(client_id="c1", route="mi", reporting_date=rd,
                           source_file_id=f"sha256:{i}", cadence="monthly",
                           upload_timestamp=f"{rd}T09:00:00")
        store.register_snapshot(h, df)
    return store


# --------------------------------------------------------------------------- #
# 1. Migration matrices
# --------------------------------------------------------------------------- #


def _frames():
    base = _df([
        _loan("L1", "N", "A", "Stage 1", "<0.25%", 100),
        _loan("L2", "N", "B", "Stage 1", "0.25-0.5%", 100),
        _loan("L3", "S", "C", "Stage 2", "1-2.5%", 100),
    ])
    cur = _df([
        _loan("L1", "N", "A", "Stage 2", "0.5-1%", 100),   # deteriorate
        _loan("L2", "N", "A", "Stage 1", "<0.25%", 100),   # improve
        _loan("L4", "S", "D", "Stage 1", "<0.25%", 100),   # new (L3 exited)
    ])
    return base, cur


def test_risk_grade_migration_matrix(cfg):
    base, cur = _frames()
    res = migration_matrix(base, cur, "internal_risk_grade", config=cfg)
    types = set(res.frame["movement_type"])
    # grade: L1 A->A unchanged, L2 B->A improved, L3 exited, L4 new.
    assert {"improved", "new", "exited", "unchanged"}.issubset(types)
    assert res.metadata["ordered"] is True


def test_ifrs9_migration_matrix(cfg):
    base, cur = _frames()
    res = migration_matrix(base, cur, "ifrs9_stage", config=cfg)
    row = res.frame[(res.frame["from_value"] == "Stage 1")
                    & (res.frame["to_value"] == "Stage 2")].iloc[0]
    assert row["movement_type"] == "deteriorated"
    assert row["loan_count"] == 1


def test_pd_bucket_migration_matrix(cfg):
    base, cur = _frames()
    res = migration_matrix(base, cur, "pd_bucket", config=cfg)
    assert "deteriorated" in set(res.frame["movement_type"])
    assert res.metadata["ordered"] is True


def test_unordered_dimension_changed_not_directional():
    base = _df([{"loan_id": "L1", "internal_risk_stage": "watch",
                 "current_outstanding_balance": 10.0},
                {"loan_id": "L2", "internal_risk_stage": "normal",
                 "current_outstanding_balance": 20.0}])
    cur = _df([{"loan_id": "L1", "internal_risk_stage": "normal",
                "current_outstanding_balance": 10.0},
               {"loan_id": "L2", "internal_risk_stage": "normal",
                "current_outstanding_balance": 20.0}])
    res = migration_matrix(base, cur, "internal_risk_stage", config=None)
    types = set(res.frame["movement_type"])
    assert "changed" in types and "unchanged" in types
    assert "improved" not in types and "deteriorated" not in types
    assert RM.UNORDERED_MIGRATION_DIMENSION in res.issue_codes()
    assert res.metadata["ordered"] is False


def test_migration_new_exited_unchanged_counts(cfg):
    base, cur = _frames()
    res = migration_matrix(base, cur, "ifrs9_stage", config=cfg)
    counts = res.frame.groupby("movement_type")["loan_count"].sum().to_dict()
    assert counts.get("new") == 1       # L4
    assert counts.get("exited") == 1    # L3


def test_migration_missing_stable_key():
    base = _df([{"ifrs9_stage": "Stage 1", "current_outstanding_balance": 1.0}])
    cur = _df([{"ifrs9_stage": "Stage 2", "current_outstanding_balance": 1.0}])
    res = migration_matrix(base, cur, "ifrs9_stage")
    assert not res.ok
    assert RM.MISSING_STABLE_KEY_FOR_MIGRATION in res.issue_codes()


def test_migration_missing_dimension(cfg):
    base, cur = _frames()
    res = migration_matrix(base, cur, "nonexistent_dim", config=cfg)
    assert RM.MISSING_MIGRATION_DIMENSION in res.issue_codes()


# --------------------------------------------------------------------------- #
# 2. Per-loan flags
# --------------------------------------------------------------------------- #


def test_per_loan_flags(cfg):
    base, cur = _frames()
    res = per_loan_movement(base, cur, "internal_risk_grade", config=cfg)
    by = res.frame.set_index("loan_id")
    # grade: L1 A->A unchanged; L2 B->A improved; L4 new; L3 exited.
    assert by.loc["L1", "movement_type"] == "unchanged"
    assert not bool(by.loc["L1", "deterioration_flag"])
    assert not bool(by.loc["L1", "improvement_flag"])
    assert by.loc["L2", "movement_type"] == "improved"
    assert bool(by.loc["L2", "improvement_flag"])
    assert by.loc["L4", "movement_type"] == "new"
    assert by.loc["L3", "movement_type"] == "exited"


def test_per_loan_ifrs9_deterioration(cfg):
    base, cur = _frames()
    res = per_loan_movement(base, cur, "ifrs9_stage", config=cfg)
    by = res.frame.set_index("loan_id")
    assert bool(by.loc["L1", "deterioration_flag"]) is True   # Stage1->Stage2
    assert by.loc["L1", "movement_type"] == "deteriorated"
    assert by.loc["L1", "balance_change"] == 0.0


def test_per_loan_pd_deterioration_and_improvement(cfg):
    base, cur = _frames()
    res = per_loan_movement(base, cur, "pd_bucket", config=cfg)
    by = res.frame.set_index("loan_id")
    assert bool(by.loc["L1", "deterioration_flag"]) is True   # <0.25% -> 0.5-1%
    assert bool(by.loc["L2", "improvement_flag"]) is True     # 0.25-0.5% -> <0.25%


# --------------------------------------------------------------------------- #
# 3. Concentration
# --------------------------------------------------------------------------- #


def test_funded_concentration_by_region(cfg):
    fr = _df([_loan(f"L{i}", r, "A", "Stage 1", "<0.25%", b)
              for i, (r, b) in enumerate(
                  [("N", 100), ("N", 100), ("S", 300), ("S", 300), ("S", 300)])])
    res = funded_concentration(fr, "geographic_region_obligor", config=cfg)
    by = res.frame.set_index("geographic_region_obligor")
    assert by.loc["S", "status"] == "red"           # 0.82 share
    assert by.loc["N", "status"] == "green"          # 0.18 share


def test_forecast_funded_concentration():
    # A forecast-shaped frame: concentrate on the forecast contribution column.
    fr = _df([
        {"loan_id": "L1", "geographic_region_obligor": "N",
         "forecast_contribution": 100.0},
        {"loan_id": "L2", "geographic_region_obligor": "S",
         "forecast_contribution": 400.0},
    ])
    res = funded_concentration(fr, "geographic_region_obligor",
                               balance_col="forecast_contribution")
    by = res.frame.set_index("geographic_region_obligor")
    assert by.loc["S", "balance_share"] == pytest.approx(0.8)


def test_concentration_amber_and_approaching():
    # A=0.28 share -> amber and approaching red(0.30); B=0.72 -> red.
    fr = _df([{"loan_id": "L1", "region": "A", "current_outstanding_balance": 28.0},
              {"loan_id": "L2", "region": "B", "current_outstanding_balance": 72.0}])
    res = funded_concentration(fr, "region",
                               thresholds={"amber": 0.20, "red": 0.30},
                               approaching_at=0.90)
    by = res.frame.set_index("region")
    assert by.loc["A", "status"] == "amber"
    assert bool(by.loc["A", "approaching_limit"]) is True
    assert by.loc["B", "status"] == "red"


def test_concentration_movement_baseline_current():
    base = _df([{"loan_id": "L1", "region": "N", "current_outstanding_balance": 100.0},
                {"loan_id": "L2", "region": "S", "current_outstanding_balance": 100.0}])
    cur = _df([{"loan_id": "L1", "region": "N", "current_outstanding_balance": 100.0},
               {"loan_id": "L2", "region": "S", "current_outstanding_balance": 300.0}])
    res = concentration_movement(base, cur, "region")
    by = res.frame.set_index("region")
    assert by.loc["S", "baseline_share"] == pytest.approx(0.5)
    assert by.loc["S", "current_share"] == pytest.approx(0.75)
    assert bool(by.loc["S", "increasing"]) is True


# --------------------------------------------------------------------------- #
# 4. Trajectory (trend-based)
# --------------------------------------------------------------------------- #


def test_trajectory_increasing_concentration_warning(store3, cfg):
    res = run_trajectory(store3, "c1", "geographic_region_obligor", route="mi",
                         start_date="2024-01-01", end_date="2024-12-31",
                         config=cfg)
    assert res.metadata["assembled"] is True
    by = res.frame.set_index("geographic_region_obligor")
    # Region S share rises 0.33 -> 0.60 -> 0.857 across 3 snapshots.
    assert bool(by.loc["S", "increasing"]) is True
    assert bool(by.loc["S", "warning"]) is True
    assert by.loc["S", "n_snapshots"] == 3


def test_trajectory_insufficient_snapshots(store3, cfg):
    res = run_trajectory(store3, "c1", "geographic_region_obligor", route="mi",
                         start_date="2024-01-01", end_date="2024-02-29",
                         config=cfg)  # only 2 snapshots
    assert RM.INSUFFICIENT_SNAPSHOTS_FOR_TRAJECTORY in res.issue_codes()


# --------------------------------------------------------------------------- #
# 5. Route gating
# --------------------------------------------------------------------------- #


def test_mi_route_allows_risk_monitor(store3, cfg):
    res = run_migration(store3, "c1", "ifrs9_stage", route="mi",
                        baseline_date="2024-01-31", current_date="2024-02-29",
                        config=cfg)
    assert res.metadata["assembled"] is True
    assert res.row_count > 0


def test_mna_route_rejects_unless_enabled():
    assert validate_risk_monitor_route("mna") is not None
    assert validate_risk_monitor_route("mna")["code"] == \
        RM.UNSUPPORTED_RISK_MONITOR_ROUTE
    assert validate_risk_monitor_route("mna", allow_mna_override=True) is None


def test_mna_override_assembles(tmp_path, cfg):
    store = LocalFsSnapshotStore(root=tmp_path / "mna")
    base, cur = _frames()
    for i, (rd, df) in enumerate([("2024-01-31", base), ("2024-02-29", cur)]):
        h = SnapshotHeader(client_id="m", route="mna", reporting_date=rd,
                           source_file_id=f"sha256:m{i}", cadence="adhoc",
                           upload_timestamp=f"{rd}T09:00:00")
        store.register_snapshot(h, df)
    res = run_migration(store, "m", "ifrs9_stage", route="mna",
                        baseline_date="2024-01-31", current_date="2024-02-29",
                        config=cfg, allow_mna_override=True)
    assert res.metadata["assembled"] is True


def test_regulatory_route_rejects_risk_monitor(store3, cfg):
    assert validate_risk_monitor_route("regulatory_annex2")["code"] == \
        RM.UNSUPPORTED_RISK_MONITOR_ROUTE
    res = run_migration(store3, "c1", "ifrs9_stage", route="regulatory_annex2",
                        baseline_date="2024-01-31", current_date="2024-02-29")
    assert res.metadata["assembled"] is False
    assert RM.UNSUPPORTED_RISK_MONITOR_ROUTE in res.issue_codes()


def test_run_migration_missing_snapshot(store3, cfg):
    res = run_migration(store3, "c1", "ifrs9_stage", route="mi",
                        baseline_date="2019-01-01", current_date="2024-02-29",
                        config=cfg)
    assert res.metadata["assembled"] is False
    assert RM.MISSING_BASELINE_SNAPSHOT in res.issue_codes()


# --------------------------------------------------------------------------- #
# 6. Guards
# --------------------------------------------------------------------------- #


def test_no_forbidden_imports_in_risk_monitor():
    pkg = REPO_ROOT / "mi_agent" / "risk_monitor"
    banned = ("import streamlit", "import plotly", "import azure",
              "from azure", "from analytics ", "from analytics.")
    for py in pkg.glob("*.py"):
        text = py.read_text(encoding="utf-8")
        for token in banned:
            assert token not in text, f"{py.name} contains forbidden {token!r}"
        for line in text.splitlines():
            s = line.strip()
            assert s != "import analytics" and not s.startswith("import analytics."), s


def test_no_regulatory_or_annex2_files_modified():
    try:
        if subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse",
                           "--verify", "main"], capture_output=True).returncode != 0:
            pytest.skip("no 'main' ref to diff against")
        diff = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "diff", "--name-only", "main"],
            capture_output=True, text=True, check=True).stdout.split()
        status = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "status", "--porcelain"],
            capture_output=True, text=True, check=True).stdout.splitlines()
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"git not available: {exc}")

    changed = set(diff) | {ln[3:].strip() for ln in status if ln.strip()}
    forbidden_prefixes = ("config/regime/", "config/delivery/", "engine/gate_",
                          "engine/delivery_xml_agent/", "engine/projection_agent/")
    forbidden_substr = ("annex2", "annex_2", "annex12", "_xsd", ".xsd")
    for path in changed:
        low = path.lower()
        assert not any(low.startswith(p) for p in forbidden_prefixes), path
        assert not any(s in low for s in forbidden_substr), path
