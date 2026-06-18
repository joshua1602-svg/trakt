"""Phase 4 — temporal MI (compare / trend) tests.

Use LocalFsSnapshotStore + synthetic multi-snapshot DataFrames to prove
deterministic compare and trend over MI states, config-driven forecast
probability fallback, movement counts via stable keys, route eligibility, and
structured temporal issues. No Streamlit / legacy analytics / Azure imports; no
Annex 2 / regulatory files modified.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pandas as pd
import pytest

from mi_agent.states import (
    compare,
    load_stage_probabilities,
    trend,
)
from mi_agent.states import models as M
from snapshot.adapters import LocalFsSnapshotStore
from snapshot.model import SnapshotHeader

REPO_ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


def _row(loan, status, stage, bal, pf, spv, orig, prob=None):
    return {"loan_identifier": loan, "funded_status": status,
            "pipeline_stage": stage, "current_outstanding_balance": float(bal),
            "portfolio_id": pf, "spv_id": spv, "origination_date": orig,
            "forecast_funding_probability": prob}


def _register(store, rd, src, df, client="mi1", route="mi"):
    h = SnapshotHeader(client_id=client, route=route, reporting_date=rd,
                       source_file_id=src, cadence="monthly",
                       upload_timestamp=f"{rd}T09:00:00")
    return store.register_snapshot(h, df)


@pytest.fixture
def tstore(tmp_path):
    """Three monthly MI snapshots with funded + pipeline rows."""
    store = LocalFsSnapshotStore(root=tmp_path / "s")
    s1 = pd.DataFrame([
        _row("F1", "funded", "completed", 100, "PA", "SX", "2020-01-15"),
        _row("F2", "funded", "completed", 200, "PA", "SY", "2020-06-10"),
        _row("PL1", "pipeline", "OFFER", 50, "PA", "SX", "2021-02-02", 0.5),
    ])
    s2 = pd.DataFrame([
        _row("F1", "funded", "completed", 100, "PA", "SX", "2020-01-15"),
        _row("F2", "funded", "completed", 220, "PA", "SY", "2020-06-10"),
        _row("F3", "funded", "completed", 300, "PB", "SX", "2020-01-20"),
        _row("PL1", "pipeline", "OFFER", 50, "PA", "SX", "2021-02-02", 0.5),
        _row("PL2", "pipeline", "KFI", 80, "PB", "SY", "2020-06-25"),  # no prob
    ])
    s3 = pd.DataFrame([
        _row("F1", "funded", "completed", 100, "PA", "SX", "2020-01-15"),
        _row("F3", "funded", "completed", 300, "PB", "SX", "2020-01-20"),
        _row("F4", "funded", "completed", 400, "PB", "SY", "2021-03-03"),
        _row("PL2", "pipeline", "KFI", 80, "PB", "SY", "2020-06-25"),  # no prob
    ])
    _register(store, "2024-01-31", "sha256:1", s1)
    _register(store, "2024-02-29", "sha256:2", s2)
    _register(store, "2024-03-31", "sha256:3", s3)
    return store


# --------------------------------------------------------------------------- #
# 1. Compare — funded / pipeline
# --------------------------------------------------------------------------- #


def test_compare_total_funded(tstore):
    res = compare(tstore, "total_funded", "mi1", route="mi",
                  baseline_date="2024-01-31", current_date="2024-03-31")
    assert res.ok and res.row_count == 1
    r = res.frame.iloc[0]
    assert r["baseline_count"] == 2 and r["current_count"] == 3
    assert r["baseline_balance"] == 300.0 and r["current_balance"] == 800.0
    assert r["balance_change"] == 500.0
    assert r["count_change"] == 1


def test_compare_total_pipeline(tstore):
    res = compare(tstore, "total_pipeline", "mi1", route="mi",
                  baseline_date="2024-01-31", current_date="2024-03-31")
    r = res.frame.iloc[0]
    assert r["baseline_count"] == 1 and r["current_count"] == 1
    assert r["baseline_balance"] == 50.0 and r["current_balance"] == 80.0
    assert r["balance_change"] == 30.0


def test_compare_movement_new_exited_retained(tstore):
    res = compare(tstore, "total_funded", "mi1", route="mi",
                  baseline_date="2024-01-31", current_date="2024-03-31")
    r = res.frame.iloc[0]
    # baseline funded {F1,F2}; current {F1,F3,F4}.
    assert r["new_count"] == 2          # F3, F4
    assert r["exited_count"] == 1       # F2
    assert r["retained_count"] == 1     # F1


# --------------------------------------------------------------------------- #
# 2. Compare — forecast funded
# --------------------------------------------------------------------------- #


def test_compare_forecast_row_probability(tstore):
    # No config: PL1 uses its row probability (0.5); PL2 has none -> missing.
    res = compare(tstore, "total_forecast_funded", "mi1", route="mi",
                  baseline_date="2024-01-31", current_date="2024-02-29")
    r = res.frame.iloc[0]
    # baseline: funded 300 + PL1 50*0.5=25 -> 325.
    assert r["baseline_balance"] == pytest.approx(325.0)
    # current: funded 620 + PL1 25 + PL2(null) -> 645.
    assert r["current_balance"] == pytest.approx(645.0)
    assert M.MISSING_FORECAST_PROBABILITY in res.issue_codes()


def test_compare_forecast_config_stage_probability(tstore):
    sp = load_stage_probabilities()  # KFI -> 0.20
    res = compare(tstore, "total_forecast_funded", "mi1", route="mi",
                  baseline_date="2024-01-31", current_date="2024-02-29",
                  stage_probabilities=sp)
    r = res.frame.iloc[0]
    # current: funded 620 + PL1 25 + PL2 80*0.20=16 -> 661.
    assert r["current_balance"] == pytest.approx(661.0)
    assert M.FORECAST_PROBABILITY_FROM_CONFIG in res.issue_codes()


def test_forecast_does_not_invent_probability(tstore):
    res = compare(tstore, "total_forecast_funded", "mi1", route="mi",
                  baseline_date="2024-01-31", current_date="2024-02-29")
    # Without config, the KFI pipeline row contributes nothing (not invented).
    assert M.MISSING_FORECAST_PROBABILITY in res.issue_codes()
    assert M.FORECAST_PROBABILITY_FROM_CONFIG not in res.issue_codes()


# --------------------------------------------------------------------------- #
# 3. Compare — stratified + divide-by-zero
# --------------------------------------------------------------------------- #


def test_compare_stratified_by_portfolio(tstore):
    res = compare(tstore, "total_funded", "mi1", route="mi",
                  baseline_date="2024-01-31", current_date="2024-03-31",
                  stratify_by="portfolio_id")
    by = res.frame.set_index("portfolio_id")
    # PA baseline F1+F2=300, current F1=100 -> change -200.
    assert by.loc["PA", "baseline_balance"] == 300.0
    assert by.loc["PA", "current_balance"] == 100.0
    # PB baseline 0, current F3+F4=700.
    assert by.loc["PB", "baseline_balance"] == 0.0
    assert by.loc["PB", "current_balance"] == 700.0


def test_compare_divide_by_zero_pct(tmp_path):
    store = LocalFsSnapshotStore(root=tmp_path / "z")
    only_funded = pd.DataFrame([
        _row("F1", "funded", "completed", 100, "PA", "SX", "2020-01-15")])
    with_pipeline = pd.DataFrame([
        _row("F1", "funded", "completed", 100, "PA", "SX", "2020-01-15"),
        _row("PL1", "pipeline", "OFFER", 50, "PA", "SX", "2021-02-02", 0.5)])
    _register(store, "2024-01-31", "sha256:z1", only_funded, client="z")
    _register(store, "2024-02-29", "sha256:z2", with_pipeline, client="z")
    res = compare(store, "total_pipeline", "z", route="mi",
                  baseline_date="2024-01-31", current_date="2024-02-29")
    r = res.frame.iloc[0]
    assert r["baseline_balance"] == 0.0
    assert r["balance_pct_change"] is None
    assert M.PERCENTAGE_CHANGE_DIVIDE_BY_ZERO in res.issue_codes()


# --------------------------------------------------------------------------- #
# 4. Trend
# --------------------------------------------------------------------------- #


def test_trend_total_funded(tstore):
    res = trend(tstore, "total_funded", "mi1", route="mi",
                start_date="2024-01-01", end_date="2024-12-31")
    assert res.row_count == 3
    assert list(res.frame["balance"]) == [300.0, 620.0, 800.0]
    assert list(res.frame["count"]) == [2, 3, 3]
    assert list(res.frame["reporting_date"]) == ["2024-01-31", "2024-02-29",
                                                 "2024-03-31"]


def test_trend_total_pipeline(tstore):
    res = trend(tstore, "total_pipeline", "mi1", route="mi",
                start_date="2024-01-01", end_date="2024-12-31")
    assert list(res.frame["balance"]) == [50.0, 130.0, 80.0]


def test_trend_forecast_funded_with_config(tstore):
    sp = load_stage_probabilities()
    res = trend(tstore, "total_forecast_funded", "mi1", route="mi",
                start_date="2024-01-01", end_date="2024-12-31",
                stage_probabilities=sp)
    # S1 325, S2 661, S3 800 + PL2 16 = 816.
    assert list(res.frame["balance"]) == pytest.approx([325.0, 661.0, 816.0])
    assert M.FORECAST_PROBABILITY_FROM_CONFIG in res.issue_codes()


def test_trend_by_portfolio(tstore):
    res = trend(tstore, "total_funded", "mi1", route="mi",
                start_date="2024-01-01", end_date="2024-12-31",
                stratify_by="portfolio_id")
    assert "portfolio_id" in res.frame.columns
    # Final snapshot: PA = F1 (100); PB = F3+F4 (700).
    last = res.frame[res.frame["reporting_date"] == "2024-03-31"].set_index(
        "portfolio_id")
    assert last.loc["PA", "balance"] == 100.0
    assert last.loc["PB", "balance"] == 700.0


def test_trend_by_spv(tstore):
    res = trend(tstore, "total_funded", "mi1", route="mi",
                start_date="2024-01-01", end_date="2024-12-31",
                segment="spv_id")
    assert "spv_id" in res.frame.columns
    assert set(res.frame["spv_id"]) <= {"SX", "SY"}


def test_cohort_trend_by_origination_month(tstore):
    res = trend(tstore, "cohort_by_origination_date", "mi1", route="mi",
                start_date="2024-01-01", end_date="2024-12-31",
                stratify_by="origination_date_cohort", period="M")
    assert "origination_date_cohort" in res.frame.columns
    assert res.row_count > 0
    # Each snapshot's cohort balances sum to that snapshot's total balance.
    s1 = res.frame[res.frame["reporting_date"] == "2024-01-31"]
    assert s1["balance"].sum() == pytest.approx(350.0)  # 100+200+50


# --------------------------------------------------------------------------- #
# 5. Missing snapshots / insufficient range
# --------------------------------------------------------------------------- #


def test_missing_baseline_snapshot(tstore):
    res = compare(tstore, "total_funded", "mi1", route="mi",
                  baseline_date="2019-01-01", current_date="2024-03-31")
    assert not res.ok
    assert M.MISSING_BASELINE_SNAPSHOT in res.issue_codes()


def test_missing_current_snapshot(tstore):
    # current before any snapshot -> no snapshot on/before it.
    res = compare(tstore, "total_funded", "mi1", route="mi",
                  baseline_date="2024-01-31", current_date="2019-01-01")
    assert M.MISSING_CURRENT_SNAPSHOT in res.issue_codes()


def test_insufficient_snapshots_for_trend(tstore):
    res = trend(tstore, "total_funded", "mi1", route="mi",
                start_date="2024-01-01", end_date="2024-02-01")  # only S1
    assert M.INSUFFICIENT_SNAPSHOTS_FOR_TREND in res.issue_codes()


def test_empty_range_trend(tstore):
    res = trend(tstore, "total_funded", "mi1", route="mi",
                start_date="2025-01-01", end_date="2025-12-31")
    assert M.EMPTY_TEMPORAL_RESULT in res.issue_codes()


# --------------------------------------------------------------------------- #
# 6. Route eligibility
# --------------------------------------------------------------------------- #


def test_mna_route_rejects_compare_mode(tstore):
    res = compare(tstore, "total_funded", "mi1", route="mna",
                  baseline_date="2024-01-31", current_date="2024-03-31")
    assert res.metadata["assembled"] is False
    assert M.UNAVAILABLE_TEMPORAL_MODE in res.issue_codes()


def test_mna_route_rejects_trend_mode(tstore):
    res = trend(tstore, "total_funded", "mi1", route="mna",
                start_date="2024-01-01", end_date="2024-12-31")
    assert res.metadata["assembled"] is False
    assert M.UNAVAILABLE_TEMPORAL_MODE in res.issue_codes()


def test_mna_route_rejects_pipeline_state_outright(tstore):
    res = compare(tstore, "total_pipeline", "mi1", route="mna",
                  baseline_date="2024-01-31", current_date="2024-03-31")
    assert M.UNSUPPORTED_TEMPORAL_STATE in res.issue_codes()


def test_regulatory_route_rejects_temporal_mi(tstore):
    res = compare(tstore, "total_funded", "mi1", route="regulatory_annex2",
                  baseline_date="2024-01-31", current_date="2024-03-31")
    assert M.UNSUPPORTED_TEMPORAL_STATE in res.issue_codes()


def test_mi_route_allows_compare_and_trend(tstore):
    c = compare(tstore, "total_forecast_funded", "mi1", route="mi",
                baseline_date="2024-01-31", current_date="2024-03-31")
    t = trend(tstore, "total_pipeline", "mi1", route="mi",
              start_date="2024-01-01", end_date="2024-12-31")
    assert c.metadata["assembled"] is True
    assert t.metadata["assembled"] is True


# --------------------------------------------------------------------------- #
# 7. Guards
# --------------------------------------------------------------------------- #


def test_no_forbidden_imports_in_temporal():
    pkg = REPO_ROOT / "mi_agent" / "states"
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
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"git not available: {exc}")

    changed = set(diff) | {ln[3:].strip() for ln in status if ln.strip()}
    forbidden_prefixes = ("config/regime/", "config/delivery/", "engine/gate_",
                          "engine/delivery_xml_agent/", "engine/projection_agent/")
    forbidden_substr = ("annex2", "annex_2", "annex12", "_xsd", ".xsd")
    for path in changed:
        low = path.lower()
        assert not any(low.startswith(p) for p in forbidden_prefixes), path
        assert not any(s in low for s in forbidden_substr), path
