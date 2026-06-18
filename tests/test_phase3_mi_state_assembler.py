"""Phase 3 — MI state assembler tests.

Use LocalFsSnapshotStore + small synthetic DataFrames to prove deterministic
state assembly, forecast-funded maths, cohort derivation, segmentation, route
eligibility, and graceful handling of optional-field gaps. No Streamlit / legacy
analytics / Azure imports; no Annex 2 / regulatory files modified.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pandas as pd
import pytest

from mi_agent.states import (
    SnapshotSelector,
    assemble_state,
    cohort_by_acquired_portfolio,
    cohort_by_date,
    cohort_by_portfolio,
    cohort_by_spv,
    is_state_allowed,
    total_forecast_funded,
    total_funded,
    total_pipeline,
    validate_state_for_route,
)
from mi_agent.states import models as M
from snapshot.adapters import LocalFsSnapshotStore
from snapshot.model import SnapshotHeader

REPO_ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


def base_frame():
    return pd.DataFrame({
        "loan_identifier": [f"L{i}" for i in range(6)],
        "funded_status": ["funded", "funded", "pipeline", "pipeline",
                          "funded", "pipeline"],
        "pipeline_stage": ["completed", "completed", "offer", "kfi",
                           "completed", "application"],
        "current_outstanding_balance": [100.0, 200.0, 50.0, 80.0, 150.0, 40.0],
        "forecast_funded_balance": [None, None, 30.0, None, None, None],
        "forecast_funding_probability": [None, None, None, 0.5, None, 0.25],
        "origination_date": ["2020-01-15", "2020-06-01", "2021-03-01",
                             "2021-09-09", "2020-11-11", "2021-01-01"],
        "funding_date": ["2020-02-01", "2020-07-01", "2021-04-01", "2021-10-01",
                         "2020-12-01", "2021-02-01"],
        "acquisition_date": ["2022-01-01", "2022-01-01", "2023-01-01",
                             "2023-01-01", "2022-06-01", "2023-01-01"],
        "portfolio_id": ["P1", "P1", "P2", "P2", "P1", "P2"],
        "spv_id": ["S1", "S1", "S1", "S2", "S2", "S2"],
        "acquired_portfolio_id": ["AP1", "AP1", "AP2", "AP2", "AP1", "AP2"],
    })


@pytest.fixture
def df():
    return base_frame()


@pytest.fixture
def store_with_snapshot(tmp_path):
    store = LocalFsSnapshotStore(root=tmp_path / "snaps")
    header = SnapshotHeader(
        client_id="c1", route="mi", reporting_date="2024-03-31",
        source_file_id="sha256:phase3", cadence="monthly",
        upload_timestamp="2024-04-02T10:00:00")
    store.register_snapshot(header, base_frame())
    return store


# --------------------------------------------------------------------------- #
# 1. total_funded / total_pipeline from a snapshot
# --------------------------------------------------------------------------- #


def test_total_funded_from_snapshot(store_with_snapshot):
    sel = SnapshotSelector.latest("c1", route="mi")
    res = total_funded(store_with_snapshot, selector=sel, route="mi")
    assert res.ok and res.row_count == 3
    assert set(res.frame["funded_status"]) == {"funded"}
    assert res.metadata["selection_method"] == "funded_status"
    assert res.metadata["snapshot_id"]


def test_total_pipeline_from_snapshot(store_with_snapshot):
    sel = SnapshotSelector.latest("c1", route="mi")
    res = total_pipeline(store_with_snapshot, selector=sel, route="mi")
    assert res.ok and res.row_count == 3
    assert set(res.frame["funded_status"]) == {"pipeline"}


def test_missing_snapshot_issue(tmp_path):
    store = LocalFsSnapshotStore(root=tmp_path / "empty")
    sel = SnapshotSelector.latest("nobody", route="mi")
    res = total_funded(store, selector=sel, route="mi")
    assert not res.ok
    assert M.MISSING_SNAPSHOT in res.issue_codes()


# --------------------------------------------------------------------------- #
# 2. funded/pipeline selection fallbacks
# --------------------------------------------------------------------------- #


def test_total_funded_fallback_all_rows_when_no_status():
    frame = pd.DataFrame({"current_outstanding_balance": [1.0, 2.0, 3.0]})
    res = total_funded(frame)
    assert res.row_count == 3
    assert res.metadata["selection_method"] == "fallback_all_funded"
    assert M.MISSING_FUNDED_STATUS in res.issue_codes()


def test_total_pipeline_no_fields_is_empty_with_issue():
    frame = pd.DataFrame({"current_outstanding_balance": [1.0, 2.0]})
    res = total_pipeline(frame)
    assert res.row_count == 0
    assert M.MISSING_PIPELINE_STAGE in res.issue_codes()


def test_funded_derived_from_pipeline_stage():
    frame = pd.DataFrame({
        "pipeline_stage": ["completed", "offer", "funded"],
        "current_outstanding_balance": [10.0, 20.0, 30.0],
    })
    res = total_funded(frame)
    assert res.row_count == 2  # completed + funded
    assert res.metadata["selection_method"] == "pipeline_stage_derived"


def test_missing_balance_field_issue():
    frame = pd.DataFrame({"funded_status": ["funded", "pipeline"]})
    res = total_funded(frame)
    assert M.MISSING_BALANCE_FIELD in res.issue_codes()


# --------------------------------------------------------------------------- #
# 3. total_forecast_funded
# --------------------------------------------------------------------------- #


def test_forecast_funded_uses_forecast_balance():
    frame = pd.DataFrame({
        "funded_status": ["funded", "pipeline"],
        "current_outstanding_balance": [100.0, 80.0],
        "forecast_funded_balance": [None, 60.0],
    })
    res = total_forecast_funded(frame)
    # funded 100 + explicit forecast 60 = 160.
    assert res.metadata["forecast_funded_total"] == pytest.approx(160.0)
    assert M.MISSING_FORECAST_PROBABILITY not in res.issue_codes()
    pipe = res.frame[res.frame["state_component"] == "forecast_pipeline"]
    assert float(pipe["forecast_contribution"].iloc[0]) == 60.0


def test_forecast_funded_uses_balance_times_probability():
    frame = pd.DataFrame({
        "funded_status": ["funded", "pipeline"],
        "current_outstanding_balance": [100.0, 80.0],
        "forecast_funding_probability": [None, 0.25],
    })
    res = total_forecast_funded(frame)
    # funded 100 + 80 * 0.25 = 120.
    assert res.metadata["forecast_funded_total"] == pytest.approx(120.0)
    assert M.MISSING_FORECAST_PROBABILITY not in res.issue_codes()


def test_forecast_balance_takes_priority_over_probability():
    frame = pd.DataFrame({
        "funded_status": ["pipeline"],
        "current_outstanding_balance": [80.0],
        "forecast_funded_balance": [70.0],
        "forecast_funding_probability": [0.5],  # would give 40; balance wins
    })
    res = total_forecast_funded(frame)
    assert res.metadata["forecast_funded_total"] == pytest.approx(70.0)


def test_missing_forecast_probability_produces_issue_retained_null():
    frame = pd.DataFrame({
        "funded_status": ["funded", "pipeline"],
        "current_outstanding_balance": [100.0, 80.0],
        # no forecast columns at all for the pipeline row
    })
    res = total_forecast_funded(frame)
    assert M.MISSING_FORECAST_PROBABILITY in res.issue_codes()
    assert res.metadata["unforecastable_pipeline_count"] == 1
    # funded only -> total 100; pipeline retained with null contribution.
    assert res.metadata["forecast_funded_total"] == pytest.approx(100.0)
    pipe = res.frame[res.frame["state_component"] == "forecast_pipeline"]
    assert pipe["forecast_contribution"].isna().all()
    assert len(pipe) == 1


def test_missing_forecast_probability_excluded_when_requested():
    frame = pd.DataFrame({
        "funded_status": ["funded", "pipeline"],
        "current_outstanding_balance": [100.0, 80.0],
    })
    res = total_forecast_funded(frame, include_unforecastable=False)
    pipe = res.frame[res.frame["state_component"] == "forecast_pipeline"]
    assert len(pipe) == 0
    assert M.MISSING_FORECAST_PROBABILITY in res.issue_codes()


def test_forecast_does_not_invent_probability():
    # A pipeline row with balance but no probability must NOT be assigned one.
    frame = pd.DataFrame({
        "funded_status": ["pipeline"],
        "current_outstanding_balance": [80.0],
    })
    res = total_forecast_funded(frame)
    assert res.metadata["forecast_funded_total"] == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# 4. Cohort states
# --------------------------------------------------------------------------- #


def test_cohort_by_origination_date(df):
    res = assemble_state("cohort_by_origination_date", df)
    assert res.ok and res.state == "cohort_by_origination_date"
    assert "origination_date_cohort" in res.frame.columns
    assert set(res.frame["origination_date_cohort"]) == {2020, 2021}


def test_cohort_by_funding_date(df):
    res = assemble_state("cohort_by_funding_date", df)
    assert res.ok
    assert "funding_date_cohort" in res.frame.columns


def test_cohort_by_acquisition_date_when_present(df):
    res = assemble_state("cohort_by_acquisition_date", df)
    assert res.ok
    assert "acquisition_date_cohort" in res.frame.columns
    assert set(res.frame["acquisition_date_cohort"]) == {2022, 2023}


def test_cohort_missing_date_field_is_issue_not_crash():
    frame = pd.DataFrame({"current_outstanding_balance": [1.0, 2.0]})
    res = assemble_state("cohort_by_acquisition_date", frame)
    assert M.MISSING_REQUIRED_STATE_FIELD in res.issue_codes()
    assert not res.ok  # error severity, but no exception


def test_cohort_months_on_book(df):
    res = cohort_by_date(df, date_field="origination_date",
                         as_of="2024-03-31")
    assert "months_on_book" in res.frame.columns
    assert res.metadata["months_on_book_start"] == "funding_date"
    assert (res.frame["months_on_book"] >= 0).all()


def test_cohort_quarter_period(df):
    res = cohort_by_date(df, date_field="origination_date", period="Q")
    assert str(res.frame["origination_date_cohort"].iloc[0]) == "2020Q1"


# --------------------------------------------------------------------------- #
# 5. Segmentation
# --------------------------------------------------------------------------- #


def test_cohort_by_portfolio_present(df):
    res = cohort_by_portfolio(df)
    assert res.ok
    assert res.metadata["segment_field"] == "portfolio_id"
    assert "portfolio_id" in res.frame.columns


def test_cohort_by_spv_present(df):
    res = cohort_by_spv(df)
    assert res.ok and res.metadata["segment_field"] == "spv_id"


def test_cohort_by_acquired_portfolio_present(df):
    res = cohort_by_acquired_portfolio(df)
    assert res.ok and res.metadata["segment_field"] == "acquired_portfolio_id"


def test_missing_segmentation_field_is_issue_not_crash():
    frame = pd.DataFrame({
        "origination_date": ["2020-01-01", "2021-01-01"],
        "current_outstanding_balance": [10.0, 20.0],
    })
    res = cohort_by_portfolio(frame)
    # cohort still builds; segmentation absence is a non-fatal optional issue.
    assert "origination_date_cohort" in res.frame.columns
    assert M.MISSING_OPTIONAL_STATE_FIELD in res.issue_codes()
    assert res.ok  # warning only, not an error


# --------------------------------------------------------------------------- #
# 6. Route eligibility
# --------------------------------------------------------------------------- #


def test_route_rejects_mna_pipeline_and_forecast(df):
    p = total_pipeline(df, route="mna")
    assert p.metadata["assembled"] is False
    assert M.UNSUPPORTED_STATE_FOR_ROUTE in p.issue_codes()

    f = total_forecast_funded(df, route="mna")
    assert f.metadata["assembled"] is False
    assert M.UNSUPPORTED_STATE_FOR_ROUTE in f.issue_codes()


def test_route_allows_mi_pipeline_and_forecast(df):
    assert total_pipeline(df, route="mi").metadata["assembled"] is True
    assert total_forecast_funded(df, route="mi").metadata["assembled"] is True


def test_route_allows_mna_funded_and_cohort(df):
    assert total_funded(df, route="mna").metadata["assembled"] is True
    assert is_state_allowed("cohort_by_date", "mna")
    assert is_state_allowed("cohort_by_origination_date", "mna")  # alias


def test_regulatory_route_rejects_mi_states():
    issue = validate_state_for_route("total_funded", "regulatory_annex2")
    assert issue is not None and issue["code"] == M.UNSUPPORTED_STATE_FOR_ROUTE


def test_mna_does_not_allow_spv_cohort():
    assert not is_state_allowed("cohort_by_spv", "mna")
    assert is_state_allowed("cohort_by_spv", "mi")


# --------------------------------------------------------------------------- #
# 7. Guards: no forbidden imports / no regulatory file changes
# --------------------------------------------------------------------------- #


def test_no_forbidden_imports_in_states_package():
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
    """The Phase 3 change set must not touch regulatory/Annex 2/XML logic."""
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
