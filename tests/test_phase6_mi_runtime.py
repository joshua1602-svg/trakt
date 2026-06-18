"""Phase 6 — MI runtime integration tests (read-only/query-only).

Step 0 compatibility/semantic fixes + governed runtime dispatch over
LocalFsSnapshotStore. Backward compatibility with the flat single-CSV MI Agent
is asserted. No LLM parsing, no Azure, no legacy analytics imports, no Annex 2
files touched.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pandas as pd
import pytest

from mi_agent.mi_query_spec import MIQuerySpec
from mi_agent.mi_query_validator import load_mi_semantics
from mi_agent.mi_runtime import run_mi_query
from mi_agent import mi_runtime as RT
from mi_agent.portfolio_reference import load_portfolio_reference_config
from mi_agent.quantile_buckets import materialise_quantile_bucket
from mi_agent.semantic_resolver import (
    INVALID_STAGE_CONTEXT,
    MISSING_PORTFOLIO_REFERENCE_CONFIG,
    resolve_dimension,
    resolve_route_dimensions,
)
from mi_agent.risk_monitor import load_risk_monitor_config
from snapshot.adapters import LocalFsSnapshotStore
from snapshot.model import SnapshotHeader

REPO_ROOT = Path(__file__).resolve().parents[1]
PORTFOLIO_CFG = REPO_ROOT / "config" / "client" / "portfolio_reference_example.yaml"


@pytest.fixture(scope="module")
def semantics():
    return load_mi_semantics(
        REPO_ROOT / "mi_agent" / "mi_semantics_field_registry.yaml")


@pytest.fixture(scope="module")
def portfolio_cfg():
    return load_portfolio_reference_config(PORTFOLIO_CFG)


@pytest.fixture(scope="module")
def risk_cfg():
    return load_risk_monitor_config()


def _row(lid, status, stage, bal, region, grade, prob=None):
    return {"loan_identifier": lid, "funded_status": status,
            "pipeline_stage": stage, "current_outstanding_balance": float(bal),
            "geographic_region_obligor": region, "internal_risk_grade": grade,
            "forecast_funding_probability": prob, "origination_date": "2020-01-15"}


def _register(store, rd, src, df, client="c1", route="mi", cadence="monthly"):
    store.register_snapshot(
        SnapshotHeader(client_id=client, route=route, reporting_date=rd,
                       source_file_id=src, cadence=cadence,
                       upload_timestamp=f"{rd}T09:00:00"), df)


@pytest.fixture
def store(tmp_path):
    s = LocalFsSnapshotStore(root=tmp_path / "snaps")
    s1 = pd.DataFrame([
        _row("F1", "funded", "completed", 100, "N", "A"),
        _row("F2", "funded", "completed", 200, "S", "B"),
        _row("P1", "pipeline", "offer", 50, "N", None, 0.5),
    ])
    s2 = pd.DataFrame([
        _row("F1", "funded", "completed", 100, "N", "A"),
        _row("F2", "funded", "completed", 220, "S", "B"),
        _row("F3", "funded", "completed", 300, "N", "C"),
        _row("P1", "pipeline", "offer", 50, "N", None, 0.5),
    ])
    s3 = pd.DataFrame([
        _row("F1", "funded", "completed", 100, "N", "A"),
        _row("F2", "funded", "completed", 220, "S", "C"),   # B->C deteriorate
        _row("F3", "funded", "completed", 300, "N", "C"),
        _row("P1", "pipeline", "offer", 50, "N", None, 0.5),
    ])
    _register(s, "2024-01-31", "sha256:1", s1)
    _register(s, "2024-02-29", "sha256:2", s2)
    _register(s, "2024-03-31", "sha256:3", s3)
    return s


@pytest.fixture
def flat_df():
    return pd.DataFrame({
        "loan_identifier": [f"L{i}" for i in range(6)],
        "current_outstanding_balance": [100, 200, 300, 400, 500, 600],
        "collateral_geography": ["N", "S", "E", "N", "S", "E"],
        "current_loan_to_value": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
    })


# =========================================================================
# Step 0 — semantic compatibility fixes
# =========================================================================


def test_portfolio_resolves_to_trakt_reference(semantics, portfolio_cfg):
    r = resolve_dimension("portfolio", portfolio_config=portfolio_cfg,
                          semantics=semantics)
    assert r.field == "portfolio_id" and r.kind == "portfolio_ref"


def test_acquired_portfolio_resolves(semantics):
    assert resolve_dimension("acquired portfolio",
                             semantics=semantics).field == "acquired_portfolio_id"


def test_spv_resolves(semantics):
    assert resolve_dimension("spv", semantics=semantics).field == "spv_id"


def test_portfolio_without_config_is_issue(semantics):
    r = resolve_dimension("portfolio", portfolio_config=None, semantics=semantics)
    assert r.field is None
    assert r.issues[0]["code"] == MISSING_PORTFOLIO_REFERENCE_CONFIG


def test_portfolio_never_resolves_to_acquired(semantics):
    r = resolve_dimension("portfolio", portfolio_config=None, semantics=semantics)
    assert r.field != "acquired_portfolio_id"


def test_stage_resolves_to_pipeline_stage_in_pipeline_context(semantics):
    assert resolve_dimension("stage", context="pipeline",
                             semantics=semantics).field == "pipeline_stage"
    assert resolve_dimension("stage", state="total_pipeline",
                             semantics=semantics).field == "pipeline_stage"


def test_pipeline_stage_term(semantics):
    assert resolve_dimension("pipeline stage", context="pipeline",
                             semantics=semantics).field == "pipeline_stage"


def test_ifrs_stage_term(semantics):
    assert resolve_dimension("IFRS stage", semantics=semantics).field == "ifrs9_stage"
    assert resolve_dimension("IFRS 9 stage", semantics=semantics).field == "ifrs9_stage"


def test_risk_stage_term(semantics):
    assert resolve_dimension("risk stage",
                             semantics=semantics).field == "internal_risk_stage"
    assert resolve_dimension("internal risk stage",
                             semantics=semantics).field == "internal_risk_stage"


@pytest.mark.parametrize("ctx,state", [("funded", "total_funded"),
                                       ("mna", None), ("regulatory", None),
                                       ("risk", None)])
def test_stage_invalid_context(semantics, ctx, state):
    r = resolve_dimension("stage", context=ctx, state=state, semantics=semantics)
    assert r.field is None
    assert r.issues[0]["code"] == INVALID_STAGE_CONTEXT


def test_balance_bucket_quantile_default():
    df = pd.DataFrame({"current_outstanding_balance": list(range(1, 13))})
    out, issues = materialise_quantile_bucket(df, "balance_band")
    assert set(out["balance_band"]) == {"Q1", "Q2", "Q3", "Q4"}
    assert not [i for i in issues if i["code"] == "quantile_bucket_insufficient_data"]


def test_interest_rate_bucket_quantile_default():
    df = pd.DataFrame({"current_interest_rate": [0.01, 0.02, 0.03, 0.04,
                                                 0.05, 0.06, 0.07, 0.08]})
    out, _ = materialise_quantile_bucket(df, "interest_rate_bucket")
    assert "interest_rate_bucket" in out.columns
    assert out["interest_rate_bucket"].nunique() >= 2


def test_time_on_book_bucket_quantile_default():
    df = pd.DataFrame({"months_on_book": [1, 3, 6, 12, 18, 24, 36, 60]})
    out, _ = materialise_quantile_bucket(df, "time_on_book_bucket")
    assert "time_on_book_bucket" in out.columns
    assert out["time_on_book_bucket"].nunique() >= 2


def test_quantile_insufficient_data_issue():
    df = pd.DataFrame({"current_outstanding_balance": [5.0]})
    out, issues = materialise_quantile_bucket(df, "balance_band")
    assert "balance_band" not in out.columns
    assert issues[0]["code"] == "quantile_bucket_insufficient_data"


def test_route_dimensions_resolve(semantics, portfolio_cfg):
    for route in ("mi", "mna"):
        res = resolve_route_dimensions(route, semantics=semantics,
                                       portfolio_config=portfolio_cfg)
        unresolved = [d for d, r in res.items() if not r.ok]
        assert not unresolved, f"{route}: {unresolved}"


def test_no_stale_registry_count_or_version(semantics):
    m = semantics["metadata"]
    # Dynamic, not pinned: count must equal the actual field map length.
    assert m["field_count"] == len(semantics["fields"])
    assert isinstance(m["version"], str) and m["version"]


# =========================================================================
# Runtime — flat backward compatibility
# =========================================================================


def test_flat_query_still_works(semantics, flat_df):
    spec = MIQuerySpec(intent="chart", chart_type="bar",
                       metric="current_outstanding_balance",
                       dimension="collateral_geography", aggregation="sum")
    res = run_mi_query(spec, semantics=semantics, data=flat_df)
    assert res.mode == "flat" and res.ok
    assert res.row_count == 3
    assert res.chart_instruction == {"chart_type": "bar"}


def test_flat_uses_existing_chart_factory(semantics, flat_df):
    spec = MIQuerySpec(intent="chart", chart_type="bar",
                       metric="current_outstanding_balance",
                       dimension="collateral_geography", aggregation="sum")
    res = run_mi_query(spec, semantics=semantics, data=flat_df, build_chart=True)
    assert res.metadata.get("chart", {}).get("rendered") is True


def test_flat_rejects_virtual_field_without_crash(semantics, flat_df):
    spec = MIQuerySpec(intent="table",
                       metric="current_outstanding_balance",
                       dimension="portfolio_id", aggregation="sum")
    res = run_mi_query(spec, semantics=semantics, data=flat_df)
    assert not res.ok
    assert RT.VIRTUAL_FIELD_NOT_AVAILABLE_IN_FLAT_MODE in res.issue_codes()


# =========================================================================
# Runtime — state path
# =========================================================================


def test_state_total_funded(semantics, store):
    spec = MIQuerySpec(route_id="mi", state="total_funded",
                       snapshot_client_id="c1")
    res = run_mi_query(spec, semantics=semantics, store=store)
    assert res.mode == "state" and res.ok
    assert res.row_count == 3                 # latest snapshot funded rows


def test_state_total_pipeline(semantics, store):
    spec = MIQuerySpec(route_id="mi", state="total_pipeline",
                       snapshot_client_id="c1")
    res = run_mi_query(spec, semantics=semantics, store=store)
    assert res.ok and res.row_count == 1


def test_state_total_forecast_funded(semantics, store):
    spec = MIQuerySpec(route_id="mi", state="total_forecast_funded",
                       snapshot_client_id="c1")
    res = run_mi_query(spec, semantics=semantics, store=store)
    assert res.ok
    # funded (620) + pipeline 50*0.5 = 645.
    assert res.metadata["forecast_funded_total"] == pytest.approx(645.0)


def test_state_missing_client_id(semantics, store):
    spec = MIQuerySpec(state="total_funded")
    res = run_mi_query(spec, semantics=semantics, store=store)
    assert RT.MISSING_SNAPSHOT_CLIENT_ID in res.issue_codes()


def test_state_missing_store(semantics):
    spec = MIQuerySpec(state="total_funded", snapshot_client_id="c1")
    res = run_mi_query(spec, semantics=semantics)
    assert RT.SNAPSHOT_STORE_REQUIRED in res.issue_codes()


# =========================================================================
# Runtime — temporal path
# =========================================================================


def test_temporal_funded_compare(semantics, store):
    spec = MIQuerySpec(route_id="mi", state="total_funded",
                       temporal_mode="compare", baseline_date="2024-01-31",
                       current_date="2024-03-31", snapshot_client_id="c1")
    res = run_mi_query(spec, semantics=semantics, store=store)
    assert res.mode == "temporal" and res.ok
    r = res.data.iloc[0]
    assert r["baseline_balance"] == 300.0 and r["current_balance"] == 620.0


def test_temporal_funded_trend(semantics, store):
    spec = MIQuerySpec(route_id="mi", state="total_funded",
                       temporal_mode="trend", start_date="2024-01-01",
                       end_date="2024-12-31", snapshot_client_id="c1")
    res = run_mi_query(spec, semantics=semantics, store=store)
    assert res.ok and res.row_count == 3
    assert list(res.data["balance"]) == [300.0, 620.0, 620.0]
    assert res.chart_instruction == {"chart_type": "line"}


def test_temporal_pipeline_trend(semantics, store):
    spec = MIQuerySpec(route_id="mi", state="total_pipeline",
                       temporal_mode="trend", start_date="2024-01-01",
                       end_date="2024-12-31", snapshot_client_id="c1")
    res = run_mi_query(spec, semantics=semantics, store=store)
    assert res.ok and list(res.data["balance"]) == [50.0, 50.0, 50.0]


def test_temporal_forecast_trend(semantics, store):
    spec = MIQuerySpec(route_id="mi", state="total_forecast_funded",
                       temporal_mode="trend", start_date="2024-01-01",
                       end_date="2024-12-31", snapshot_client_id="c1")
    res = run_mi_query(spec, semantics=semantics, store=store)
    assert res.ok
    assert list(res.data["balance"]) == pytest.approx([325.0, 645.0, 645.0])


def test_temporal_incomplete_selector(semantics, store):
    spec = MIQuerySpec(route_id="mi", state="total_funded",
                       temporal_mode="compare", snapshot_client_id="c1")
    res = run_mi_query(spec, semantics=semantics, store=store)
    assert RT.TEMPORAL_SELECTOR_INCOMPLETE in res.issue_codes()


# =========================================================================
# Runtime — risk path
# =========================================================================


def test_risk_grade_migration(semantics, store, risk_cfg):
    spec = MIQuerySpec(route_id="mi", risk_monitor="migration",
                       dimension="internal_risk_grade",
                       baseline_date="2024-01-31", current_date="2024-03-31",
                       snapshot_client_id="c1")
    res = run_mi_query(spec, semantics=semantics, store=store, risk_config=risk_cfg)
    assert res.mode == "risk" and res.ok
    assert "deteriorated" in set(res.data["movement_type"])  # F2 B->C


def test_risk_concentration_warning(semantics, store, risk_cfg):
    spec = MIQuerySpec(route_id="mi", risk_monitor="concentration",
                       dimension="geographic_region_obligor",
                       snapshot_client_id="c1")
    res = run_mi_query(spec, semantics=semantics, store=store, risk_config=risk_cfg)
    assert res.mode == "risk" and res.ok
    assert "status" in res.data.columns and res.row_count >= 1


def test_risk_trajectory(semantics, store, risk_cfg):
    spec = MIQuerySpec(route_id="mi", risk_monitor="trajectory",
                       dimension="geographic_region_obligor",
                       start_date="2024-01-01", end_date="2024-12-31",
                       snapshot_client_id="c1")
    res = run_mi_query(spec, semantics=semantics, store=store, risk_config=risk_cfg)
    assert res.mode == "risk" and res.ok
    assert "warning" in res.data.columns


# =========================================================================
# Runtime — route gating
# =========================================================================


def test_mna_route_rejects_pipeline_state(semantics, store):
    spec = MIQuerySpec(route_id="mna", state="total_pipeline",
                       snapshot_client_id="c1")
    res = run_mi_query(spec, semantics=semantics, store=store)
    assert RT.INVALID_ROUTE_FOR_STATE in res.issue_codes()


def test_mna_route_rejects_forecast_state(semantics, store):
    spec = MIQuerySpec(route_id="mna", state="total_forecast_funded",
                       snapshot_client_id="c1")
    res = run_mi_query(spec, semantics=semantics, store=store)
    assert RT.INVALID_ROUTE_FOR_STATE in res.issue_codes()


def test_mna_route_rejects_risk_unless_enabled(semantics, store, risk_cfg):
    spec = MIQuerySpec(route_id="mna", risk_monitor="concentration",
                       dimension="geographic_region_obligor",
                       snapshot_client_id="c1")
    res = run_mi_query(spec, semantics=semantics, store=store, risk_config=risk_cfg)
    assert RT.RISK_MONITOR_NOT_ENABLED in res.issue_codes()


def test_mna_route_risk_enabled_override(semantics, tmp_path, risk_cfg):
    s = LocalFsSnapshotStore(root=tmp_path / "mna")
    df = pd.DataFrame([
        _row("F1", "funded", "completed", 100, "N", "A"),
        _row("F2", "funded", "completed", 200, "S", "B"),
    ])
    _register(s, "2024-01-31", "sha256:m1", df, client="m", route="mna",
              cadence="adhoc")
    spec = MIQuerySpec(route_id="mna", risk_monitor="concentration",
                       dimension="geographic_region_obligor",
                       snapshot_client_id="m")
    res = run_mi_query(spec, semantics=semantics, store=s, risk_config=risk_cfg,
                       allow_mna_risk=True)
    assert res.ok and RT.RISK_MONITOR_NOT_ENABLED not in res.issue_codes()


def test_temporal_mna_rejected(semantics, store):
    spec = MIQuerySpec(route_id="mna", state="total_funded",
                       temporal_mode="trend", start_date="2024-01-01",
                       end_date="2024-12-31", snapshot_client_id="c1")
    res = run_mi_query(spec, semantics=semantics, store=store)
    assert RT.INVALID_TEMPORAL_MODE_FOR_ROUTE in res.issue_codes()


def test_regulatory_route_rejects_state(semantics, store):
    spec = MIQuerySpec(route_id="regulatory_annex2", state="total_funded",
                       snapshot_client_id="c1")
    res = run_mi_query(spec, semantics=semantics, store=store)
    assert RT.INVALID_ROUTE_FOR_STATE in res.issue_codes()


def test_regulatory_route_rejects_temporal(semantics, store):
    spec = MIQuerySpec(route_id="regulatory_annex2", state="total_funded",
                       temporal_mode="trend", start_date="2024-01-01",
                       end_date="2024-12-31", snapshot_client_id="c1")
    res = run_mi_query(spec, semantics=semantics, store=store)
    assert not res.ok


# =========================================================================
# Guards
# =========================================================================


def test_no_forbidden_imports_in_phase6_modules():
    files = ["mi_runtime.py", "semantic_resolver.py", "portfolio_reference.py",
             "quantile_buckets.py"]
    banned = ("import streamlit", "import plotly", "import azure",
              "from azure", "from analytics ", "from analytics.")
    for fn in files:
        text = (REPO_ROOT / "mi_agent" / fn).read_text(encoding="utf-8")
        for token in banned:
            assert token not in text, f"{fn} contains forbidden {token!r}"
        for line in text.splitlines():
            s = line.strip()
            assert s != "import analytics" and not s.startswith("import analytics."), s


def test_no_regulatory_or_annex2_files_modified():
    try:
        if subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse",
                           "--verify", "main"], capture_output=True).returncode != 0:
            pytest.skip("no 'main' ref")
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
