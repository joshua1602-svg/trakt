"""Phase 6C — multi-artefact consolidation proof for the MI runtime.

Proves Trakt can take fragmented synthetic source artefacts (borrowers, loans,
collateral, cashflows, portfolio/SPV map, pipeline), deterministically
consolidate them into canonical MI snapshot frames across three reporting dates,
register them in a LocalFsSnapshotStore, and run the same governed MI runtime
(`run_mi_query`) over them. Deterministic synthetic proof only — not a
production consolidation engine. No onboarding/Azure/Streamlit/LLM.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

from mi_agent.mi_query_spec import MIQuerySpec
from mi_agent.mi_query_validator import load_mi_semantics
from mi_agent.mi_runtime import run_mi_query
from mi_agent.risk_monitor import load_risk_monitor_config
from snapshot.adapters import LocalFsSnapshotStore

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures" / "phase6c_multi_artifact"
SEMANTICS_PATH = REPO_ROOT / "mi_agent" / "mi_semantics_field_registry.yaml"
HELPER_PATH = REPO_ROOT / "tests" / "helpers" / "phase6c_consolidation.py"
FLAT_CSV = REPO_ROOT / "tests" / "fixtures" / "phase6b_flat_canonical.csv"

CLIENT = "smoke"
REPORTING_DATES = ["2024-01-31", "2024-02-29", "2024-03-31"]


# Load the consolidation helper by file path (no package-import assumptions).
def _load_helper():
    spec = importlib.util.spec_from_file_location("phase6c_consolidation",
                                                  HELPER_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


C = _load_helper()


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def semantics():
    return load_mi_semantics(SEMANTICS_PATH)


@pytest.fixture(scope="module")
def risk_cfg():
    return load_risk_monitor_config()


@pytest.fixture(scope="module")
def artifacts():
    arts, issues = C.load_artifacts(FIXTURE_DIR)
    return arts, issues


@pytest.fixture(scope="module")
def consolidated(artifacts):
    arts, _ = artifacts
    frames, lineage, issues = C.consolidate(arts, client_id=CLIENT)
    return frames, lineage, issues


@pytest.fixture
def store(tmp_path, consolidated):
    frames, _, _ = consolidated
    s = LocalFsSnapshotStore(root=tmp_path / "6c_snaps")
    C.register_snapshots(s, frames, client_id=CLIENT, route="mi")
    return s


def _spec(**kw):
    kw.setdefault("route_id", "mi")
    kw.setdefault("snapshot_client_id", CLIENT)
    return MIQuerySpec(**kw)


# =========================================================================
# 1. Artefacts load
# =========================================================================


def test_artifacts_load(artifacts):
    arts, issues = artifacts
    for name in ("borrowers", "loans", "collateral", "cashflows",
                 "portfolio_map", "pipeline"):
        assert name in arts and not arts[name].empty
    assert not [i for i in issues if i["severity"] == "error"]


# =========================================================================
# 2. Consolidation produces one canonical frame per reporting date
# =========================================================================


def test_one_frame_per_reporting_date(consolidated):
    frames, _, _ = consolidated
    assert sorted(frames) == REPORTING_DATES


def test_artifacts_contribute_expected_fields(consolidated):
    frames, _, _ = consolidated
    f = frames["2024-03-31"]
    # loan/borrower/collateral/cashflow/portfolio/pipeline contributions:
    funded = f[f["funded_status"] == "funded"].set_index("loan_id")
    assert funded.loc["F1", "current_outstanding_balance"] == 100000.0   # loans
    assert funded.loc["F1", "geographic_region_obligor"] == "North"      # borrowers
    assert funded.loc["F1", "youngest_borrower_age"] == 55               # borrowers
    assert funded.loc["F1", "borrower_structure"] == "single"            # borrowers
    assert funded.loc["F1", "current_loan_to_value"] == 0.50             # collateral
    assert funded.loc["F2", "arrears_status"] == "31-60"                 # cashflows
    assert funded.loc["F1", "portfolio_id"] == "PF_001"                  # portfolio_map
    assert funded.loc["F2", "acquired_portfolio_id"] == "ACQ_2023_01"
    assert int(funded.loc["F1", "months_on_book"]) > 0                   # derived
    pipe = f[f["funded_status"] == "pipeline"].set_index("loan_id")
    assert pipe.loc["O1", "pipeline_stage"] == "OFFER"                   # pipeline
    assert pipe.loc["O1", "forecast_funded_balance"] == 25000.0          # derived


def test_pipeline_namespace_distinct_from_loans(consolidated):
    frames, _, _ = consolidated
    f = frames["2024-01-31"]
    funded_ids = set(f[f["funded_status"] == "funded"]["loan_id"])
    pipe_ids = set(f[f["funded_status"] == "pipeline"]["loan_id"])
    assert funded_ids.isdisjoint(pipe_ids)
    assert funded_ids == {"F1", "F2"} and pipe_ids == {"O1", "O2"}


# =========================================================================
# 3. Lineage metadata
# =========================================================================


def test_lineage_records_key_field_sources(consolidated):
    _, lineage, _ = consolidated
    for field, must_contain in [
        ("current_outstanding_balance", "loans"),
        ("current_loan_to_value", "collateral"),
        ("geographic_region_obligor", "borrowers"),
        ("arrears_status", "cashflows"),
        ("portfolio_id", "portfolio_map"),
        ("pipeline_stage", "pipeline"),
        ("forecast_funded_balance", "derived"),
    ]:
        assert field in lineage, f"missing lineage for {field}"
        assert must_contain in lineage[field].lower()


# =========================================================================
# 4. Structured issues (unmatched optional rows; no crash)
# =========================================================================


def test_unmatched_collateral_rows_flagged(consolidated):
    # collateral.csv has an orphan F9 not present in loans.
    _, _, issues = consolidated
    unmatched = [i for i in issues if i["code"] == C.UNMATCHED_ARTIFACT_ROWS]
    assert any(i.get("field") == "loan_id" for i in unmatched)
    # Orphan does not pollute the consolidated frames.
    # (handled in the per-date frames; see other tests)


def test_optional_missing_artifact_is_issue_not_crash(tmp_path):
    # A directory with only the two required artefacts -> optional ones flagged.
    d = tmp_path / "partial"
    d.mkdir()
    (d / "borrowers.csv").write_text((FIXTURE_DIR / "borrowers.csv").read_text())
    (d / "loans.csv").write_text((FIXTURE_DIR / "loans.csv").read_text())
    arts, issues = C.load_artifacts(d)
    codes = {i["code"] for i in issues}
    assert C.MISSING_OPTIONAL_ARTIFACT in codes
    assert not [i for i in issues if i["severity"] == "error"]
    frames, _, _ = C.consolidate(arts, client_id=CLIENT)
    assert sorted(frames) == REPORTING_DATES   # still consolidates, no crash


# =========================================================================
# 5. Snapshot registration + selectors
# =========================================================================


def test_three_snapshots_registered(store):
    snaps = store.list_snapshots(CLIENT, route="mi")
    assert len(snaps) == 3
    assert [h.reporting_date for h in snaps] == REPORTING_DATES


def test_selectors_work(store):
    assert store.resolve_latest(CLIENT, route="mi").reporting_date == "2024-03-31"
    assert store.resolve_as_of(CLIENT, "2024-02-15",
                               route="mi").reporting_date == "2024-01-31"
    rng = store.resolve_range(CLIENT, "2024-01-01", "2024-02-29", route="mi")
    assert [h.reporting_date for h in rng] == ["2024-01-31", "2024-02-29"]


def test_registered_frames_preserve_fields(store):
    latest = store.resolve_latest(CLIENT, route="mi")
    loans = store.load_loans(latest.snapshot_id)
    for col in ("loan_id", "funded_status", "current_outstanding_balance",
                "geographic_region_obligor", "portfolio_id",
                "internal_risk_grade", "ifrs9_stage", "pd_bucket",
                "arrears_status", "current_loan_to_value"):
        assert col in loans.columns


# =========================================================================
# 6. Runtime proof through run_mi_query (consolidated snapshots)
# =========================================================================


def test_total_funded_latest(semantics, store):
    res = run_mi_query(_spec(state="total_funded"), semantics=semantics, store=store)
    assert res.ok and res.row_count == 3
    assert float(res.data["current_outstanding_balance"].sum()) == 620000.0


def test_total_pipeline_latest(semantics, store):
    res = run_mi_query(_spec(state="total_pipeline"), semantics=semantics, store=store)
    assert res.ok and res.row_count == 1


def test_total_forecast_funded_latest(semantics, store):
    res = run_mi_query(_spec(state="total_forecast_funded"), semantics=semantics,
                       store=store)
    assert res.ok
    assert res.metadata["forecast_funded_total"] == pytest.approx(645000.0)


def test_funded_compare(semantics, store):
    res = run_mi_query(_spec(state="total_funded", temporal_mode="compare",
                             baseline_date="2024-01-31", current_date="2024-03-31"),
                       semantics=semantics, store=store)
    r = res.data.iloc[0]
    assert r["baseline_balance"] == 300000.0 and r["current_balance"] == 620000.0
    assert r["new_count"] == 1 and r["retained_count"] == 2


def test_funded_trend(semantics, store):
    res = run_mi_query(_spec(state="total_funded", temporal_mode="trend",
                             start_date="2024-01-01", end_date="2024-12-31"),
                       semantics=semantics, store=store)
    assert list(res.data["balance"]) == [300000.0, 620000.0, 620000.0]


def test_pipeline_trend(semantics, store):
    res = run_mi_query(_spec(state="total_pipeline", temporal_mode="trend",
                             start_date="2024-01-01", end_date="2024-12-31"),
                       semantics=semantics, store=store)
    assert list(res.data["balance"]) == [90000.0, 90000.0, 50000.0]


def test_forecast_funded_trend(semantics, store):
    res = run_mi_query(_spec(state="total_forecast_funded", temporal_mode="trend",
                             start_date="2024-01-01", end_date="2024-12-31"),
                       semantics=semantics, store=store)
    assert list(res.data["balance"]) == pytest.approx([335000.0, 663000.0, 645000.0])


def test_funded_by_portfolio(semantics, store, risk_cfg):
    res = run_mi_query(_spec(risk_monitor="concentration", state="total_funded",
                             dimension="portfolio_id"), semantics=semantics,
                       store=store, risk_config=risk_cfg)
    by = res.data.set_index("portfolio_id")
    assert by.loc["PF_001", "balance_sum"] == 400000.0
    assert by.loc["PF_002", "balance_sum"] == 220000.0


def test_funded_by_region(semantics, store, risk_cfg):
    res = run_mi_query(_spec(risk_monitor="concentration", state="total_funded",
                             dimension="geographic_region_obligor"),
                       semantics=semantics, store=store, risk_config=risk_cfg)
    by = res.data.set_index("geographic_region_obligor")
    assert by.loc["North", "balance_sum"] == 400000.0
    assert by.loc["South", "balance_sum"] == 220000.0


def test_pipeline_by_stage(semantics, store, risk_cfg):
    res = run_mi_query(_spec(risk_monitor="concentration", state="total_pipeline",
                             dimension="pipeline_stage", as_of_date="2024-02-29"),
                       semantics=semantics, store=store, risk_config=risk_cfg)
    by = res.data.set_index("pipeline_stage")
    assert set(by.index) == {"OFFER", "APPLICATION"}


def test_forecast_funded_by_region(semantics, store, risk_cfg):
    res = run_mi_query(_spec(risk_monitor="concentration",
                             state="total_forecast_funded",
                             dimension="geographic_region_obligor"),
                       semantics=semantics, store=store, risk_config=risk_cfg)
    by = res.data.set_index("geographic_region_obligor")
    # North = F1(100k)+F3(300k)+O1 forecast(25k) = 425k ; South = F2(220k).
    assert by.loc["North", "balance_sum"] == pytest.approx(425000.0)
    assert by.loc["South", "balance_sum"] == pytest.approx(220000.0)


def test_concentration_warning(semantics, store, risk_cfg):
    res = run_mi_query(_spec(risk_monitor="concentration", state="total_funded",
                             dimension="geographic_region_obligor"),
                       semantics=semantics, store=store, risk_config=risk_cfg)
    by = res.data.set_index("geographic_region_obligor")
    assert by.loc["North", "status"] == "red"      # 400k/620k ~ 64.5%


def test_risk_grade_migration(semantics, store, risk_cfg):
    res = run_mi_query(_spec(risk_monitor="migration",
                             dimension="internal_risk_grade",
                             baseline_date="2024-01-31", current_date="2024-03-31"),
                       semantics=semantics, store=store, risk_config=risk_cfg)
    deter = res.data[(res.data["from_value"] == "B") & (res.data["to_value"] == "C")]
    assert not deter.empty and deter.iloc[0]["movement_type"] == "deteriorated"


def test_ifrs9_migration(semantics, store, risk_cfg):
    res = run_mi_query(_spec(risk_monitor="migration", dimension="ifrs9_stage",
                             baseline_date="2024-01-31", current_date="2024-03-31"),
                       semantics=semantics, store=store, risk_config=risk_cfg)
    deter = res.data[(res.data["from_value"] == "Stage 1")
                     & (res.data["to_value"] == "Stage 2")]
    assert not deter.empty and deter.iloc[0]["movement_type"] == "deteriorated"


def test_pd_bucket_migration(semantics, store, risk_cfg):
    res = run_mi_query(_spec(risk_monitor="migration", dimension="pd_bucket",
                             baseline_date="2024-01-31", current_date="2024-03-31"),
                       semantics=semantics, store=store, risk_config=risk_cfg)
    assert "deteriorated" in set(res.data["movement_type"])


# =========================================================================
# 7. Backward compatibility — flat single-CSV
# =========================================================================


def test_flat_single_csv_still_works(semantics):
    spec = MIQuerySpec(intent="chart", chart_type="bar",
                       metric="current_outstanding_balance",
                       dimension="collateral_geography", aggregation="sum")
    res = run_mi_query(spec, semantics=semantics, data=str(FLAT_CSV))
    assert res.mode == "flat" and res.ok and res.row_count == 3


# =========================================================================
# 8. Guards
# =========================================================================


def test_no_forbidden_imports():
    # Scan the consolidation helper module (the new non-test code). The test
    # file itself names these tokens in its assertions, so it is excluded.
    text = HELPER_PATH.read_text(encoding="utf-8")
    banned = ("import streamlit", "import azure", "from azure",
              "from analytics ", "from analytics.")
    for token in banned:
        assert token not in text, f"helper contains {token!r}"
    for line in text.splitlines():
        s = line.strip()
        assert s != "import analytics" and not s.startswith("import analytics."), s


def test_no_regulatory_or_annex2_files_modified():
    import subprocess
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
    bad_prefixes = ("config/regime/", "config/delivery/", "engine/gate_",
                    "engine/delivery_xml_agent/", "engine/projection_agent/")
    bad_substr = ("annex2", "annex_2", "annex12", "_xsd", ".xsd")
    for path in changed:
        low = path.lower()
        assert not any(low.startswith(p) for p in bad_prefixes), path
        assert not any(s in low for s in bad_substr), path
