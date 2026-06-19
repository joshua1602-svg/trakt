"""Phase 7 — MIQuerySpec v2 + interpretation contract tests.

Proves the single governed contract is additively expanded (backward
compatible), validates all Phase 0-6D MI capabilities, rejects invalid/ambiguous
specs with structured issues, and that normalised v2 specs remain runtime
compatible with run_mi_query. No external LLM calls.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from mi_agent.mi_query_spec import MIQuerySpec
from mi_agent.mi_query_spec_v2_examples import EXAMPLES, INVALID_EXAMPLES
from mi_agent.mi_query_validator import load_mi_semantics, validate_mi_query
from mi_agent.mi_runtime import run_mi_query
from mi_agent.mi_spec_validation import (
    AMBIGUOUS_DIMENSION,
    INVALID_ENUM_VALUE,
    INVALID_RISK_MONITOR_SPEC,
    INVALID_ROUTE_FOR_STATE,
    INVALID_TEMPORAL_MODE_FOR_ROUTE,
    MISSING_SNAPSHOT_CLIENT_ID,
    RISK_MONITOR_NOT_ENABLED,
    TEMPORAL_SELECTOR_INCOMPLETE,
    VIRTUAL_FIELD_NOT_AVAILABLE_IN_FLAT_MODE,
    validate_query_spec,
)
from mi_agent.risk_monitor import load_risk_monitor_config
from snapshot.adapters import LocalFsSnapshotStore
from snapshot.model import SnapshotHeader

REPO_ROOT = Path(__file__).resolve().parents[1]
SEMANTICS_PATH = REPO_ROOT / "mi_agent" / "mi_semantics_field_registry.yaml"


@pytest.fixture(scope="module")
def semantics():
    return load_mi_semantics(SEMANTICS_PATH)


# =========================================================================
# Backward compatibility
# =========================================================================


def test_v1_spec_unchanged_defaults():
    spec = MIQuerySpec(intent="chart", chart_type="bar",
                       metric="current_outstanding_balance",
                       dimension="collateral_geography", aggregation="sum")
    assert spec.route_id == "mi"            # additive default
    assert spec.execution_mode is None
    assert spec.effective_execution_mode() == "flat"
    # normalized() is idempotent for a flat v1 spec (no risk/dimension change).
    norm = spec.normalized()
    assert norm.dimension == "collateral_geography"
    assert norm.risk_monitor is None and norm.execution_mode == "flat"


def test_v1_dict_roundtrip_ignores_nothing():
    d = {"intent": "chart", "chart_type": "bar",
         "metric": "current_loan_to_value", "dimension": "collateral_geography",
         "aggregation": "weighted_avg"}
    spec = MIQuerySpec.from_dict(d)
    out = spec.to_dict()
    for k, v in d.items():
        assert out[k] == v


def test_v1_spec_still_validates_with_v1_validator(semantics):
    spec = MIQuerySpec.from_dict(EXAMPLES["flat_ltv_by_region"])
    vr = validate_mi_query(spec, semantics)
    assert vr.ok, vr.errors


def test_flat_v1_query_still_executes(semantics):
    df = pd.DataFrame({
        "loan_identifier": ["L1", "L2", "L3", "L4"],
        "current_loan_to_value": [0.3, 0.5, 0.4, 0.6],
        "collateral_geography": ["N", "S", "N", "S"],
        "current_outstanding_balance": [100, 200, 300, 400],
    })
    spec = MIQuerySpec.from_dict(EXAMPLES["flat_ltv_by_region"])
    res = run_mi_query(spec, semantics=semantics, data=df)
    assert res.mode == "flat" and res.ok


# =========================================================================
# Examples validate / invalid examples are rejected
# =========================================================================


@pytest.mark.parametrize("name", sorted(EXAMPLES))
def test_valid_examples_validate(name, semantics):
    spec = MIQuerySpec.from_dict(EXAMPLES[name])
    result = validate_query_spec(spec)
    assert result.ok, (name, result.codes())


@pytest.mark.parametrize("name", sorted(INVALID_EXAMPLES))
def test_invalid_examples_rejected(name):
    spec = MIQuerySpec.from_dict(INVALID_EXAMPLES[name])
    result = validate_query_spec(spec)
    assert not result.ok, name


def test_examples_roundtrip_deterministic():
    # Specs are plain data: from_dict -> to_dict -> from_dict is stable.
    for name, d in EXAMPLES.items():
        s1 = MIQuerySpec.from_dict(d)
        s2 = MIQuerySpec.from_dict(s1.to_dict())
        assert s1.to_dict() == s2.to_dict(), name


# =========================================================================
# Targeted validation rules
# =========================================================================


def test_state_spec_validates():
    r = validate_query_spec(MIQuerySpec.from_dict(EXAMPLES["total_funded_latest"]))
    assert r.ok


def test_temporal_compare_validates():
    r = validate_query_spec(MIQuerySpec.from_dict(EXAMPLES["funded_compare"]))
    assert r.ok


def test_temporal_trend_validates():
    r = validate_query_spec(MIQuerySpec.from_dict(EXAMPLES["funded_trend"]))
    assert r.ok


def test_risk_migration_validates():
    r = validate_query_spec(MIQuerySpec.from_dict(EXAMPLES["risk_grade_migration"]))
    assert r.ok


def test_concentration_warning_validates():
    r = validate_query_spec(
        MIQuerySpec.from_dict(EXAMPLES["concentration_warning_by_region"]))
    assert r.ok


def test_quantile_bucket_spec_validates():
    r = validate_query_spec(
        MIQuerySpec.from_dict(EXAMPLES["funded_by_balance_quantile"]))
    assert r.ok
    assert r.normalized.bucket_strategy == "quantile"


def test_mna_route_rejection():
    r = validate_query_spec(MIQuerySpec.from_dict(INVALID_EXAMPLES["mna_pipeline_state"]))
    assert INVALID_ROUTE_FOR_STATE in r.codes()


def test_regulatory_route_rejection():
    r = validate_query_spec(MIQuerySpec.from_dict(INVALID_EXAMPLES["regulatory_state"]))
    assert INVALID_ROUTE_FOR_STATE in r.codes()


def test_missing_client_issue():
    r = validate_query_spec(MIQuerySpec.from_dict(INVALID_EXAMPLES["state_missing_client"]))
    assert MISSING_SNAPSHOT_CLIENT_ID in r.codes()


def test_invalid_temporal_selector_issue():
    r = validate_query_spec(MIQuerySpec.from_dict(INVALID_EXAMPLES["compare_missing_dates"]))
    assert TEMPORAL_SELECTOR_INCOMPLETE in r.codes()


def test_invalid_risk_monitor_issue():
    r = validate_query_spec(MIQuerySpec.from_dict(INVALID_EXAMPLES["risk_without_mode"]))
    assert INVALID_RISK_MONITOR_SPEC in r.codes()


def test_invalid_enum_issue():
    r = validate_query_spec(MIQuerySpec.from_dict(INVALID_EXAMPLES["invalid_state_enum"]))
    assert INVALID_ENUM_VALUE in r.codes()


def test_ambiguous_dimension_issue():
    r = validate_query_spec(
        MIQuerySpec.from_dict(INVALID_EXAMPLES["ambiguous_stage_dimension"]))
    assert AMBIGUOUS_DIMENSION in r.codes()


def test_mna_temporal_rejected():
    spec = MIQuerySpec(route_id="mna", state="total_funded", temporal_mode="trend",
                       start_date="2024-01-01", end_date="2024-12-31",
                       snapshot_client_id="c1")
    r = validate_query_spec(spec)
    assert INVALID_TEMPORAL_MODE_FOR_ROUTE in r.codes()


def test_mna_risk_rejected():
    spec = MIQuerySpec(route_id="mna", risk_monitor_mode="concentration",
                       concentration_dimension="geographic_region_obligor",
                       snapshot_client_id="c1")
    r = validate_query_spec(spec)
    assert RISK_MONITOR_NOT_ENABLED in r.codes()


def test_flat_virtual_field_rejected(semantics):
    spec = MIQuerySpec(intent="table", metric="current_outstanding_balance",
                       dimension="portfolio_id", aggregation="sum")
    r = validate_query_spec(spec, semantics=semantics,
                            available_columns={"current_outstanding_balance",
                                               "collateral_geography"})
    assert VIRTUAL_FIELD_NOT_AVAILABLE_IN_FLAT_MODE in r.codes()


# =========================================================================
# Normalisation -> runtime compatibility
# =========================================================================


def test_normalize_maps_risk_mode_to_canonical():
    spec = MIQuerySpec.from_dict(EXAMPLES["risk_grade_migration"]).normalized()
    assert spec.risk_monitor == "migration"
    assert spec.dimension == "internal_risk_grade"


def test_normalized_v2_risk_spec_runs_through_runtime(semantics, tmp_path):
    store = LocalFsSnapshotStore(root=tmp_path / "s")

    def frame(grade_f2):
        return pd.DataFrame([
            {"loan_identifier": "F1", "funded_status": "funded",
             "current_outstanding_balance": 100.0, "internal_risk_grade": "A"},
            {"loan_identifier": "F2", "funded_status": "funded",
             "current_outstanding_balance": 200.0, "internal_risk_grade": grade_f2},
        ])
    for i, (rd, g) in enumerate([("2024-01-31", "B"), ("2024-03-31", "C")]):
        store.register_snapshot(
            SnapshotHeader(client_id="clientA", route="mi", reporting_date=rd,
                           source_file_id=f"sha256:{i}", cadence="monthly",
                           upload_timestamp=f"{rd}T09:00:00"), frame(g))
    spec = MIQuerySpec.from_dict(EXAMPLES["risk_grade_migration"]).normalized()
    res = run_mi_query(spec, semantics=semantics, store=store,
                       risk_config=load_risk_monitor_config())
    assert res.mode == "risk" and res.ok
    assert "deteriorated" in set(res.data["movement_type"])


# =========================================================================
# Parser / LLM boundary guard
# =========================================================================


def test_no_llm_or_forbidden_imports_in_spec_modules():
    files = ["mi_query_spec.py", "mi_spec_validation.py",
             "mi_query_spec_v2_examples.py"]
    banned = ("import openai", "import anthropic", "import streamlit",
              "import azure", "from azure", "from analytics ", "from analytics.")
    for fn in files:
        text = (REPO_ROOT / "mi_agent" / fn).read_text(encoding="utf-8")
        for token in banned:
            assert token not in text, f"{fn} contains {token!r}"
