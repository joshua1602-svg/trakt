"""
MI Agent regression tests: the business concept "Region" must resolve to a
true NUTS region field (obligor / collateral) and must NEVER resolve to
`geographic_region_classification` (which is the NUTS classification YEAR).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from mi_agent.llm_query_parser import EXPLICIT_DIMENSION_TERMS, _deterministic_parse
from mi_agent.mi_agent_workflow import run_mi_agent_query
from mi_agent.mi_query_spec import MIQuerySpec
from mi_agent.mi_query_validator import load_mi_semantics, validate_mi_query

REPO_ROOT = Path(__file__).resolve().parents[2]
SEMANTICS_PATH = REPO_ROOT / "mi_agent" / "mi_semantics_field_registry.yaml"


@pytest.fixture(scope="module")
def semantics():
    return load_mi_semantics(SEMANTICS_PATH)


@pytest.fixture
def df():
    regions = ["North", "South", "East"]
    return pd.DataFrame([{
        "current_outstanding_balance": 100_000 + i * 1000,
        "geographic_region_obligor": regions[i % 3],
        "geographic_region_collateral": regions[(i + 1) % 3],
    } for i in range(30)])


def test_classification_not_in_mi_registry(semantics):
    # The classification (year) field must not be a curated MI dimension.
    assert "geographic_region_classification" not in semantics["fields"]


def test_region_term_maps_to_obligor_not_classification():
    assert EXPLICIT_DIMENSION_TERMS["region"] == "geographic_region_obligor"
    assert EXPLICIT_DIMENSION_TERMS["region"] != "geographic_region_classification"
    # No explicit-term value points at the classification field.
    assert "geographic_region_classification" not in EXPLICIT_DIMENSION_TERMS.values()


_REGION_FIELDS = {"collateral_geography", "geographic_region_collateral",
                  "geographic_region_obligor"}


def test_balance_by_region_uses_a_true_region_field(semantics):
    spec, _ = _deterministic_parse("Show balance by region", semantics)
    # A true region field (readable display preferred), never the year field.
    assert spec.dimension in _REGION_FIELDS
    assert spec.dimension != "geographic_region_classification"


def test_balance_by_region_prefers_readable_display_when_present(semantics):
    d = pd.DataFrame([{
        "current_outstanding_balance": 100,
        "collateral_geography": "North",            # readable display field
        "geographic_region_obligor": "TLG31",       # NUTS code
    } for _ in range(5)])
    res = run_mi_agent_query("Show balance by region", d, semantics)
    assert res["ok"], res.get("error")
    assert res["spec"]["dimension"] == "collateral_geography"
    assert "collateral_geography" in res["query_result"].data.columns


def test_balance_by_region_falls_back_to_nuts_when_no_display(df, semantics):
    # df has obligor + collateral NUTS fields but no readable display field.
    res = run_mi_agent_query("Show balance by region", df, semantics)
    assert res["ok"], res.get("error")
    assert res["spec"]["dimension"] in {"geographic_region_collateral",
                                        "geographic_region_obligor"}
    assert res["spec"]["dimension"] != "geographic_region_classification"


def test_classification_rejected_as_mi_dimension(semantics):
    spec = MIQuerySpec(intent="chart", chart_type="bar",
                       metric="current_outstanding_balance",
                       dimension="geographic_region_classification",
                       aggregation="sum")
    vr = validate_mi_query(spec, semantics)
    assert not vr.ok
    assert any("geographic_region_classification" in e for e in vr.errors)


def test_region_fails_clearly_when_no_true_region_field(semantics):
    # If no region field is present, the region query must fail validation
    # cleanly (never fall back to a classification year).
    df_no_region = pd.DataFrame({"current_outstanding_balance": [1, 2, 3]})
    res = run_mi_agent_query("Show balance by region", df_no_region, semantics)
    assert res["ok"] is False
    # classification year is never used as the region dimension
    assert (res["spec"] or {}).get("dimension") != "geographic_region_classification"


def test_itl3_drilldown_fields_available_in_mi(semantics):
    # Granular ITL3 codes are MI drilldown dimensions, distinct from the readable
    # Region (collateral_geography) and never GBZZZ / classification year.
    for key in ("geographic_region_collateral_itl3", "geographic_region_obligor_itl3"):
        assert key in semantics["fields"], f"{key} missing from MI registry"
        assert semantics["fields"][key]["role"] == "dimension"
    # the readable Region is collateral_geography, not an ITL3 field
    assert semantics["fields"]["collateral_geography"]["business_name"] == "Region"


def test_mi_region_never_resolves_to_gbzzz_or_classification(semantics):
    from mi_agent.llm_query_parser import EXPLICIT_DIMENSION_TERMS, _preferred_region
    assert "geographic_region_classification" not in EXPLICIT_DIMENSION_TERMS.values()
    # preferred region is a readable/region field, never the classification year
    pref = _preferred_region(semantics, available_columns={
        "collateral_geography", "geographic_region_obligor"})
    assert pref == "collateral_geography"
    assert pref != "geographic_region_classification"
