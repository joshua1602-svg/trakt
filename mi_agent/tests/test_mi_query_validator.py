"""
Smoke tests for the MI Agent v1 foundation.

These tests are intentionally tolerant of the exact canonical field names: they
DISCOVER suitable fields from the generated MI semantic registry by role /
format / keyword rather than hard-coding names, so they keep working across
different canonical registries.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mi_agent import build_mi_semantics_registry as builder
from mi_agent.llm_query_parser import (
    find_field,
    parse_user_question,
)
from mi_agent.mi_query_spec import MIQuerySpec
from mi_agent.mi_query_validator import load_mi_semantics, validate_mi_query

REPO_ROOT = Path(__file__).resolve().parents[2]
SEMANTICS_PATH = REPO_ROOT / "mi_agent" / "mi_semantics_field_registry.yaml"


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def built_registry():
    """Build the registry from the live canonical source (Part 1/2)."""
    return builder.build_registry(builder.DEFAULT_SOURCE)


@pytest.fixture(scope="module")
def semantics():
    if not SEMANTICS_PATH.exists():
        # generate it if missing so the suite is self-contained
        reg = builder.build_registry(builder.DEFAULT_SOURCE)
        builder.write_registry(reg, SEMANTICS_PATH)
    return load_mi_semantics(SEMANTICS_PATH)


# Discovery helpers -------------------------------------------------------- #


def _balance(semantics):
    return find_field(semantics, role="metric", fmt="currency",
                      keywords=("balance", "outstanding", "principal"))


def _ltv(semantics):
    return find_field(semantics, role="metric", fmt="percent",
                      keywords=("ltv", "loan_to_value"))


def _age(semantics):
    return find_field(semantics, role="metric", fmt="integer", keywords=("age",))


def _region(semantics):
    return find_field(semantics, role="dimension",
                      keywords=("region", "geograph", "country"))


def _broker(semantics):
    key = find_field(semantics, role="dimension", keywords=("broker", "channel"))
    return key


def _any_dimension(semantics, exclude=()):
    return find_field(semantics, role="dimension", exclude=exclude)


# --------------------------------------------------------------------------- #
# 1 & 2 — registry generation
# --------------------------------------------------------------------------- #


def test_build_registry_from_canonical(built_registry):
    assert "fields" in built_registry
    assert built_registry["metadata"]["field_count"] > 0
    assert len(built_registry["fields"]) == built_registry["metadata"]["field_count"]
    # every entry references a canonical field and carries source_criteria
    for key, entry in built_registry["fields"].items():
        assert entry["canonical_field"] == key
        assert entry["source_criteria"], f"{key} missing source_criteria"


def test_generated_semantics_file_has_fields(semantics):
    assert SEMANTICS_PATH.exists()
    assert len(semantics["fields"]) > 0


# --------------------------------------------------------------------------- #
# 3-6 — valid charts
# --------------------------------------------------------------------------- #


def test_bar_balance_by_region(semantics):
    metric, dimension = _balance(semantics), _region(semantics)
    assert metric and dimension
    spec = MIQuerySpec(intent="chart", chart_type="bar", metric=metric,
                       dimension=dimension, aggregation="sum")
    result = validate_mi_query(spec, semantics)
    assert result.ok, result.errors


def test_bubble_ltv_by_age_sized_by_balance(semantics):
    x, y, size = _age(semantics), _ltv(semantics), _balance(semantics)
    assert x and y and size
    spec = MIQuerySpec(intent="chart", chart_type="bubble", x=x, y=y, size=size,
                       aggregation="loan_level")
    result = validate_mi_query(spec, semantics)
    assert result.ok, result.errors


def test_heatmap_two_dimensions_and_metric(semantics):
    d1 = _region(semantics)
    d2 = _any_dimension(semantics, exclude={d1})
    metric = _ltv(semantics)
    assert d1 and d2 and metric and d1 != d2
    spec = MIQuerySpec(intent="chart", chart_type="heatmap", metric=metric,
                       dimensions=[d1, d2], aggregation="avg")
    result = validate_mi_query(spec, semantics)
    assert result.ok, result.errors


def test_treemap_balance_by_region_and_broker(semantics):
    d1 = _region(semantics)
    d2 = _broker(semantics) or _any_dimension(semantics, exclude={d1})
    metric = _balance(semantics)
    assert d1 and d2 and metric
    spec = MIQuerySpec(intent="chart", chart_type="treemap", metric=metric,
                       hierarchy=[d1, d2], aggregation="sum", top_n=10)
    result = validate_mi_query(spec, semantics)
    assert result.ok, result.errors


# --------------------------------------------------------------------------- #
# 7-9 — rejections
# --------------------------------------------------------------------------- #


def test_reject_unknown_semantic_field(semantics):
    spec = MIQuerySpec(intent="chart", chart_type="bar",
                       metric="this_field_does_not_exist",
                       dimension=_region(semantics), aggregation="sum")
    result = validate_mi_query(spec, semantics)
    assert not result.ok
    assert any("Unknown semantic field" in e for e in result.errors)


def test_reject_invalid_aggregation_sum_on_ltv(semantics):
    ltv, dimension = _ltv(semantics), _region(semantics)
    spec = MIQuerySpec(intent="chart", chart_type="bar", metric=ltv,
                       dimension=dimension, aggregation="sum")
    result = validate_mi_query(spec, semantics)
    assert not result.ok
    assert any("not allowed for metric" in e for e in result.errors)


def test_reject_bubble_size_is_dimension(semantics):
    x, y = _age(semantics), _ltv(semantics)
    dim_as_size = _region(semantics)
    spec = MIQuerySpec(intent="chart", chart_type="bubble", x=x, y=y,
                       size=dim_as_size, aggregation="loan_level")
    result = validate_mi_query(spec, semantics)
    assert not result.ok


# --------------------------------------------------------------------------- #
# 10 — deterministic parser
# --------------------------------------------------------------------------- #


def test_deterministic_parser_ltv_by_age_by_balance(semantics):
    spec = parse_user_question("ltv by age by balance", SEMANTICS_PATH,
                               llm_enabled=False)
    assert isinstance(spec, MIQuerySpec)
    assert spec.chart_type == "bubble"
    assert spec.x and spec.y and spec.size
    result = validate_mi_query(spec, semantics)
    assert result.ok, result.errors
