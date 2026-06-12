"""
Tests for the v0.2.1 cleanup pass on the MI semantic layer:
    1. numeric axis roles (x / bucket)
    2. weighted_avg defaults for rate/LTV fields
    3. monetary performance fields normalised to currency / sum
    4. number_of_properties_*_date is not a date
    5. generated YAML has no anchors/aliases
    6. parser prefers mi_tier: core fields
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mi_agent.llm_query_parser import find_field, parse_user_question
from mi_agent.mi_query_validator import load_mi_semantics, validate_mi_query

REPO_ROOT = Path(__file__).resolve().parents[2]
SEMANTICS_PATH = REPO_ROOT / "mi_agent" / "mi_semantics_field_registry.yaml"


@pytest.fixture(scope="module")
def semantics():
    if not SEMANTICS_PATH.exists():
        from mi_agent import build_mi_semantics_registry as builder
        builder.write_registry(
            builder.build_registry(builder.DEFAULT_SOURCE), SEMANTICS_PATH
        )
    return load_mi_semantics(SEMANTICS_PATH)


def _entry(semantics, key):
    return semantics["fields"][key]


# --------------------------------------------------------------------------- #
# Task 1 — numeric axis / chart roles
# --------------------------------------------------------------------------- #


def test_youngest_borrower_age_allows_x_and_bucket(semantics):
    entry = _entry(semantics, "youngest_borrower_age")
    roles = set(entry["allowed_chart_roles"])
    assert "x" in roles, entry["allowed_chart_roles"]
    assert "bucket" in roles, entry["allowed_chart_roles"]
    assert entry["default_chart_role"] == "x"
    assert entry["bucket_field"] == "age_bucket"


def test_number_of_days_in_arrears_allows_x_and_bucket(semantics):
    entry = _entry(semantics, "number_of_days_in_arrears")
    roles = set(entry["allowed_chart_roles"])
    assert "x" in roles
    assert "bucket" in roles
    assert entry["default_chart_role"] == "x"
    assert entry["format"] == "integer"


def test_original_term_allows_x_and_bucket(semantics):
    entry = _entry(semantics, "original_term")
    roles = set(entry["allowed_chart_roles"])
    assert {"x", "bucket"}.issubset(roles)
    assert entry["format"] == "integer"


# --------------------------------------------------------------------------- #
# Task 2 — weighted_avg defaults for rate / LTV / percentage fields
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("field", [
    "current_loan_to_value",
    "indexed_loan_to_value",
    "original_loan_to_value",
    "current_interest_rate",
    "current_interest_rate_margin",
])
def test_rate_and_ltv_default_to_weighted_avg(semantics, field):
    entry = _entry(semantics, field)
    assert entry["format"] == "percent"
    assert "weighted_avg" in entry["allowed_aggregations"]
    assert entry["default_aggregation"] == "weighted_avg", (
        f"{field} default_aggregation should be weighted_avg "
        f"(got {entry['default_aggregation']!r})"
    )
    assert entry["weight_field"], f"{field} must have a weight_field"


# --------------------------------------------------------------------------- #
# Task 3 — monetary performance fields normalised to currency / sum
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("field", [
    "recoveries_in_period",
    "redemptions_received_in_period",
    "cumulative_prepayments",
    "principal_arrears_amount",
    "interest_arrears_amount",
    "default_amount",
    "allocated_losses",
])
def test_monetary_performance_fields_are_currency_sum(semantics, field):
    entry = _entry(semantics, field)
    assert entry["format"] == "currency", (
        f"{field} format should be currency (got {entry['format']!r})"
    )
    assert entry["default_aggregation"] == "sum"
    assert "sum" in entry["allowed_aggregations"]
    assert "avg" in entry["allowed_aggregations"]
    assert "median" in entry["allowed_aggregations"]


# --------------------------------------------------------------------------- #
# Task 4 — number_of_properties_* is NOT a date
# --------------------------------------------------------------------------- #


def test_number_of_properties_is_not_a_date(semantics):
    field = "number_of_properties_at_data_cut_off_date"
    if field not in semantics["fields"]:
        pytest.skip(f"{field} not curated in this build")
    entry = _entry(semantics, field)
    assert entry["role"] != "date", entry
    assert entry["format"] != "date", entry
    # Sensible MI defaults: integer metric with sum default.
    assert entry["format"] == "integer"
    assert entry["role"] == "metric"
    assert entry["default_aggregation"] == "sum"


# --------------------------------------------------------------------------- #
# Task 5 — generated YAML contains no anchors / aliases
# --------------------------------------------------------------------------- #


def test_generated_yaml_has_no_anchors_or_aliases():
    raw = SEMANTICS_PATH.read_text(encoding="utf-8")
    for line_no, line in enumerate(raw.splitlines(), 1):
        # YAML anchor / alias markers always appear at column-position tokens
        # like ``&id001`` or ``*id001``. Reject any occurrence.
        assert "&id" not in line, f"anchor on line {line_no}: {line!r}"
        assert "*id" not in line, f"alias on line {line_no}: {line!r}"


# --------------------------------------------------------------------------- #
# Task 6 — parser prefers core-tier fields
# --------------------------------------------------------------------------- #


def test_find_field_prefers_core_tier(semantics):
    # ``age`` keyword: the only ``age`` metric in core is youngest_borrower_age.
    key = find_field(semantics, role="metric", fmt="integer", keywords=("age",))
    assert key == "youngest_borrower_age"
    assert _entry(semantics, key)["mi_tier"] == "core"


def test_deterministic_parser_resolves_age_to_core(semantics):
    spec = parse_user_question("ltv by age by balance", SEMANTICS_PATH,
                               llm_enabled=False)
    assert spec.chart_type == "bubble"
    # x should be the core borrower-age field, not some extended look-alike
    assert spec.x == "youngest_borrower_age"
    assert _entry(semantics, spec.x)["mi_tier"] == "core"
    # spec must still validate end-to-end
    result = validate_mi_query(spec, semantics)
    assert result.ok, result.errors


# --------------------------------------------------------------------------- #
# Task 7 — metadata
# --------------------------------------------------------------------------- #


def test_metadata_version_and_cleanup_notes(semantics):
    meta = semantics["metadata"]
    assert meta["version"] == "0.2.1"
    notes = meta.get("cleanup_notes") or []
    assert any("weighted_avg" in n for n in notes)
    assert any("numeric axis" in n for n in notes)
    assert any("YAML aliases" in n for n in notes)
    # tier counts must still be present
    assert "core_field_count" in meta
    assert "extended_field_count" in meta
    assert meta["field_count"] == (
        meta["core_field_count"] + meta["extended_field_count"]
    )
