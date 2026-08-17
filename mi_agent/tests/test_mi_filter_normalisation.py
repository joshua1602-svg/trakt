#!/usr/bin/env python3
"""tests/test_mi_filter_normalisation.py

``MIQuerySpec.filters`` is canonically ``{field_key: condition}``, but a JSON
producer — the LLM parser above all — naturally writes a LIST of predicate
objects, or a singular ``filter`` key. Before normalisation those shapes reached
the dataclass unconverted:

  * a list made ``referenced_fields()`` raise
    ``TypeError: unhashable type: 'dict'``, which the service caught as
    "The MI Agent could not interpret this question" — a crash presented as a
    refusal;
  * a singular ``filter`` key was dropped as unknown, so a correctly-identified
    predicate produced a confident WHOLE-BOOK answer with no warning.

The payloads in ``test_captured_llm_specs_*`` are the verbatim JSON
``claude-haiku-4-5`` returned for the review question bank
(``due_diligence/MI_QUERY_AGENT_CAPABILITY_REVIEW.md`` §6.1) — the shapes that
actually broke, not invented ones.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mi_agent.llm_query_parser import parse_llm_response_to_spec
from mi_agent.mi_query_executor import _apply_filters
from mi_agent.mi_query_spec import MIQuerySpec, normalise_filters


# --------------------------------------------------------------------------- #
# Shape folding
# --------------------------------------------------------------------------- #
def test_canonical_mapping_is_unchanged():
    filters, notes = normalise_filters({"geographic_region_obligor": "London"})
    assert filters == {"geographic_region_obligor": "London"}
    assert notes == []


def test_list_of_predicates_folds_to_mapping():
    filters, notes = normalise_filters([
        {"field": "geographic_region_obligor", "operator": "equals", "value": "London"},
        {"field": "youngest_borrower_age", "operator": "greater_than", "value": 85},
    ])
    assert filters == {
        "geographic_region_obligor": "London",
        "youngest_borrower_age": {"op": "greater_than", "value": 85},
    }
    assert notes == []


def test_single_predicate_object_folds_to_mapping():
    filters, _ = normalise_filters(
        {"field": "youngest_borrower_age", "operator": "greater_than", "value": 85})
    assert filters == {"youngest_borrower_age": {"op": "greater_than", "value": 85}}


def test_equality_predicate_becomes_a_bare_value():
    """The executor matches a bare string case-insensitively — that IS what an
    equality predicate on a categorical field means."""
    filters, _ = normalise_filters(
        [{"field": "collateral_geography", "operator": "eq", "value": "South East"}])
    assert filters == {"collateral_geography": "South East"}


def test_membership_predicate_becomes_a_list():
    filters, _ = normalise_filters(
        [{"field": "collateral_geography", "operator": "in",
          "values": ["London", "South East"]}])
    assert filters == {"collateral_geography": ["London", "South East"]}


def test_min_max_bounds_fold_to_between():
    filters, _ = normalise_filters(
        [{"field": "current_loan_to_value", "min": 60, "max": 80}])
    assert filters == {"current_loan_to_value": {"op": "between", "value": [60, 80]}}


@pytest.mark.parametrize("alias", ["field", "field_key", "key", "column", "name"])
def test_field_name_aliases(alias):
    filters, _ = normalise_filters([{alias: "borrower_type", "value": "Joint"}])
    assert filters == {"borrower_type": "Joint"}


# --------------------------------------------------------------------------- #
# Nothing is silently dropped
# --------------------------------------------------------------------------- #
def test_unfoldable_predicate_is_recorded_not_dropped():
    filters, notes = normalise_filters([{"operator": "gt", "value": 5}])  # no field
    assert filters == {}
    assert len(notes) == 1 and "could not be interpreted" in notes[0]


def test_unfoldable_predicate_reaches_unavailable_filters():
    spec = MIQuerySpec.from_dict({
        "intent": "summary", "aggregation": "sum",
        "metric": "current_outstanding_balance",
        "filters": [
            {"field": "collateral_geography", "operator": "equals", "value": "London"},
            {"operator": "greater_than", "value": 85},          # no field named
        ],
    })
    assert spec.filters == {"collateral_geography": "London"}
    assert len(spec.unavailable_filters) == 1
    assert "NOT applied" in spec.unavailable_filters[0]


def test_existing_unavailable_filters_are_preserved():
    spec = MIQuerySpec.from_dict({
        "intent": "summary",
        "unavailable_filters": ["borrower_structure is not in this dataset"],
        "filters": ["not-a-predicate"],
    })
    assert spec.unavailable_filters[0] == "borrower_structure is not in this dataset"
    assert len(spec.unavailable_filters) == 2


def test_scalar_filters_value_is_reported_not_raised():
    filters, notes = normalise_filters("region = London")
    assert filters == {} and len(notes) == 1


# --------------------------------------------------------------------------- #
# referenced_fields must never raise
# --------------------------------------------------------------------------- #
def test_referenced_fields_survives_a_list_built_around_from_dict():
    """A spec constructed directly (bypassing from_dict) can still carry a raw
    list. Introspection must degrade, never raise — this is the exact call that
    produced TypeError: unhashable type: 'dict'."""
    spec = MIQuerySpec(intent="summary", metric="current_outstanding_balance")
    spec.filters = [{"field": "collateral_geography", "value": "London"}]  # type: ignore[assignment]
    assert spec.referenced_fields() == ["current_outstanding_balance",
                                        "collateral_geography"]


def test_referenced_fields_survives_a_junk_filters_container():
    spec = MIQuerySpec(intent="summary", metric="current_outstanding_balance")
    spec.filters = 42  # type: ignore[assignment]
    assert spec.referenced_fields() == ["current_outstanding_balance"]


# --------------------------------------------------------------------------- #
# Captured LLM output — the payloads that actually broke
# --------------------------------------------------------------------------- #
_CAPTURED_LONDON = """```json
{
  "intent": "chart",
  "chart_type": "bar",
  "metric": "current_loan_to_value",
  "aggregation": "weighted_avg",
  "dimensions": ["borrower_type"],
  "filters": [
    {"field": "collateral_geography", "operator": "equals", "value": "London"}
  ],
  "explanation": "Average LTV and borrower age by borrower type, London only."
}
```"""

_CAPTURED_OVER_85 = """{
  "intent": "summary",
  "chart_type": "none",
  "metric": "current_outstanding_balance",
  "aggregation": "sum",
  "filter": {
    "field": "youngest_borrower_age",
    "operator": "greater_than",
    "value": 85
  },
  "explanation": "Total exposure to loans whose youngest borrower is over 85."
}"""


def test_captured_llm_specs_london_no_longer_raises():
    spec = parse_llm_response_to_spec(_CAPTURED_LONDON)
    assert spec.filters == {"collateral_geography": "London"}
    spec.referenced_fields()  # the call that used to raise


def test_captured_llm_specs_over_85_keeps_the_predicate():
    """The singular ``filter`` key used to be discarded, so this question was
    answered from the whole book (£1.96bn against a true £31.1m)."""
    spec = parse_llm_response_to_spec(_CAPTURED_OVER_85)
    assert spec.filters == {"youngest_borrower_age": {"op": "greater_than", "value": 85}}


# --------------------------------------------------------------------------- #
# End to end: a folded predicate actually narrows the frame
# --------------------------------------------------------------------------- #
@pytest.fixture()
def frame():
    return pd.DataFrame({
        "current_outstanding_balance": [100.0, 200.0, 300.0, 400.0],
        "youngest_borrower_age": [70, 80, 86, 90],
        "collateral_geography": ["London", "South East", "London", "Wales"],
    })


@pytest.fixture()
def semantics():
    return {"fields": {
        "current_outstanding_balance": {"canonical_field": "current_outstanding_balance",
                                        "format": "currency", "role": "metric"},
        "youngest_borrower_age": {"canonical_field": "youngest_borrower_age",
                                  "format": "integer", "role": "metric"},
        "collateral_geography": {"canonical_field": "collateral_geography",
                                 "format": "string", "role": "dimension"},
    }}


def test_folded_age_predicate_narrows_the_frame(frame, semantics):
    spec = parse_llm_response_to_spec(_CAPTURED_OVER_85)
    warnings: list = []
    out = _apply_filters(frame, spec, semantics, warnings)
    assert list(out["youngest_borrower_age"]) == [86, 90]
    assert out["current_outstanding_balance"].sum() == 700.0
    assert any("kept 2/4" in w for w in warnings)


def test_folded_region_predicate_narrows_the_frame(frame, semantics):
    spec = MIQuerySpec.from_dict({
        "intent": "summary", "metric": "current_outstanding_balance",
        "aggregation": "sum",
        "filters": [{"field": "collateral_geography",
                     "operator": "equals", "value": "London"}],
    })
    warnings: list = []
    out = _apply_filters(frame, spec, semantics, warnings)
    assert len(out) == 2
    assert out["current_outstanding_balance"].sum() == 400.0


@pytest.mark.parametrize("verbose_op,expected_rows", [
    ("greater_than_or_equal_to", 3),   # >= 80
    ("less_than_or_equal_to", 2),      # <= 80
    ("not_equals", 3),                 # != 80
])
def test_verbose_operator_spellings_are_honoured(frame, semantics,
                                                 verbose_op, expected_rows):
    """An operator spelled out in English must not fall through to the ``eq``
    default, which would silently compare a threshold as an exact match."""
    spec = MIQuerySpec.from_dict({
        "intent": "summary", "metric": "current_outstanding_balance",
        "filters": [{"field": "youngest_borrower_age",
                     "operator": verbose_op, "value": 80}],
    })
    out = _apply_filters(frame, spec, semantics, [])
    assert len(out) == expected_rows
