#!/usr/bin/env python3
"""tests/test_mi_predicate_extraction.py

Hardened predicate extraction (sprint Part B): currency / k-m / comma parsing,
age equality, multi-predicate AND, the le-vs-ge comparator fix, and surfacing of
predicates that could not be applied (never silently dropped).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mi_agent.mi_query_validator import load_mi_semantics
from mi_agent.llm_query_parser import _deterministic_parse, _parse_filters
from mi_agent.mi_agent_workflow import run_mi_agent_query

_SEMANTICS = _REPO_ROOT / "mi_agent" / "mi_semantics_field_registry.yaml"
_COLS = {"current_loan_to_value", "youngest_borrower_age", "current_outstanding_balance",
         "total_balance", "borrower_structure", "number_of_borrowers",
         "broker_channel", "geographic_region_obligor", "current_interest_rate"}


@pytest.fixture(scope="module")
def semantics():
    return load_mi_semantics(_SEMANTICS)


def _f(q, semantics):
    return _parse_filters(q.lower(), semantics, _COLS)


# --------------------------------------------------------------------------- #
# Currency / k-m / comma parsing
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("q,expected", [
    ("balance more than £100k", 100_000.0),
    ("balance more than £100K", 100_000.0),
    ("balance more than 100000", 100_000.0),
    ("balance above 100,000", 100_000.0),
    ("balance over £0.2m", 200_000.0),
    ("balance over £200k", 200_000.0),
    ("balance over £1.5m", 1_500_000.0),
])
def test_currency_parsing(q, expected, semantics):
    cond = _f(q, semantics)["current_outstanding_balance"]
    assert cond["op"] == "gt" and cond["value"] == expected


def test_between_currency(semantics):
    cond = _f("balance between £100k and £0.5m", semantics)["current_outstanding_balance"]
    assert cond["op"] == "between" and cond["value"] == [100_000.0, 500_000.0]


# --------------------------------------------------------------------------- #
# Age equality phrasings
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("q", ["60 year old", "aged 60", "age 60", "60-year-old",
                               "60 yo", "60 years of age"])
def test_age_equality(q, semantics):
    cond = _f(q, semantics)["youngest_borrower_age"]
    assert cond["op"] == "eq" and cond["value"] == 60.0


def test_age_or_above_is_ge_not_equality(semantics):
    cond = _f("aged 70 or above", semantics)["youngest_borrower_age"]
    assert cond["op"] == "ge" and cond["value"] == 70.0


# --------------------------------------------------------------------------- #
# Comparator words + the le/ge fix
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("q,op", [
    ("ltv more than 40%", "gt"), ("ltv over 40%", "gt"), ("ltv greater than 40%", "gt"),
    ("ltv less than 40%", "lt"), ("ltv under 40%", "lt"),
    ("ltv at least 40%", "ge"), ("ltv at most 40%", "le"),
    ("ltv no more than 40%", "le"), ("ltv no less than 40%", "ge"),
])
def test_comparators(q, op, semantics):
    cond = _f(q, semantics)["current_loan_to_value"]
    assert cond["op"] == op and cond["value"] == 40.0


# --------------------------------------------------------------------------- #
# The complex multi-predicate query — every predicate is applied
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("q,balance", [
    ("how many 60 year old borrowers, that are joint borrowers, and have LTV of "
     "more than 40% and balance more than £100K", 100_000.0),
    ("how many 60 year old joint borrowers with LTV over 40% and balance more than £200K",
     200_000.0),
])
def test_complex_borrower_query_extracts_all_predicates(q, balance, semantics):
    spec, _ = _deterministic_parse(q, semantics, available_columns=_COLS)
    fl = spec.filters
    assert fl["youngest_borrower_age"] == {"op": "eq", "value": 60.0}
    assert fl["current_loan_to_value"] == {"op": "gt", "value": 40.0}
    assert fl["current_outstanding_balance"] == {"op": "gt", "value": balance}
    assert fl["borrower_structure"] == "Joint"


def test_complex_query_executes_all_filters(semantics):
    # A real count: only loans matching ALL four predicates are counted.
    rng = np.random.default_rng(3)
    n = 400
    df = pd.DataFrame({
        "loan_identifier": [f"L{i}" for i in range(n)],
        "current_outstanding_balance": rng.uniform(50_000, 400_000, n).round(2),
        "current_loan_to_value": rng.uniform(15, 70, n).round(1),
        "youngest_borrower_age": rng.integers(55, 90, n),
        "borrower_structure": rng.choice(["Joint", "Sole"], n),
        "broker_channel": rng.choice(["Alpha", "Beta"], n),
    })
    res = run_mi_agent_query(
        "how many 60 year old borrowers, that are joint borrowers, and have LTV of "
        "more than 40% and balance more than £100K", df, semantics)
    assert res["ok"], res.get("error")
    expected = int(((df["youngest_borrower_age"] == 60)
                    & (df["current_loan_to_value"] > 40)
                    & (df["current_outstanding_balance"] > 100_000)
                    & (df["borrower_structure"] == "Joint")).sum())
    qr = res["query_result"].to_dict()
    rows = qr.get("data") or []
    # The summary count is the loan_count of the single result row.
    got = int(rows[0].get("loan_count") or rows[0].get("count") or list(rows[0].values())[0]) if rows else 0
    assert got == expected, (got, expected, rows)


# --------------------------------------------------------------------------- #
# Unavailable predicate is surfaced, not silently dropped
# --------------------------------------------------------------------------- #
def test_joint_unavailable_is_surfaced(semantics):
    # Neither borrower_structure nor number_of_borrowers present.
    cols = {"current_loan_to_value", "current_outstanding_balance", "youngest_borrower_age"}
    spec, _ = _deterministic_parse(
        "how many joint borrowers with LTV over 40%", semantics, available_columns=cols)
    assert spec.unavailable_filters, "joint predicate should be recorded as unavailable"
    assert "current_loan_to_value" in spec.filters  # the applicable predicate still applied


def test_unavailable_predicate_becomes_a_warning(semantics):
    df = pd.DataFrame({
        "current_outstanding_balance": [100_000.0, 200_000.0],
        "current_loan_to_value": [30.0, 60.0],
        "youngest_borrower_age": [60, 70],
    })
    res = run_mi_agent_query("how many joint borrowers with LTV over 40%", df, semantics)
    assert any("not applied" in w.lower() for w in res.get("warnings", []))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
