#!/usr/bin/env python3
"""tests/test_p1a_single_filter.py — P1A safe single-filter analytics.

One supported measure + one supported predicate + a straightforward aggregation.

Every positive case reconciles the agent's answer to a figure computed directly
from the fixture frame with pandas. The MI Agent is never its own oracle: the
expected value is derived from the dataframe, not from a previous run.

Canonical unit convention asserted explicitly (P1A brief §2): LTV and interest
rate are stored in PERCENTAGE POINTS in the governed frame (LTV 42.7 means
42.7%), so "below 75%" is the predicate ``< 75`` and never ``< 0.75``.

Negative cases prove the boundary still holds: an unmatched value, an unknown
field, an unsupported operator and a malformed threshold must refuse without
substituting a broader population.
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

from mi_agent import execution_receipt as er
from mi_agent.mi_agent_workflow import run_mi_agent_query
from mi_agent.mi_query_validator import load_mi_semantics

_SEMANTICS_PATH = _REPO_ROOT / "mi_agent" / "mi_semantics_field_registry.yaml"
_BALANCE = "current_outstanding_balance"
_LTV = "current_loan_to_value"
_AGE = "youngest_borrower_age"
_RATE = "current_interest_rate"
_GEO = "collateral_geography"


@pytest.fixture(scope="module")
def semantics():
    return load_mi_semantics(_SEMANTICS_PATH)


@pytest.fixture(scope="module")
def frame():
    """A canonical-shaped funded tape. Percentages in POINTS, as the governed
    executor stores them."""
    rng = np.random.default_rng(20260818)
    n = 900
    return pd.DataFrame({
        _BALANCE: rng.uniform(40_000, 500_000, n).round(2),
        _LTV: rng.uniform(15.0, 88.0, n).round(3),
        _RATE: rng.uniform(3.5, 9.5, n).round(3),
        _AGE: rng.integers(58, 96, n),
        "current_valuation_amount": rng.uniform(150_000, 1_200_000, n).round(2),
        _GEO: rng.choice(["London", "South East", "Wales", "Scotland",
                          "North West"], n),
        "account_status": rng.choice(["Active", "Redeemed"], n, p=[.94, .06]),
        "ltv_bucket": rng.choice(["40-50%", "50-60%"], n),
        "data_cut_off_date": ["2026-06-30"] * n,
    })


def _ask(question, frame, semantics):
    return run_mi_agent_query(question, frame, semantics)


def _summary(result) -> dict:
    return result.get("execution_receipt") or {}


def _kpi_values(result) -> dict:
    """``{field: rawValue}`` for the executed result row."""
    qr = result.get("query_result")
    rows = (qr.to_dict().get("data") or []) if qr is not None else []
    return dict(rows[0]) if rows else {}


def _wavg(df, column, mask):
    x, w = df.loc[mask, column], df.loc[mask, _BALANCE]
    keep = x.notna() & w.notna()
    return float((x[keep] * w[keep]).sum() / w[keep].sum())


# =========================================================================== #
# Canonical unit convention — asserted, not assumed
# =========================================================================== #
def test_percent_fields_are_stored_in_points_not_fractions(frame):
    assert frame[_LTV].median() > 1.5
    assert frame[_RATE].median() > 1.5


def test_a_percent_threshold_is_compared_in_points(frame, semantics):
    """"below 75%" must select LTV < 75, never LTV < 0.75 (which would select
    nothing on a points-stored column)."""
    result = _ask("how much balance is below 75% ltv", frame, semantics)
    assert result["ok"], result.get("error")
    expected = int((frame[_LTV] < 75).sum())
    assert _summary(result)["population"] == expected
    assert expected > 0


# =========================================================================== #
# Positive bank — every case reconciles to independently computed truth
# =========================================================================== #
def _geo_mask(df, region):
    return df[_GEO] == region


POSITIVE = [
    # --- geography -------------------------------------------------------- #
    ("What is the average LTV in London?", "wavg_ltv", lambda d: _geo_mask(d, "London")),
    ("What is the average LTV for London?", "wavg_ltv", lambda d: _geo_mask(d, "London")),
    ("What is the weighted average LTV in the South East?", "wavg_ltv",
     lambda d: _geo_mask(d, "South East")),
    ("How much exposure do I have in London?", "balance", lambda d: _geo_mask(d, "London")),
    ("What is the balance in Scotland?", "balance", lambda d: _geo_mask(d, "Scotland")),
    ("How many loans are in the South East?", "count", lambda d: _geo_mask(d, "South East")),
    ("What is the average interest rate for loans in Wales?", "wavg_rate",
     lambda d: _geo_mask(d, "Wales")),
    ("What is the total balance in the North West?", "balance",
     lambda d: _geo_mask(d, "North West")),
    # --- borrower age ----------------------------------------------------- #
    ("What is my exposure to borrowers over 85?", "balance", lambda d: d[_AGE] > 85),
    ("How many borrowers are aged above 80?", "count", lambda d: d[_AGE] > 80),
    ("What is the balance for borrowers younger than 70?", "balance",
     lambda d: d[_AGE] < 70),
    ("What is the average LTV for borrowers over 75?", "wavg_ltv", lambda d: d[_AGE] > 75),
    ("How many loans have a borrower aged at least 90?", "count", lambda d: d[_AGE] >= 90),
    # --- LTV -------------------------------------------------------------- #
    ("How much balance is above 60% LTV?", "balance", lambda d: d[_LTV] > 60),
    ("How many loans have LTV over 70%?", "count", lambda d: d[_LTV] > 70),
    ("What is the average interest rate for loans below 50% LTV?", "wavg_rate",
     lambda d: d[_LTV] < 50),
    ("What is the outstanding balance for loans with LTV between 40% and 60%?",
     "balance", lambda d: (d[_LTV] >= 40) & (d[_LTV] <= 60)),
    ("How many loans have LTV between 50 and 70?", "count",
     lambda d: (d[_LTV] >= 50) & (d[_LTV] <= 70)),
    # --- balance ---------------------------------------------------------- #
    ("How many loans have a balance over £250,000?", "count",
     lambda d: d[_BALANCE] > 250_000),
    ("What is the average LTV for loans above £300k?", "wavg_ltv",
     lambda d: d[_BALANCE] > 300_000),
]


@pytest.mark.parametrize("question,kind,mask_fn", POSITIVE,
                         ids=[q[:52] for q, _, _ in POSITIVE])
def test_single_filter_reconciles_to_truth(question, kind, mask_fn, frame, semantics):
    result = _ask(question, frame, semantics)
    assert result["ok"], f"{question!r} -> {result.get('error')}"

    mask = mask_fn(frame)
    expected_n = int(mask.sum())
    assert expected_n > 0, "fixture must exercise a non-empty population"

    summary = _summary(result)
    # 1. The executed population is the FILTERED population, never the book.
    assert summary["population"] == expected_n, summary.get("receipt")
    assert summary["population"] < len(frame)
    assert summary["narrowed"] is True

    # 2. The value reconciles to pandas.
    values = _kpi_values(result)
    if kind == "count":
        assert int(values["loan_count"]) == expected_n
    elif kind == "balance":
        expected = float(frame.loc[mask, _BALANCE].sum())
        got = float(values[f"{_BALANCE}_sum"])
        assert got == pytest.approx(expected, rel=1e-9)
    elif kind == "wavg_ltv":
        expected = _wavg(frame, _LTV, mask)
        got = float(values[f"{_LTV}_weighted_avg"])
        assert got == pytest.approx(expected, rel=1e-9)
    elif kind == "wavg_rate":
        expected = _wavg(frame, _RATE, mask)
        got = float(values[f"{_RATE}_weighted_avg"])
        assert got == pytest.approx(expected, rel=1e-9)

    # 3. The P0 guard sees the predicate as APPLIED, on execution evidence.
    assert result["semantic_guard"]["verdict"] == er.VERDICT_OK
    material = [f for f in summary["facets"]
                if f["kind"] in (er.KIND_GEOGRAPHIC_SCOPE, er.KIND_THRESHOLD)]
    assert material, f"no material facet detected for {question!r}"
    assert all(f["status"] == er.APPLIED for f in material)

    # 4. The receipt states the predicate and the filtered population.
    receipt = summary["receipt"]
    assert receipt.startswith("Calculated:")
    assert f"{expected_n:,} loans" in receipt
    assert summary["filtersApplied"], receipt


# =========================================================================== #
# Share / proportion — two populations, denominator auditable
# =========================================================================== #
SHARE = [
    ("What percentage of the book is below 75% LTV?", lambda d: d[_LTV] < 75),
    ("What proportion of the book is above 60% LTV?", lambda d: d[_LTV] > 60),
    ("What percentage of the book is with borrowers over 85?", lambda d: d[_AGE] > 85),
]


@pytest.mark.parametrize("question,mask_fn", SHARE, ids=[q[:46] for q, _ in SHARE])
def test_share_uses_two_populations_and_states_the_denominator(
        question, mask_fn, frame, semantics):
    result = _ask(question, frame, semantics)
    assert result["ok"], f"{question!r} -> {result.get('error')}"
    mask = mask_fn(frame)
    expected = float(frame.loc[mask, _BALANCE].sum() / frame[_BALANCE].sum() * 100)

    values = _kpi_values(result)
    got = float(values[f"{_BALANCE}_share_pct"])
    assert got == pytest.approx(expected, rel=1e-9)
    # NOT answered with a weighted-average LTV, the pre-P1A failure.
    assert f"{_LTV}_weighted_avg" not in values

    summary = _summary(result)
    assert summary["population"] == int(mask.sum())
    assert summary["populationTotal"] == len(frame)
    receipt = summary["receipt"]
    assert "Share of" in receipt
    assert f"of {len(frame):,}" in receipt, receipt


# =========================================================================== #
# Negative bank — the boundary must not move
# =========================================================================== #
def test_a_geography_with_no_matching_rows_refuses(frame, semantics):
    """"in Atlantis" parses as a scope but selects nothing. Reporting £0 would
    be a confident answer about a place the book has no exposure to."""
    result = _ask("what is the average ltv in atlantis", frame, semantics)
    assert result["ok"] is False
    assert "match that filter" in result["error"]
    assert "whole-book" in result["error"]


def test_an_unavailable_field_still_refuses_by_name(frame, semantics):
    result = _ask("what is the average ltv for fixed rate loans by product type",
                  frame, semantics)
    assert result["ok"] is False


def test_an_unresolvable_threshold_field_is_surfaced_not_applied(frame, semantics):
    """A threshold whose FIELD is not governed must never bind to another
    column that happens to be numeric."""
    result = _ask("how many loans have a risk score above 700", frame, semantics)
    assert result["ok"] is False
    # The governed unmapped refusal is the right outcome here: no field matches,
    # so nothing was guessed and no substitute population was returned.
    assert "nothing was guessed" in result["error"] or "not available" in result["error"]
    assert result.get("query_result") is None


def test_a_stress_query_is_still_refused_after_p1a(frame, semantics):
    """P1A must not accidentally widen the surface to scenario questions."""
    result = _ask("what would a 10% house price fall do to my weighted average ltv?",
                  frame, semantics)
    assert result["ok"] is False
    assert "unstressed" in result["error"]


def test_a_ranking_query_is_still_refused_after_p1a(frame, semantics):
    result = _ask("which region has grown the most in the last month?",
                  frame, semantics)
    assert result["ok"] is False


def test_a_multi_measure_query_shares_one_filtered_population_after_p1e(
        frame, semantics):
    """P1A's filter and P1E's measure set compose over ONE population.

    This question used to refuse because a spec carried one measure. It now
    answers both — and the point P1A cares about is that the London filter
    applies to BOTH figures, not to whichever measure happened to be parsed
    first.
    """
    result = _ask("what is the average ltv and the average borrower age in london?",
                  frame, semantics)
    assert result["ok"] is True, result.get("error")
    metadata = getattr(result.get("query_result"), "metadata", None) or {}
    columns = [m["column"] for m in (metadata.get("measures_executed") or [])]
    assert any("loan_to_value" in c for c in columns)
    assert any("age" in c for c in columns)
    # One population, stated once: the receipt cannot claim two.
    receipt = _summary(result).get("receipt") or ""
    assert receipt.count("London") == 1


def test_the_guard_is_not_weakened_a_dropped_filter_still_refuses(frame, semantics):
    """The distinction P1A must preserve: capability expanded means MORE
    EXECUTION EVIDENCE exists, not that the guard accepts less. With the
    predicate stripped from the spec after parsing, the facet must NOT be
    APPLIED and the answer must refuse — proving APPLIED is earned from
    execution, not granted by detection."""
    import mi_agent.mi_agent_workflow as wf

    original = wf.execute_mi_query

    def _strip_filters(spec, df, semantics_, **kw):
        spec.filters = {}          # simulate a filter lost after validation
        return original(spec, df, semantics_, **kw)

    wf.execute_mi_query = _strip_filters
    try:
        result = _ask("what is the average ltv in london", frame, semantics)
    finally:
        wf.execute_mi_query = original

    assert result["ok"] is False
    assert "London" in result["error"]
    assert "not substituted" in result["error"]


def test_whole_book_questions_are_unaffected_by_p1a(frame, semantics):
    for question in ("what is the total balance?",
                     "what is the weighted average ltv?",
                     "show me balance by region"):
        result = _ask(question, frame, semantics)
        assert result["ok"] is True, f"{question!r} -> {result.get('error')}"
        assert _summary(result)["population"] == len(frame)


# =========================================================================== #
# Known P1A limitations — OUT of scope, and they fail SAFE
# =========================================================================== #
# Both phrasings below put a measure word inside the PREDICATE ("... with LTV
# below 50%", "... is with borrowers over 85"). The parser resolves the METRIC
# from the whole question, so the predicate's measure wins the metric slot and
# the question is answered about the wrong measure — or no metric resolves at
# all. That is a metric-side resolution issue, not a filter one: fixing it means
# masking predicate spans before metric resolution, which is a change to the
# parser's measure resolution and deliberately not attempted inside P1A.
#
# What matters for launch is that they fail SAFE. The P0 guard catches the
# measure substitution and refuses; no wrong number is returned.
def test_known_limitation_measure_inside_predicate_fails_safe(frame, semantics):
    result = _ask("What is the balance for loans with LTV below 50%?",
                  frame, semantics)
    assert result["ok"] is False
    assert "asked about balance" in result["error"]
    assert "not returned" in result["error"]


def test_a_share_question_is_never_answered_as_a_whole_book_summary(frame, semantics):
    """"What share of the book is ..." names no measure, because "the book" IS
    the basis. It previously reached the whole-book portfolio summary — the
    exact opposite of the question asked."""
    result = _ask("What share of the book is with borrowers over 85?",
                  frame, semantics)
    assert result["ok"] is True, result.get("error")
    values = _kpi_values(result)
    expected = float(frame.loc[frame[_AGE] > 85, _BALANCE].sum()
                     / frame[_BALANCE].sum() * 100)
    assert float(values[f"{_BALANCE}_share_pct"]) == pytest.approx(expected, rel=1e-9)
    assert _summary(result)["populationTotal"] == len(frame)
