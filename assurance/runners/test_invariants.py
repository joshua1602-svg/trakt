"""Invariant tests (Phase 6 §22).

These assert governed properties that must hold regardless of the question, using
the independent oracle over the base and mutation fixtures. They are property
checks on the oracle contract AND, where a governed route exists, on the live
system. The oracle checks encode the contract the production code must meet; the
live checks confirm the primary route meets it.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "assurance"))
from oracle import oracle as O  # noqa: E402

FIX = ROOT / "assurance/fixtures/generated"


def _load(name: str) -> pd.DataFrame:
    return pd.read_csv(FIX / f"{name}.csv")


# --------------------------------------------------------------------------- #
# Oracle-contract invariants
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("dim", ["property_type", "source_portfolio_id"])
def test_count_shares_sum_to_one_including_unknown(dim):
    df = _load("unknown_categories")
    dist = O.distribution(df, dim)
    assert dist["count_share_sum"] == pytest.approx(1.0)


def test_exposure_shares_sum_to_one_when_valid():
    df = _load("three_portfolios")
    dist = O.distribution(df, "property_type")
    assert dist["exposure_share_sum"] == pytest.approx(1.0)


def test_unknown_bucket_is_never_silently_removed():
    df = _load("unknown_categories")
    dist = O.distribution(df, "property_type")
    cats = {r["category"] for r in dist["rows"]}
    assert O.UNKNOWN in cats
    # The unknown population is exactly the nulled rows.
    unknown_rows = next(r for r in dist["rows"] if r["category"] == O.UNKNOWN)
    assert unknown_rows["count"] == 8


def test_grouped_counts_equal_population():
    df = _load("three_portfolios")
    dist = O.distribution(df, "property_type")
    assert sum(r["count"] for r in dist["rows"]) == len(O.as_of_latest(df))


def test_filtered_total_never_exceeds_unfiltered():
    df = _load("three_portfolios")
    full = O.total(df).value
    direct = O.total(O.apply_scope(df, source_portfolio_type="direct")).value
    assert direct <= full


def test_disjoint_subscopes_sum_to_total():
    df = _load("three_portfolios")
    full = O.total(df).value
    parts = 0.0
    for pid in df["source_portfolio_id"].unique():
        parts += O.total(O.apply_scope(df, source_portfolio_id=pid)).value
    assert parts == pytest.approx(full)


def test_weighted_average_discloses_denominator():
    df = _load("three_portfolios")
    wa = O.weighted_average(df, "current_loan_to_value")
    assert wa.denominator is not None and wa.denominator > 0
    assert wa.value is not None


def test_zero_weight_population_does_not_fall_back_to_simple_mean():
    df = _load("zero_weights")
    wa = O.weighted_average(df, "current_loan_to_value")
    assert wa.value is None
    simple = O.simple_average(df, "current_loan_to_value").value
    assert simple is not None  # a simple mean exists, but must NOT be substituted


def test_mixed_currency_produces_no_monetary_total():
    df = _load("mixed_currency")
    t = O.total(df)
    assert t.value is None and t.monetary_suppressed is True


def test_negative_weights_excluded_from_denominator():
    df = _load("negative_weights")
    wa = O.weighted_average(df, "current_loan_to_value")
    # Only strictly-positive weights count; the 5 negative-weight rows are excluded.
    positive = (pd.to_numeric(O.as_of_latest(df)["current_outstanding_balance"],
                              errors="coerce") > 0).sum()
    assert wa.population == positive


def test_concentration_ranking_is_deterministic():
    df = _load("three_portfolios")
    a = O.concentration_top_n(df, "geographic_region_obligor", 5)
    b = O.concentration_top_n(df.sample(frac=1, random_state=1), "geographic_region_obligor", 5)
    assert a["top_share"] == pytest.approx(b["top_share"])


def test_row_ordering_does_not_change_totals():
    df = _load("three_portfolios")
    a = O.total(df).value
    b = O.total(df.iloc[::-1]).value
    assert a == pytest.approx(b)


def test_single_loan_concentration_is_one():
    df = _load("one_loan")
    s = O.single_name_top(df, 1)
    assert s["top_share"] == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# Live-system invariant: repeated execution is byte-identical (determinism)
# --------------------------------------------------------------------------- #
def test_repeated_execution_is_identical(monkeypatch):
    monkeypatch.setenv("TRAKT_RUNTIME_MODE", "test")
    monkeypatch.setenv("MI_AGENT_PLATFORM_CANONICAL", str(FIX / "three_portfolios.csv"))
    monkeypatch.delenv("MI_AGENT_ONBOARDING_OUTPUT_ROOT", raising=False)
    monkeypatch.setenv("MI_AGENT_CLIENT_ID", "client_001")
    monkeypatch.setenv("MI_AGENT_LLM_ENABLED", "0")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    from mi_agent_api import data_source, datasets
    from trakt_core.context import ExecutionContext
    from mi_agent_api.mi_service import MiQueryRequest, execute_governed_mi_query

    def run():
        data_source.reset_cache()
        datasets._CLIENT_CURRENCY_CACHE.clear()
        ctx = ExecutionContext.for_internal("client_001")
        res = execute_governed_mi_query(
            MiQueryRequest(question="What is the total outstanding balance?"), ctx)
        art = (res.result or {}).get("artifacts")[0]
        return (art.get("reconciliation") or {}).get("total_balance")

    assert run() == run()
