"""Metamorphic tests (Phase 6 §23).

Each test transforms the input and asserts the governed relationship between the
answers, using the independent oracle. Metamorphic testing catches whole classes
of calculation error without needing a hand-computed expected value.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "assurance"))
from oracle import oracle as O  # noqa: E402

FIX = ROOT / "assurance/fixtures/generated"


def _load(name: str) -> pd.DataFrame:
    return pd.read_csv(FIX / f"{name}.csv")


def test_row_permutation_preserves_answers():
    df = _load("three_portfolios")
    shuffled = df.sample(frac=1, random_state=7).reset_index(drop=True)
    assert O.total(df).value == pytest.approx(O.total(shuffled).value)
    assert (O.weighted_average(df, "current_loan_to_value").value
            == pytest.approx(O.weighted_average(shuffled, "current_loan_to_value").value))
    a = O.concentration_top_n(df, "property_type", 2)["top_share"]
    b = O.concentration_top_n(shuffled, "property_type", 2)["top_share"]
    assert a == pytest.approx(b)


def test_dataset_duplication_doubles_counts_preserves_ratios():
    df = _load("three_portfolios")
    dup = df.copy()
    dup["loan_identifier"] = dup["loan_identifier"].astype(str) + "_DUP"
    doubled = pd.concat([df, dup], ignore_index=True)

    assert O.count(doubled).value == pytest.approx(2 * O.count(df).value)
    assert O.total(doubled).value == pytest.approx(2 * O.total(df).value)
    # Averages, shares and concentration percentages are invariant under duplication.
    assert (O.weighted_average(doubled, "current_loan_to_value").value
            == pytest.approx(O.weighted_average(df, "current_loan_to_value").value))
    d1 = O.distribution(df, "property_type")
    d2 = O.distribution(doubled, "property_type")
    m1 = {r["category"]: r["exposure_share"] for r in d1["rows"]}
    m2 = {r["category"]: r["exposure_share"] for r in d2["rows"]}
    for cat in m1:
        assert m1[cat] == pytest.approx(m2[cat])
    c1 = O.concentration_top_n(df, "property_type", 2)["top_share"]
    c2 = O.concentration_top_n(doubled, "property_type", 2)["top_share"]
    assert c1 == pytest.approx(c2)


def test_scope_narrowing_matches_direct_calculation():
    df = _load("three_portfolios")
    for pid in df["source_portfolio_id"].unique():
        via_scope = O.total(O.apply_scope(df, source_portfolio_id=pid)).value
        direct = pd.to_numeric(df[df["source_portfolio_id"] == pid][O.BALANCE],
                               errors="coerce").dropna().sum()
        assert via_scope == pytest.approx(direct)


def test_category_relabelling_moves_population_to_unknown():
    df = _load("three_portfolios")
    base = O.distribution(df, "property_type")
    base_total = sum(r["count"] for r in base["rows"])
    # Relabel one known category's rows to null (-> unknown).
    victim = base["rows"][0]["category"]
    mutated = df.copy()
    mutated.loc[mutated["property_type"] == victim, "property_type"] = None
    after = O.distribution(mutated, "property_type")
    after_total = sum(r["count"] for r in after["rows"])
    # Total population unchanged; unknown rises; the known category falls to 0/absent.
    assert after_total == base_total
    unknown_after = next((r["count"] for r in after["rows"] if r["category"] == O.UNKNOWN), 0)
    unknown_before = next((r["count"] for r in base["rows"] if r["category"] == O.UNKNOWN), 0)
    victim_count = next(r["count"] for r in base["rows"] if r["category"] == victim)
    assert unknown_after == unknown_before + victim_count


def test_currency_mutation_suppresses_monetary_keeps_counts():
    single = _load("three_portfolios")
    mixed = _load("mixed_currency")
    # Monetary total: valid on single-currency, suppressed on mixed.
    assert O.total(single).value is not None
    assert O.total(mixed).value is None and O.total(mixed).monetary_suppressed
    # Count is unaffected by the currency mutation.
    assert O.count(single).value == O.count(mixed).value


def test_portfolio_order_reversal_reverses_difference_sign():
    df = _load("three_portfolios")
    a = O.total(O.apply_scope(df, source_portfolio_id="alp_origination")).value
    b = O.total(O.apply_scope(df, source_portfolio_id="alp_acquired")).value
    assert (a - b) == pytest.approx(-(b - a))
