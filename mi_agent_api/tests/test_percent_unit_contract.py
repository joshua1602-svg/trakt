"""The governed percent unit contract for funded MI.

ONE convention, stated once: every percent metric the funded evolution and
cohort routes emit is a FRACTION (0.395 == 39.5%), whatever the tape stores, so
the display rule is "multiply by 100 exactly once" everywhere.

Two defects made that untrue, and both are pinned here.

* ``funded_evolution`` emitted ``wa_ltv`` / ``wa_interest_rate`` through the raw
  weighted average while ``funded_cohort_progression`` emitted the same keys
  through ``_pct_fraction``. On a tape storing LTV in points the first rendered
  3945% and the second 39.5% — the same metric, the same module, two
  conventions.
* the storage scale was resolved from the SUBSET being aggregated rather than
  the column. A cohort of genuinely low-LTV loans in a points column has a
  median below the 1.5 threshold, so it was read as a fraction and rendered
  1.23 as 123% — while the row beneath it, on the same tape, scaled correctly.

The canonical representation is established by ``funded_prep._derive_ltv``,
which normalises LTV to a 0..1 ratio. These tests assert against that, and
against a points-stored tape, so neither convention can silently drift.
"""

from __future__ import annotations

import pandas as pd
import pytest

from mi_agent_api.cohorts import _weighted_avg_pct, cohort_analysis
from mi_agent_api.evolution import _column, _pct_fraction, _weighted_avg
from mi_agent_api.funded_prep import prepare_funded_mi_dataset

#: The React formatter's contract-aware path, mirrored: a fraction times 100,
#: once. (frontend/mi-agent-ui/src/lib/utils.ts::formatValue with
#: scale="percent_fraction".)
def render(value):
    return None if value is None else round(value * 100.0, 1)


def frame(ltv, rate=None, balance=None):
    n = len(ltv)
    data = {
        "current_loan_to_value": ltv,
        "current_outstanding_balance": balance or [100_000] * n,
    }
    if rate is not None:
        data["current_interest_rate"] = rate
    return pd.DataFrame(data)


class TestCanonicalRepresentation:
    """What prep guarantees, and what it must display as."""

    def test_prep_normalises_ltv_to_a_ratio(self):
        raw = pd.DataFrame({
            "current_outstanding_balance": [112_620.0, 90_000.0],
            "current_valuation_amount": [280_000.0, 250_000.0],
        })
        prepared, _ = prepare_funded_mi_dataset(raw)
        ltv = prepared["current_loan_to_value"].dropna()
        assert (ltv < 1.5).all(), "canonical LTV is a 0..1 ratio, not points"

    def test_a_canonical_ratio_displays_as_percentage_points(self):
        df = frame([0.40, 0.39, 0.395, 0.40])
        assert render(_pct_fraction(df, "current_loan_to_value")) == 39.6

    def test_a_points_tape_displays_identically(self):
        """The convention of the tape must not change the number on screen."""
        assert render(_pct_fraction(frame([40.0, 39.0, 39.5, 40.0]),
                                    "current_loan_to_value")) == 39.6


class TestOneContractAcrossSurfaces:
    """Evolution, cohort progression and the cohort table agree — the defect
    was that they did not."""

    @pytest.mark.parametrize("ltv, label", [
        ([0.40, 0.39, 0.395, 0.40], "fraction tape"),
        ([40.0, 39.0, 39.5, 40.0], "points tape"),
    ])
    def test_every_surface_emits_the_same_fraction(self, ltv, label):
        df = frame(ltv)
        evolution = _pct_fraction(df, "current_loan_to_value")
        progression = _pct_fraction(df, "current_loan_to_value",
                                    _column(df, "current_loan_to_value"))
        table = _weighted_avg_pct(df["current_loan_to_value"],
                                  df["current_outstanding_balance"])
        assert render(evolution) == render(progression) == render(table) == 39.6, label

    def test_the_raw_weighted_average_is_not_what_routes_emit(self):
        """Guards the specific regression: emitting _weighted_avg on a points
        tape renders 3945%, which is what funded_evolution used to do."""
        df = frame([40.0, 39.0, 39.5, 40.0])
        assert render(_weighted_avg(df, "current_loan_to_value")) == 3962.5
        assert render(_pct_fraction(df, "current_loan_to_value")) == 39.6


class TestNoDoubleScaling:
    def test_a_fraction_is_never_divided_again(self):
        df = frame([0.395] * 4)
        assert _pct_fraction(df, "current_loan_to_value") == pytest.approx(0.395)

    def test_repeated_normalisation_is_idempotent(self):
        df = frame([40.0] * 4)
        once = _pct_fraction(df, "current_loan_to_value")
        twice = _pct_fraction(frame([once] * 4), "current_loan_to_value")
        assert once == pytest.approx(twice), "normalising twice must not shrink"

    def test_a_rate_in_points_is_scaled_once_not_twice(self):
        df = frame([0.40] * 4, rate=[9.55] * 4)
        assert render(_pct_fraction(df, "current_interest_rate")) == 9.6


class TestScaleIsAPropertyOfTheColumn:
    """The subset-heuristic defect: one cohort scaled differently from its
    neighbour in the same table."""

    def test_a_low_value_cohort_is_scaled_like_its_column(self):
        column = pd.Series([40.0, 39.0, 1.2, 1.3, 1.1, 1.4, 41.0, 38.0])
        cohort = frame([1.2, 1.3, 1.1, 1.4])
        assert render(_pct_fraction(cohort, "current_loan_to_value")) == 125.0
        assert render(_pct_fraction(cohort, "current_loan_to_value", column)) == 1.2

    def test_the_cohort_table_scales_every_row_from_the_column(self):
        column = pd.Series([40.0, 39.0, 1.2, 1.3])
        low = pd.Series([1.2, 1.3])
        weights = pd.Series([100_000, 100_000])
        assert render(_weighted_avg_pct(low, weights)) == 125.0
        assert render(_weighted_avg_pct(low, weights, column)) == 1.2

    def test_neighbouring_cohort_rows_share_one_scale(self):
        df = pd.DataFrame({
            "current_loan_to_value": [40.0, 41.0, 1.2, 1.3],
            "current_outstanding_balance": [100_000] * 4,
            "ltv_bucket": ["40-50%", "40-50%", "<20%", "<20%"],
        })
        rows = cohort_analysis(df, client_id="c", dimension="ltv")["cohorts"]
        by_label = {r["cohort"]: r["waLtv"] for r in rows}
        assert render(by_label["<20%"]) == 1.2
        assert render(by_label["40-50%"]) == 40.5


class TestWeightingReconciles:
    def test_weighted_ltv_reconciles_to_loan_balances(self):
        # 0.30 on 300k and 0.60 on 100k -> (90k + 60k) / 400k = 0.375
        df = frame([0.30, 0.60], balance=[300_000, 100_000])
        assert _pct_fraction(df, "current_loan_to_value") == pytest.approx(0.375)

    def test_a_zero_balance_book_reports_nothing_not_zero(self):
        df = frame([0.40, 0.40], balance=[0, 0])
        assert _pct_fraction(df, "current_loan_to_value") is None


class TestScopesAndPeriods:
    def test_the_latest_period_cannot_use_a_different_scale(self):
        """Each period is scaled from its own column, so a book that changes
        convention between snapshots still displays consistently."""
        periods = {"2025-10": [0.40] * 4, "2025-11": [40.0] * 4, "2026-06": [0.40] * 4}
        rendered = {p: render(_pct_fraction(frame(v), "current_loan_to_value"))
                    for p, v in periods.items()}
        assert set(rendered.values()) == {40.0}, rendered

    @pytest.mark.parametrize("scope", ["direct", "acquired", "total"])
    def test_every_scope_uses_the_same_contract(self, scope):
        df = frame([0.40] * 4)
        df["source_portfolio_type"] = ["direct", "direct", "acquired", "acquired"]
        sliced = df if scope == "total" else df[df["source_portfolio_type"] == scope]
        assert render(_pct_fraction(sliced, "current_loan_to_value",
                                    _column(df, "current_loan_to_value"))) == 40.0
