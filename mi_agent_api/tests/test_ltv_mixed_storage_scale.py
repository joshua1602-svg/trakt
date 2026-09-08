"""LTV normalisation when one reporting frame carries two storage conventions.

Reported defect: a vintage's weighted-average LTV read 41.8% and 39.5% in the
periods where the frame held only the originated book, then 0.4% in the period
where an acquired book joined it — a hundredfold drop, on the same loans.

Cause: ``_to_ratio`` made a SINGLE column-level decision. The acquired book
stored LTV in percentage points and outnumbered the originations, so the column
median said "points", and the division by 100 was applied to the originations
too — which were already ratios.

The column rule is right for a coherent column and is kept for every
single-source tape. These tests pin that, pin the mixed case, and pin the one
case a naive row-wise rule would break: a ratio column with a genuinely
rolled-up tail above 150% LTV, which must NOT be treated as a mixture.
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

from mi_agent_api.funded_prep import _to_ratio  # noqa: E402


def _ratios(n: int, mean: float = 0.418, seed: int = 3) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(np.clip(rng.normal(mean, 0.05, n), 0.05, 0.95))


def _points(n: int, mean: float = 25.0, seed: int = 4) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(np.clip(rng.normal(mean, 6.0, n), 5.0, 60.0))


# --------------------------------------------------------------------------- #
# Coherent columns keep the column rule, unchanged
# --------------------------------------------------------------------------- #
def test_a_coherent_ratio_column_is_left_alone():
    col = _ratios(73)
    out, note = _to_ratio(col)
    assert note is None
    pd.testing.assert_series_equal(out, col)


def test_a_coherent_points_column_is_divided_once():
    col = _points(700)
    out, note = _to_ratio(col)
    assert note is None
    pd.testing.assert_series_equal(out, col / 100.0)


def test_a_rolled_up_tail_is_not_mistaken_for_a_mixture():
    """Equity release rolls up: LTVs above 100%, even above 150%, are real.
    Those rows are the ratio column's own tail, not a second convention, and a
    row-wise rule that divided them by 100 would report 170% LTV as 1.7%."""
    col = pd.Series(list(_ratios(200, mean=0.6)) + [1.6, 1.75, 1.9, 2.1])
    out, note = _to_ratio(col)
    assert note is None, "a rolled-up tail must not be reported as mixed scales"
    pd.testing.assert_series_equal(out, col)
    assert out.max() > 1.5, "the rolled-up loans keep their real LTV"


# --------------------------------------------------------------------------- #
# The reported defect
# --------------------------------------------------------------------------- #
@pytest.fixture
def mixed_frame():
    """700 acquired loans in points, 73 originated loans already ratios."""
    return pd.concat([_points(700), _ratios(73)], ignore_index=True), slice(700, None)


def test_a_mixed_frame_is_normalised_per_row_and_reported(mixed_frame):
    col, _ = mixed_frame
    out, note = _to_ratio(col)
    assert note and "mixed LTV storage scales" in note
    assert "700 row(s) in percentage points" in note
    assert "73 already a ratio" in note


def test_the_originated_vintage_keeps_its_real_ltv(mixed_frame):
    """The defect in one assertion: this vintage read 0.4% instead of ~41.8%."""
    col, vintage = mixed_frame
    out, _ = _to_ratio(col)
    wa = float(out.iloc[vintage].mean())
    assert 0.35 < wa < 0.50, f"expected ~0.42 (42% LTV), got {wa}"


def test_the_acquired_book_is_also_correct(mixed_frame):
    col, _ = mixed_frame
    out, _ = _to_ratio(col)
    wa = float(out.iloc[:700].mean())
    assert 0.15 < wa < 0.35, f"expected ~0.25 (25% LTV), got {wa}"


def test_every_row_lands_in_a_plausible_ltv_band(mixed_frame):
    col, _ = mixed_frame
    out, _ = _to_ratio(col)
    assert out.max() <= 1.5 and out.min() > 0.0


def test_the_column_level_rule_would_have_been_wrong(mixed_frame):
    """Guards the fix against being reverted to the median-only rule."""
    col, vintage = mixed_frame
    column_rule = col / 100.0 if float(col.median()) > 1.5 else col
    assert float(column_rule.iloc[vintage].mean()) < 0.01, (
        "the old rule put the originated vintage two orders of magnitude too low")


# --------------------------------------------------------------------------- #
# Degenerate inputs never raise
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("values", [[], [np.nan, np.nan], [0.42], [42.0]])
def test_degenerate_columns_are_safe(values):
    out, _note = _to_ratio(pd.Series(values, dtype="float64"))
    assert len(out) == len(values)
