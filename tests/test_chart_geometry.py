#!/usr/bin/env python3
"""tests/test_chart_geometry.py

Charts use the canvas they are given.

A left margin expressed as a FRACTION of the figure scales with the figure and
the thing it has to clear does not. "£800.0MM" is about seven tenths of an inch
beside a 5.8in panel and beside a 12.25in full-width chart alike — but 0.145 of
the figure reserves 0.84in on the first and 1.78in on the second. That is where
the empty left-hand band on Funded Stock and Funded Balance Movement came from:
not a chart drawn too small, a gutter sized for a figure three times narrower.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

pytest.importorskip("matplotlib")

from mi_agent_pptx import render as R  # noqa: E402

#: The width a slide-width chart panel actually gets.
FULL_WIDTH_IN = 12.25
#: A half-width panel, the other common case.
PANEL_WIDTH_IN = 5.8

TICKS = ["£0", "£40.0MM", "£80.0MM"]


def test_a_wide_chart_does_not_reserve_a_wide_gutter():
    """Catches: 1.78in of empty left-hand band on a full-width chart.

    The label needs the same inches either way, so the FRACTION must shrink as
    the figure grows. The old constant did the opposite of nothing — it grew.
    """
    wide = R.axis_left(FULL_WIDTH_IN, TICKS)
    narrow = R.axis_left(PANEL_WIDTH_IN, TICKS)
    assert wide < narrow, (wide, narrow)
    assert wide * FULL_WIDTH_IN == pytest.approx(narrow * PANEL_WIDTH_IN, abs=0.01)


def test_a_full_width_chart_spends_most_of_its_width_on_the_plot():
    """Catches: a plot area that starts a seventh of the way across the slide."""
    left = R.axis_left(FULL_WIDTH_IN, TICKS)
    assert left <= 0.08, f"left gutter is {left * FULL_WIDTH_IN:.2f}in of {FULL_WIDTH_IN}in"


def test_a_narrow_panel_keeps_the_room_its_labels_need():
    """The fix must not squeeze a small chart: the margin is a floor in inches,
    so a 5.8in panel is left as it was rather than pinched to a wide chart's
    fraction."""
    left = R.axis_left(PANEL_WIDTH_IN, TICKS)
    assert 0.10 <= left <= 0.20, left
    assert left * PANEL_WIDTH_IN >= R._text_in("£80.0MM", 9.0)


def test_a_longer_label_gets_more_room_not_less():
    short = R.axis_left(FULL_WIDTH_IN, ["£0", "£9.0MM"])
    long = R.axis_left(FULL_WIDTH_IN, ["£0", "£1,234.5MM"])
    assert long > short


def test_the_margin_never_eats_the_plot():
    """A pathological label is capped rather than allowed to take the chart."""
    left = R.axis_left(2.0, ["a" * 400])
    assert left <= 0.34


def test_a_small_chart_still_gets_a_minimum_gutter():
    """With no labels at all the axis still needs air; zero would put the plot
    against the panel edge."""
    assert R.axis_left(FULL_WIDTH_IN, []) > 0.0
    assert R.axis_left(PANEL_WIDTH_IN, []) * PANEL_WIDTH_IN >= 0.29


# --------------------------------------------------------------------------- #
# The tick samples the margin is measured from.
# --------------------------------------------------------------------------- #

def test_tick_samples_come_from_the_data_range():
    """The axis formatter runs after the axes exist, so the widest tick cannot
    be measured before the margin is chosen. The extremes of the data,
    formatted the way the axis will format them, are what it will look like."""
    samples = R._money_ticks([1_000.0, 78_400_000.0], lambda v: f"£{v/1e6:.1f}MM")
    assert "£78.4MM" in samples


def test_a_broken_formatter_never_breaks_a_chart():
    def boom(_v):
        raise ValueError("nope")

    assert R._money_ticks([1.0, 2.0], boom) == []
    assert R._money_ticks([], str) == []
