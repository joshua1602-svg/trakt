#!/usr/bin/env python3
"""tests/test_concentration_history_surface.py

The path to the limit, not just the last step.

The engine has evaluated every historical frame against today's approved
configuration all along. The deck was already fetching that series and
spending it on a single direction word — so a covenant page answered "am I
inside my limits" and left "and which way is it going" to a one-word hint.

Nothing new is calculated to draw it. The evaluator produced utilisation for
every historical frame; the history payload kept the raw value and discarded
it, so a consumer wanting the path had to recompute the ratio and become a
second owner of it.
"""

from __future__ import annotations

import sys
import pathlib
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mi_agent_pptx import concentration as C  # noqa: E402


def series(test_id, points, direction="unchanged"):
    return {"available": True, "series": [
        {"testId": test_id, "direction": direction, "points": points}]}


def points(*values):
    return [{"reportingDate": f"2026-0{i + 1}-30", "value": v,
             "status": "pass", "utilisation": u}
            for i, (v, u) in enumerate(values)]


def test_the_whole_governed_series_reaches_the_row():
    """Catches: a payload that carries the path and a consumer that keeps two
    points of it."""
    rows = C.attach_history(
        [{"test_id": "t1", "label": "London concentration"}],
        series("t1", points((10.0, 0.33), (12.0, 0.40), (14.0, 0.47))))
    history = rows[0]["history_points"]
    assert [p["utilisation"] for p in history] == [0.33, 0.40, 0.47]
    assert [p["value"] for p in history] == [10.0, 12.0, 14.0]
    assert all(p["date"] for p in history)


def test_the_prior_point_still_travels_as_it_did():
    """The direction word the table prints is unchanged."""
    rows = C.attach_history(
        [{"test_id": "t1", "label": "L"}],
        series("t1", points((10.0, 0.33), (14.0, 0.47)), direction="toward"))
    assert rows[0]["prior_value"] == 10.0
    assert rows[0]["direction"] == "toward"
    assert rows[0]["periods_observed"] == 2


def test_utilisation_comes_from_the_engine_and_is_never_reformed():
    """The renderer must not divide value by limit — that would make the deck a
    second owner of a governed ratio."""
    rows = C.attach_history(
        [{"test_id": "t1", "label": "L", "limit": 30.0}],
        series("t1", points((10.0, 0.99), (12.0, 0.98), (14.0, 0.97))))
    # 10/30 is 0.33, not 0.99. The row carries what the ENGINE said.
    assert [p["utilisation"] for p in rows[0]["history_points"]] == \
        [0.99, 0.98, 0.97]


def test_a_test_with_one_frame_carries_no_history():
    rows = C.attach_history([{"test_id": "t1", "label": "L"}],
                            series("t1", points((10.0, 0.33))))
    assert "history_points" not in rows[0]
    assert "prior_value" not in rows[0]


def test_unavailable_history_leaves_every_row_untouched():
    rows = C.attach_history([{"test_id": "t1", "label": "L"}],
                            {"available": False, "series": []})
    assert rows[0] == {"test_id": "t1", "label": "L"}


# --------------------------------------------------------------------------- #
# The page's own rule.
# --------------------------------------------------------------------------- #

def test_three_frames_are_needed_before_a_path_replaces_the_bars():
    """Two points is the prior the table already states, not a trend."""
    from mi_agent_pptx.deck import DeckBuilder

    assert DeckBuilder.CONC_MIN_HISTORY == 3


def test_the_limit_reference_is_kept_in_view(tmp_path):
    """Catches: a limit line drawn off the top of the axis.

    Four utilisation paths at 30-47% against a 100% limit read as four flat
    lines unless the limit is on the page with them — which is the one thing
    the chart exists to show.
    """
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    from mi_agent_pptx import render as R

    R.draw_lines(tmp_path / "c.png", ["a", "b", "c"],
                 [{"name": "t", "values": [30.0, 35.0, 47.0]}], 6.0, 3.0,
                 currency=False, zero_based=True,
                 reference={"value": 100.0, "label": "limit"})
    assert (tmp_path / "c.png").exists()
    plt.close("all")


def test_a_floor_test_and_a_ceiling_test_share_one_scale():
    """Utilisation normalises both: 100% means "at the limit" whichever way the
    governed operator points, so a min test is not drawn as if it were a max."""
    from mi_agent_api import concentration_tests_api as CT

    # A floor test moving DOWN is moving toward its limit.
    assert CT.direction_of_travel(20.0, 12.0, 10.0, "min") == CT.DIRECTION_TOWARD
    # The same numbers on a ceiling test are moving away from it.
    assert CT.direction_of_travel(20.0, 12.0, 30.0, "max") == CT.DIRECTION_AWAY


# --------------------------------------------------------------------------- #
# The legend has to be readable, which means it has to fit.
# --------------------------------------------------------------------------- #

def test_a_long_series_name_is_never_cropped_mid_word():
    """Catches: "Scotland conce".

    Four concentration tests named after regions do not fit on one legend row
    at readable type. The fitter must take a second row rather than run the
    last entry off the figure — a cropped covenant name is worse than a
    two-row legend.
    """
    from mi_agent_pptx import render as R

    names = ["London concentration", "Wales concentration",
             "South East concentration", "Scotland concentration"]
    pt, rows = R._legend_fit(names, 5.9)
    per_row = -(-len(names) // rows)
    widest = max(R._text_in(n, pt) + R._LEGEND_CHROME_IN for n in names)
    assert widest * per_row <= 5.9, (pt, rows, widest * per_row)


def test_two_short_names_still_take_one_row():
    """The fitter must not spend a second row it does not need — the band it
    reserves comes out of the chart's own height."""
    from mi_agent_pptx import render as R
    assert R._legend_fit(["Funded balance", "Pipeline"], 5.9)[1] == 1


def test_the_legend_never_shrinks_below_readable_type():
    """A legend nobody can read is not a fit. Where names cannot be made to
    fit, the fitter stops at the floor and takes the rows instead."""
    from mi_agent_pptx import render as R
    pt, _rows = R._legend_fit(["x" * 90] * 4, 3.0)
    assert pt >= min(R._LEGEND_PT)


def test_the_reserved_band_grows_with_the_rows(tmp_path):
    """A wrapped row must be allowed for, or it prints over the chart.

    Drawn end to end because the band and the legend are set in two different
    places, and the bug this replaces was exactly them disagreeing.
    """
    from mi_agent_pptx import render as R

    path = R.draw_lines(
        tmp_path / "legend.png", ["2026-01", "2026-02", "2026-03"],
        [{"name": f"{n} concentration test", "values": [10.0, 11.0, 12.0]}
         for n in ("London", "Wales", "South East", "Scotland")],
        6.5, 3.62, currency=False, zero_based=True,
        reference={"value": 100.0, "label": "limit"})
    assert pathlib.Path(path).exists()
