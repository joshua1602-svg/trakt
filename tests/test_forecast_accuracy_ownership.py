#!/usr/bin/env python3
"""tests/test_forecast_accuracy_ownership.py — the forecaster's track record.

Forecast error, bias and the widest miss are what a funder uses to judge whether
to believe the forecast on the next page. They were computed in the PPTX layer,
which put an analytical result in a renderer and left React unable to show it.

They are now ``evolution.forecast_evolution``'s. The presentation layer reads the
structured finding and chooses the English for it — the engine decides what is
true, the slide decides what to call it.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _accuracy(pairs):
    """Run the engine's summariser over ``[(period, prior, actual)]``."""
    from mi_agent_api.evolution import _forecast_accuracy
    errors = [(p, (a - f) / abs(f) * 100.0) for p, f, a in pairs if f]
    return _forecast_accuracy(errors)


# --------------------------------------------------------------------------- #
# The measure.
# --------------------------------------------------------------------------- #

def test_bias_is_signed_and_error_is_absolute():
    """Catches: a mean absolute error reported as though it showed direction.

    Two periods 5% high and 5% low have a 5% typical error and NO lean. Reporting
    only the absolute error hides that; reporting only the signed one hides how
    far off it was.
    """
    out = _accuracy([("2026-01", 100.0, 105.0), ("2026-02", 100.0, 95.0)])
    assert out["available"]
    assert out["errorPct"] == pytest.approx(5.0)
    assert out["biasPct"] == pytest.approx(0.0)


def test_a_persistently_high_forecast_shows_a_negative_bias():
    out = _accuracy([("2026-01", 100.0, 96.0), ("2026-02", 100.0, 94.0)])
    assert out["biasPct"] < 0, out
    assert out["errorPct"] == pytest.approx(5.0)


def test_one_observation_is_not_a_track_record():
    """Catches: a single close forecast reported as a measured mean error."""
    out = _accuracy([("2026-01", 100.0, 100.4)])
    assert out["available"] is False
    assert out["observations"] == 1
    assert "at least 2" in out["reason"]


def test_the_widest_miss_is_named():
    out = _accuracy([("2026-01", 100.0, 101.0), ("2026-02", 100.0, 110.0),
                     ("2026-03", 100.0, 102.0)])
    assert out["worstPeriod"] == "2026-02"
    assert out["worstPct"] == pytest.approx(10.0)


# --------------------------------------------------------------------------- #
# The wording, and the boundary between the two.
# --------------------------------------------------------------------------- #

def test_an_immaterial_lean_is_not_given_a_direction():
    """A fifth of a percent is not a lean. Naming one manufactures a finding."""
    from mi_agent_pptx.forecast_accuracy import Accuracy
    assert Accuracy(observations=4, bias_pct=0.2, error_pct=3.0,
                    available=True).lean == ""
    assert Accuracy(observations=4, bias_pct=-3.1, error_pct=3.1,
                    available=True).lean == "over"
    assert Accuracy(observations=4, bias_pct=3.1, error_pct=3.1,
                    available=True).lean == "under"


def test_the_presentation_layer_reads_the_engine_and_computes_nothing():
    """Catches the measure drifting back into the renderer.

    ``forecast_accuracy`` may format and choose words. It may not divide, and it
    may not average — those are the engine's.
    """
    tree = ast.parse((_ROOT / "mi_agent_pptx" / "forecast_accuracy.py"
                      ).read_text(encoding="utf-8"))
    divisions = [n.lineno for n in ast.walk(tree)
                 if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Div)]
    assert not divisions, f"arithmetic returned to the renderer at {divisions}"
    calls = {n.func.id for n in ast.walk(tree)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert "sum" not in calls, "the renderer is aggregating again"


def test_measure_reads_the_governed_payload():
    from mi_agent_pptx.forecast_accuracy import measure, describe
    payload = {"forecastAccuracy": {
        "available": True, "observations": 3, "biasPct": -4.2, "errorPct": 4.2,
        "worstPct": -4.8, "worstPeriod": "2026-03"}}
    acc = measure(payload)
    assert acc.available and acc.observations == 3
    assert "overstated" in describe(acc)
    assert "4.2%" in describe(acc)

    absent = measure({"forecastAccuracy": {
        "available": False, "observations": 1, "reason": "not enough history"}})
    assert not absent.available
    assert describe(absent) == "not enough history"
