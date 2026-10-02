#!/usr/bin/env python3
"""tests/test_forecast_composition.py

The forecast cuts show what the forecast is MADE OF.

A forecast bar drawn as one block shows the destination and hides the journey:
a reader cannot see how much of a category's forecast exposure already exists
and how much is expected to arrive. Those are facts of different certainty, and
a funder is buying one of them.

Nothing is recomputed to draw it. ``workspace.forecast_breakdowns`` already
emits funded, weighted-pipeline and forecast per category; the parts simply had
to survive the top-N cap, which they did not.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

pd = pytest.importorskip("pandas")

from mi_agent_api import workspace as W  # noqa: E402


def frames(n_regions=12):
    funded = pd.DataFrame([
        {"geographic_region_obligor": f"R{i % n_regions}",
         "ltv_bucket": f"{20 + (i % 6) * 10}-{30 + (i % 6) * 10}%",
         "current_outstanding_balance": 1_000_000.0 + i * 1_000}
        for i in range(120)])
    pipeline = pd.DataFrame([
        {"pipeline_stage": "KFI",
         "geographic_region_obligor": f"R{i % n_regions}",
         "ltv_bucket": f"{20 + (i % 6) * 10}-{30 + (i % 6) * 10}%",
         "expected_completion_month": "2026-07",
         "weighted_expected_funded_amount": 50_000.0 + i * 100}
        for i in range(60)])
    return funded, pipeline


def test_current_plus_incremental_equals_forecast_per_category():
    """The identity the stacked bar draws."""
    brk = W.forecast_breakdowns(*frames())
    for key in ("byRegion", "byLtvBucket"):
        for row in brk[key]:
            assert (row["fundedAmount"] + row["weightedPipelineAmount"]
                    == pytest.approx(row["forecastAmount"], abs=0.01)), row


def test_the_parts_survive_the_top_n_cap():
    """Catches: a capped row carrying the total but not what it is made of.

    The cap reshaped rows and dropped ``fundedAmount``, so the drawn form could
    show the forecast and nothing else — the bar stacked to a sliver because
    the funded part read as zero.
    """
    brk = W.forecast_breakdowns(*frames(n_regions=14))
    capped = brk["byRegionCapped"]
    assert any(r.get("isOther") for r in capped), "the cap did not bind"
    for row in capped:
        assert "fundedAmount" in row, row


def test_a_capped_row_reconciles_like_every_other_row():
    """Including the aggregated Other, whose parts must sum to its total."""
    brk = W.forecast_breakdowns(*frames(n_regions=14))
    for row in brk["byRegionCapped"]:
        parts = (row.get("fundedAmount") or 0.0) + \
            (row.get("weightedExpectedFundedAmount") or 0.0)
        assert parts == pytest.approx(row["pipelineAmount"], abs=0.05), row


def test_the_stacked_total_reconciles_to_the_headline_forecast():
    """Every category's forecast, summed, is the forecast balance."""
    funded, pipeline = frames()
    brk = W.forecast_breakdowns(funded, pipeline)
    total = float(funded["current_outstanding_balance"].sum()) + \
        float(pipeline["weighted_expected_funded_amount"].sum())
    for key in ("byRegion", "byLtvBucket"):
        assert sum(r["forecastAmount"] for r in brk[key]) == \
            pytest.approx(total, rel=1e-6), key


def test_a_book_with_no_pipeline_has_no_incremental_part():
    """The bar is then all funded — which is honest, not a rendering fault."""
    funded, _ = frames()
    brk = W.forecast_breakdowns(funded, None)
    for row in brk["byRegion"]:
        assert row["weightedPipelineAmount"] == 0.0
        assert row["forecastAmount"] == pytest.approx(row["fundedAmount"])


# --------------------------------------------------------------------------- #
# The renderer draws the parts it is given, and never rescales them.
# --------------------------------------------------------------------------- #

def test_the_renderer_records_the_segments_it_drew(tmp_path):
    """A chart becomes pixels the moment it is saved; the record is how a test
    can see what it actually drew."""
    pytest.importorskip("matplotlib")
    from mi_agent_pptx import render as R

    rows = [{"label": "London", "funded": 8.0, "expected": 2.0, "balance": 10.0},
            {"label": "Wales", "funded": 5.0, "expected": 1.0, "balance": 6.0}]
    segments = ({"key": "funded", "label": "Current funded", "color": "#8b95d6"},
                {"key": "expected", "label": "Expected additions", "color": "#3fd6a8"})
    with R.record_renders() as drawn:
        R.draw_stacked_barlist(tmp_path / "s.png", rows, segments, 6.0, 2.0,
                               total_key="balance", chart_id="fc_test")
    entry = [d for d in drawn if d["kind"] == "stacked_barlist"][0]
    assert entry["segments"] == ["funded", "expected"]
    assert entry["values"] == [10.0, 6.0]
    assert entry["categories"] == ["London", "Wales"]
