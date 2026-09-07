#!/usr/bin/env python3
"""Every surface that answers a region question answers it the same way.

A gate over ``due_diligence/evidence/mi_geography/regional_coherence.py``, which
compares four independent paths on the demo platform book:

    serving · historical · dashboard · a plain pandas oracle

The oracle carries the weight. The first three share code and could agree with
each other while all being wrong; the oracle reads the tape and groups it,
depending on nothing under test.

Skipped where the demo platform tape has not been generated — it is not checked
in (``python -m demo_platform.run_demo --generate --orchestrate`` builds it), and
a gate that silently passes on a missing fixture is worse than one that says it
did not run.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from due_diligence.evidence.mi_geography import regional_coherence as RC

pytestmark = pytest.mark.skipif(
    not RC.DEFAULT_TAPE.is_file(),
    reason="demo platform tape not generated")


@pytest.fixture(scope="module")
def report():
    return RC.run()


def test_the_region_column_is_the_configured_basis(report):
    assert report["contract"]["primaryBasis"] == "collateral"
    assert report["regionColumn"] == "collateral_geography"


def test_the_dashboard_measures_the_same_column_as_the_query(report):
    """The defect this closes: the dashboard's regional series was hard-coded to
    the BORROWER column for every book, so it and the MI Agent could describe the
    same book on two different geographies with nothing saying so."""
    assert report["dashboardColumn"] == report["regionColumn"]


@pytest.mark.parametrize("pair", ["serving_vs_oracle", "historical_vs_oracle",
                                  "dashboard_vs_oracle", "serving_vs_historical"])
def test_no_cell_disagrees(report, pair):
    assert report["disagreements"][pair] == []


def test_the_breakdown_reconciles_to_the_book(report):
    assert report["reconciliation"]["agrees"]
    assert report["reconciliation"]["excludedByOracle"] >= 0


def test_the_harmonised_column_resolves_rather_than_sitting_empty(report):
    """It is no longer what a region question binds to, but it must still work.

    Before the source order was stated MI-side, this read `populated: 0` and
    `methods: {"unresolved": 11035}` — the harmonisation read the obligor column
    first, that column holds ITL3 codes, and no ITL1 taxonomy resolves those.
    """
    harmonised = report.get("harmonised") or {}
    assert harmonised.get("populated", 0) > 0.99 * report["rows"]
    assert harmonised.get("methods", {}).get("exact", 0) > 0


def test_every_surface_reports_the_same_number_of_regions(report):
    counts = set(report["cells"].values())
    assert len(counts) == 1, report["cells"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
