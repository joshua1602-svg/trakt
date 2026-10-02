#!/usr/bin/env python3
"""tests/test_measure_basis.py — measures that do not tie must say why.

A reader who divides "average loan balance" by "weighted average property value"
expects "weighted average current LTV". They do not get it, and on a real book
the gap is 15.2 percentage points. Two independent causes, both correct
behaviour:

  * average loan balance is UNWEIGHTED (one vote per loan) while property value
    is BALANCE-WEIGHTED (one vote per pound) — averages over different
    populations do not divide into one another;
  * the LTV tile is the mean of each loan's own LTV (the typical POUND's
    gearing) while dividing the money tiles gives the ratio of the aggregates
    (the BOOK's gearing) — different economic statements, separated by Jensen's
    inequality on any book with dispersion.

So the fix is not to redefine weighted average LTV. It is to state the basis on
each measure, and to surface aggregate gearing under its own name for readers
who want the other one.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

pd = pytest.importorskip("pandas")


def _snapshot():
    from mi_agent_api import snapshots
    # Deliberate dispersion in LTV: without it the two bases coincide and the
    # test proves nothing.
    df = pd.DataFrame({
        "loan_identifier": [f"L{i}" for i in range(4)],
        "current_outstanding_balance": [100_000.0, 200_000.0, 300_000.0, 400_000.0],
        "current_valuation_amount": [500_000.0, 400_000.0, 500_000.0, 500_000.0],
        "current_loan_to_value": [20.0, 50.0, 60.0, 80.0],
        "current_interest_rate": [6.0, 7.0, 7.5, 8.0],
        "youngest_borrower_age": [65, 70, 75, 80],
    })
    return df, snapshots.compute_funded_snapshot(
        df, {}, client_id="basis", run_id="r1", reporting_date="2026-06-30")


def _tile(snapshot, kpi_id):
    return next((k for k in snapshot["kpis"] if k["id"] == kpi_id), None)


# --------------------------------------------------------------------------- #
# Every derived measure states its basis.
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("kpi_id,expected_basis", [
    ("avg_balance", "per loan, unweighted"),
    ("wa_current_ltv", "balance-weighted"),
    ("wa_rate", "balance-weighted"),
    ("wa_age", "balance-weighted"),
    ("wa_property_value", "balance-weighted"),
    ("aggregate_gearing", "ratio of aggregates"),
])
def test_each_derived_measure_states_its_basis(kpi_id, expected_basis):
    _df, snapshot = _snapshot()
    tile = _tile(snapshot, kpi_id)
    assert tile is not None, f"{kpi_id} did not resolve"
    assert tile["basis"] == expected_basis, tile
    assert tile["numerator"] and tile["denominator"], tile


def test_a_raw_count_needs_no_basis():
    """Basis is for DERIVED measures. A balance is a sum, not an average."""
    _df, snapshot = _snapshot()
    assert _tile(snapshot, "balance")["basis"] is None
    assert _tile(snapshot, "loans")["basis"] is None


# --------------------------------------------------------------------------- #
# The measures genuinely do not tie, and that is the point.
# --------------------------------------------------------------------------- #

def test_the_tiles_do_not_tie_and_are_not_made_to():
    """Catches: someone 'fixing' the gap by redefining WA LTV.

    Weighted average LTV must remain the balance-weighted mean of loan-level
    LTVs. If it ever equals the ratio of aggregates on a dispersed book, it has
    been silently redefined into a different measure.
    """
    _df, snapshot = _snapshot()
    wa_ltv = _tile(snapshot, "wa_current_ltv")["raw"]
    gearing = _tile(snapshot, "aggregate_gearing")["raw"]
    avg_balance = _tile(snapshot, "avg_balance")["raw"]
    wa_value = _tile(snapshot, "wa_property_value")["raw"]

    assert wa_ltv == pytest.approx(62.0, abs=0.5), wa_ltv
    assert gearing != pytest.approx(wa_ltv, abs=1.0), (
        "weighted average LTV has been redefined as the ratio of aggregates")
    implied = avg_balance / wa_value * 100.0
    assert implied != pytest.approx(wa_ltv, abs=1.0), (
        "the fixture has no dispersion, so this proves nothing")


def test_wa_ltv_is_the_balance_weighted_mean_of_loan_level_ltv():
    """The definition, pinned. It may not drift."""
    df, snapshot = _snapshot()
    bal = df["current_outstanding_balance"]
    expected = float((df["current_loan_to_value"] * bal).sum() / bal.sum())
    assert _tile(snapshot, "wa_current_ltv")["raw"] == pytest.approx(expected, abs=0.01)


def test_aggregate_gearing_is_the_ratio_of_aggregates():
    df, snapshot = _snapshot()
    expected = float(df["current_outstanding_balance"].sum()
                     / df["current_valuation_amount"].sum() * 100.0)
    assert _tile(snapshot, "aggregate_gearing")["raw"] == pytest.approx(expected, abs=0.01)


# --------------------------------------------------------------------------- #
# The cohort payload carries two LTV-shaped figures on two bases.
# --------------------------------------------------------------------------- #

def test_cohort_nneg_headroom_declares_its_basis():
    """``wa_ltv`` and ``nneg_headroom_pct`` are not complements of one another."""
    from mi_agent_api.evolution import _nneg_metrics
    df = pd.DataFrame({
        "current_outstanding_balance": [100_000.0, 300_000.0],
        "current_valuation_amount": [500_000.0, 400_000.0],
    })
    out = _nneg_metrics(df)
    assert out["nneg_headroom_pct_basis"].startswith("ratio of aggregates")
    assert out["wa_ltv_basis"].startswith("balance-weighted mean")


# --------------------------------------------------------------------------- #
# The methodology claim matches the evidence.
# --------------------------------------------------------------------------- #

def test_the_pack_no_longer_claims_figures_are_identical_to_the_dashboard():
    """Catches: a marketing claim stronger than the evidence.

    "Identical to the management dashboard" was true of the engine-owned
    payloads and untrue while economic values were derived independently
    downstream. The defensible claim is shared definitions.
    """
    source = (_ROOT / "mi_agent_pptx" / "deck.py").read_text(encoding="utf-8")
    body = "\n".join(l for l in source.splitlines() if not l.strip().startswith("#"))
    assert "identical to the management dashboard" not in body
    assert "shared reporting definitions" in body
