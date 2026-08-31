#!/usr/bin/env python3
"""tests/test_pipeline_live_stock.py — a case has ONE economic state.

The headline pipeline used to sum the whole weekly extract. A completed case has
funded — it is in the funded book — and a withdrawn case has gone away; counting
either as live pipeline stock reports the same exposure twice on one page, or
reports as "coming" something that is already here or already gone.

The correction is in the engine, at the layer that owns pipeline semantics, and
it is a SPLIT rather than a filter: live stock excludes terminal cases, and the
history every flow, conversion and reconciliation measure needs keeps all of
them. These tests pin both halves.

Fixture: ``tests/fixtures/pipeline_history_5w`` — five governed weekly extracts
in the real lender layout, whose cases progress and, from week four, terminate.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

pd = pytest.importorskip("pandas")

FIXTURE = _ROOT / "tests" / "fixtures" / "pipeline_history_5w"

#: Read off the fixture's own Status column, not from a run.
#: week -> (extract rows, live rows, live amount, extract amount)
TRUTH = {
    "2026-05-01": (6, 6, 2_300_000.0, 2_300_000.0),
    "2026-05-08": (7, 7, 2_800_000.0, 2_800_000.0),
    "2026-05-15": (8, 8, 3_600_000.0, 3_600_000.0),
    "2026-05-22": (8, 5, 2_400_000.0, 3_600_000.0),
    "2026-05-29": (8, 5, 2_400_000.0, 3_600_000.0),
}
TERMINAL_WEEK = "2026-05-29"


def _prepared(week: str):
    from mi_agent_api import pipeline_prep as prep
    source = next((FIXTURE / week).glob("*.csv"))
    df, report = prep.prepare_pipeline_mi_dataset(pd.read_csv(source), as_of_date=week)
    return df, report


# --------------------------------------------------------------------------- #
# 1-3. Live stock excludes terminal cases; history keeps them.
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("week", sorted(TRUTH))
def test_live_stock_matches_fixture_truth(week):
    """Catches: the headline summing the whole extract."""
    rows, live_rows, live_amount, extract_amount = TRUTH[week]
    _df, report = _prepared(week)
    assert report["row_count"] == rows
    assert report["live_row_count"] == live_rows, week
    assert report["total_pipeline_amount"] == pytest.approx(live_amount, abs=0.01)
    assert report["total_extract_amount"] == pytest.approx(extract_amount, abs=0.01)


def test_completed_cases_are_not_live_stock():
    """Catches: a funded loan counted again as pipeline."""
    df, report = _prepared(TERMINAL_WEEK)
    stages = df["pipeline_stage"].astype(str).str.upper()
    assert (stages == "COMPLETED").any(), "fixture carries no completed case"
    assert report["terminal_stage_counts"].get("COMPLETED"), report["terminal_stage_counts"]
    assert report["live_row_count"] + report["terminal_row_count"] == report["row_count"]


def test_withdrawn_cases_are_not_live_stock():
    """Catches: a case that went away reported as still coming."""
    df, report = _prepared(TERMINAL_WEEK)
    stages = df["pipeline_stage"].astype(str).str.upper()
    assert (stages == "WITHDRAWN").any(), "fixture carries no withdrawn case"
    assert report["terminal_stage_counts"].get("WITHDRAWN"), report["terminal_stage_counts"]


def test_terminal_cases_remain_available_to_history():
    """Catches: fixing the headline by DELETING rows.

    Every flow, conversion and reconciliation measure needs the terminal cases.
    The correction is a split, not a filter, so the frame must still carry them.
    """
    df, _report = _prepared(TERMINAL_WEEK)
    stages = set(df["pipeline_stage"].astype(str).str.upper())
    assert {"COMPLETED", "WITHDRAWN"} <= stages, stages
    assert len(df) == TRUTH[TERMINAL_WEEK][0], "rows were dropped from the frame"


def test_the_live_definition_is_stated_not_assumed():
    """A reader must be able to see what 'live' meant for this extract."""
    from mi_agent_api import pipeline_prep as prep
    _df, report = _prepared(TERMINAL_WEEK)
    assert report["live_stages"] == list(prep.ACTIVE_STAGES)
    assert set(prep.ACTIVE_STAGES).isdisjoint(prep.TERMINAL_STAGES)


# --------------------------------------------------------------------------- #
# 4. Forward measures use the live population.
# --------------------------------------------------------------------------- #

def test_weighted_expected_carries_no_terminal_case():
    """WEIGHTED expected was ALREADY correct, and this pins why.

    The stage-probability config assigns no completion probability to a terminal
    stage, so a completed or withdrawn case carries a NaN
    ``weighted_expected_funded_amount`` and never reached the sum. That is the
    right answer arrived at by a different route than the live/terminal split,
    and it is worth a test precisely because it is incidental: a future config
    that gave COMPLETED a probability of 1.0 would silently start reporting
    already-funded loans as expected future funding, and nothing else would
    catch it.
    """
    df, report = _prepared(TERMINAL_WEEK)
    terminal = df[df["pipeline_stage"].astype(str).str.upper().isin(
        ("COMPLETED", "WITHDRAWN"))]
    assert len(terminal) >= 2, "fixture carries no terminal case"
    assert pd.to_numeric(terminal["completion_probability"],
                         errors="coerce").isna().all(), (
        "a terminal stage was assigned a completion probability")

    live = df[df["pipeline_stage"].astype(str).str.upper().isin(
        ("KFI", "APPLICATION", "OFFER"))]
    assert report["weighted_expected_funded_amount"] == pytest.approx(
        float(pd.to_numeric(live["weighted_expected_funded_amount"],
                            errors="coerce").sum()), abs=0.01)


def test_expected_funded_excludes_terminal_cases():
    """This one WAS wrong, and the split fixed it.

    Unlike the weighted figure, ``expected_funded_amount`` is the unweighted
    balance and carries a value for every row including terminal ones — so the
    unweighted forward measure was summing already-funded and withdrawn cases.
    """
    df, report = _prepared(TERMINAL_WEEK)
    terminal = df[df["pipeline_stage"].astype(str).str.upper().isin(
        ("COMPLETED", "WITHDRAWN"))]
    terminal_expected = float(pd.to_numeric(
        terminal["expected_funded_amount"], errors="coerce").sum())
    assert terminal_expected > 0, (
        "fixture terminal cases carry no expected amount, so this proves nothing")

    live = df[df["pipeline_stage"].astype(str).str.upper().isin(
        ("KFI", "APPLICATION", "OFFER"))]
    assert report["expected_funded_amount"] == pytest.approx(
        float(pd.to_numeric(live["expected_funded_amount"], errors="coerce").sum()),
        abs=0.01)
    assert report["expected_funded_amount"] < terminal_expected + report["expected_funded_amount"]


def test_a_tape_with_no_stage_column_is_all_live():
    """Catches: an absent stage column read as evidence of termination.

    A missing stage is a coverage gap, raised separately by the dataset
    validator. It must not silently empty the pipeline.
    """
    from mi_agent_api import pipeline_prep as prep
    df = pd.DataFrame({"current_outstanding_balance": [1.0, 2.0, 3.0]})
    assert bool(prep.live_mask(df).all())


def test_a_case_cannot_be_funded_and_live_pipeline_at_once():
    """THE INVARIANT. A case has one economic state at a point in time.

    ``pipeline_status`` is the engine's own funded/pipeline/withdrawn view of a
    row. Every row it calls "funded" is a completed case sitting in the funded
    book, and not one penny of it may appear in live pipeline stock.
    """
    df, report = _prepared(TERMINAL_WEEK)
    funded_rows = df[df["pipeline_status"].astype(str) == "funded"]
    assert len(funded_rows) >= 1, "fixture carries no funded-status case"
    funded_balance = float(pd.to_numeric(
        funded_rows["current_outstanding_balance"], errors="coerce").sum())

    live_balance = report["total_pipeline_amount"]
    extract_balance = report["total_extract_amount"]
    withdrawn = df[df["pipeline_status"].astype(str) == "withdrawn"]
    withdrawn_balance = float(pd.to_numeric(
        withdrawn["current_outstanding_balance"], errors="coerce").sum())

    # The live figure is the extract less exactly the terminal balances.
    assert live_balance == pytest.approx(
        extract_balance - funded_balance - withdrawn_balance, abs=0.01)
    assert live_balance + funded_balance <= extract_balance + 0.01
