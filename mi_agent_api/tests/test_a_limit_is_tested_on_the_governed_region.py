#!/usr/bin/env python3
"""A concentration limit is tested on the region MI answers on, and says so.

THE DEFECT, reproduced below against the unfixed code. `engine.region_taxonomy`
harmonises the books onto `canonical_region_reporting`, and the MI Query Agent
prefers that column for every region question. The risk-limit evaluator never
learned about it: `_REGION_COLUMNS` began at the RAW `collateral_geography`, so
on a tape where the same region arrives spelled three ways —

    "South West", "south-west", "SOUTH WEST"

— the limit test measured three separate 25% bars and reported 25% against a
book that is 75% South West. A limit written "South West ≤ 40%" read COMPLIANT
on a book in breach, while the MI Agent, asked the same question in words,
answered 75%. Two dashboard surfaces, one word, contradictory numbers.

Two things are fixed here and they are different:

    the BASIS      the evaluator reads the governed harmonised column when the
                   tape carries one, so the limit and the answer agree.
    the DISCLOSURE the envelope names which region column produced the actuals
                   and what share of the book carried a value there, because a
                   limit measured on part of a book is only defensible if the
                   part is stated.

Neither invents a mapping. A row with no governed region stays unresolved and
is counted as such — never assigned to a region to make the coverage look
better.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mi_agent_api import risk_limits as rl

_BALANCE = "current_outstanding_balance"
REPORTING = "canonical_region_reporting"


def _split_book():
    """75% South West, spelled three ways, plus 25% London."""
    return pd.DataFrame({
        "collateral_geography": ["South West", "south-west", "SOUTH WEST",
                                 "London"],
        REPORTING: ["South West"] * 3 + ["London"],
        "region_mapping_method": ["exact", "synonym", "synonym", "exact"],
        _BALANCE: [100.0, 100.0, 100.0, 100.0],
    })


def _partial_book():
    """Half the book has no governed region at all."""
    return pd.DataFrame({
        "collateral_geography": ["South West", "London", None, None],
        REPORTING: ["South West", "London", None, None],
        "region_mapping_method": ["exact", "exact", "unresolved", "absent"],
        _BALANCE: [100.0, 100.0, 100.0, 100.0],
    })


# ------------------------------------------------------------------- basis #
def test_the_evaluator_reads_the_governed_harmonised_region():
    assert rl._region_column(_split_book()) == REPORTING


def test_three_spellings_of_one_region_are_one_bar():
    shares = rl._region_shares(_split_book())
    lookup = {str(r[rl._REGION]).lower(): round(float(r["balance_share"]), 4)
              for _, r in shares.iterrows()}
    assert lookup["south west"] == 0.75
    assert lookup["london"] == 0.25


def test_a_limit_written_south_west_reads_the_breach():
    """The whole point: 75% against a 40% limit is a BREACH, and the unfixed
    evaluator reported 25% — compliant."""
    limits = [{"limit_id": "geo-sw", "category": "geographic_concentration",
               "region": "South West", "limit_value": 40.0, "direction": "max",
               "unit": "percent"}]
    tests = rl._compute_tests(_split_book(), limits, None, "test", "c1")
    geographic = [t for t in tests if t["category"] == "geographic_concentration"]
    assert geographic and geographic[0]["actualValue"] == 75.0
    assert geographic[0]["status"] == "red"


def test_a_tape_with_no_harmonised_column_still_falls_back():
    """A deployment with no taxonomy configured behaves exactly as before."""
    raw = _split_book().drop(columns=[REPORTING, "region_mapping_method"])
    assert rl._region_column(raw) == "collateral_geography"


# -------------------------------------------------------------- disclosure #
def test_the_basis_and_its_coverage_are_published():
    basis = rl.region_basis_block(_partial_book())
    assert basis["field"] == REPORTING
    assert basis["level"] == "reporting"
    assert basis["rows"] == 4 and basis["resolved"] == 2
    assert basis["share"] == 0.5


def test_a_partially_covered_limit_says_so():
    limits = [{"limit_id": "geo-sw", "category": "geographic_concentration",
               "region": "South West", "limit_value": 40.0, "direction": "max",
               "unit": "percent"}]
    tests = rl._compute_tests(_partial_book(), limits, None, "test", "c1")
    geographic = [t for t in tests if t["category"] == "geographic_concentration"]
    assert geographic
    notes = (geographic[0].get("notes") or "")
    assert "50" in notes and "region" in notes.lower()


def test_a_fully_covered_limit_is_not_caveated():
    limits = [{"limit_id": "geo-sw", "category": "geographic_concentration",
               "region": "South West", "limit_value": 40.0, "direction": "max",
               "unit": "percent"}]
    tests = rl._compute_tests(_split_book(), limits, None, "test", "c1")
    geographic = [t for t in tests if t["category"] == "geographic_concentration"]
    assert "carry a governed region" not in (geographic[0].get("notes") or "")


def test_a_book_with_no_region_at_all_has_no_basis():
    plain = pd.DataFrame({_BALANCE: [1.0, 2.0]})
    assert rl.region_basis_block(plain) is None
