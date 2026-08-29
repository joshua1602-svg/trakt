"""tests/test_pipeline_history_fixture.py — the five-week pipeline history.

`pipeline_evolution` returned ZERO periods against the demo store, so seven of
`evolution`'s thirty-two owned questions and two of its three route identities
could not be exercised at all. Any equivalence measured there would have been
green and meaningless: a refusal compared against a refusal.

These assertions are the fixture's contract. They are written from the movement
table in `build_fixture.py`, not copied back from a run, so a prep-layer change
that silently reclassifies a stage fails HERE with the week and the stage named
rather than somewhere downstream as a number nobody can check.

The fixture is discovered by the ordinary governed globs and prepared by the
shipped prep layer. There is no production branch for it.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "pipeline_history_5w"

WEEKS = ("2026-05-01", "2026-05-08", "2026-05-15", "2026-05-22", "2026-05-29")

#: Derived from the movement table, by hand.
EXPECTED_COUNTS = {
    "2026-05-01": {"KFI": 3, "APPLICATION": 2, "OFFER": 1},
    "2026-05-08": {"KFI": 3, "APPLICATION": 2, "OFFER": 2},
    "2026-05-15": {"KFI": 3, "APPLICATION": 2, "OFFER": 3},
    "2026-05-22": {"KFI": 2, "APPLICATION": 2, "OFFER": 1,
                   "COMPLETED": 2, "WITHDRAWN": 1},
    "2026-05-29": {"KFI": 1, "APPLICATION": 1, "OFFER": 3,
                   "COMPLETED": 2, "WITHDRAWN": 1},
}
EXPECTED_CASES = {"2026-05-01": 6, "2026-05-08": 7, "2026-05-15": 8,
                  "2026-05-22": 8, "2026-05-29": 8}
#: Loan amounts are 100k..800k by case, so each subtotal names its cases.
EXPECTED_AMOUNT = {"2026-05-01": 2_300_000.0, "2026-05-08": 2_800_000.0,
                   "2026-05-15": 3_600_000.0, "2026-05-22": 3_600_000.0,
                   "2026-05-29": 3_600_000.0}


@pytest.fixture(scope="module")
def evo():
    os.environ.setdefault("TRAKT_RUNTIME_MODE", "development")
    from mi_agent_api import evolution as ev
    return ev.pipeline_evolution(str(_FIXTURE), "client_001", None)


def test_all_five_extracts_are_discovered_by_the_governed_globs():
    """The fixture must be found the way any real extract is found."""
    from mi_agent_api import pipeline_contract as pc
    inv = pc.weekly_extract_inventory(str(_FIXTURE), "client_001")
    assert inv["uniqueWeeklyExtractsUsed"] == 5, inv
    assert inv["duplicatesExcluded"] == 0, inv


def test_the_series_carries_five_weeks_in_order(evo):
    assert [p["week"] for p in evo["periods"]] == list(WEEKS)


@pytest.mark.parametrize("week", WEEKS)
def test_case_count_and_amount_per_week(evo, week):
    p = next(p for p in evo["periods"] if p["week"] == week)
    assert p["metrics"]["pipeline_case_count"] == EXPECTED_CASES[week]
    assert p["metrics"]["pipeline_amount"] == EXPECTED_AMOUNT[week]


@pytest.mark.parametrize("week", WEEKS)
def test_stage_distribution_per_week(evo, week):
    got = {r["stage"]: r["count"] for r in evo["byStage"] if r["week"] == week}
    assert got == EXPECTED_COUNTS[week], (week, got)


def test_the_movements_the_fixture_exists_to_provide(evo):
    """Entry, progression, stasis, withdrawal and completion — each present.

    Asserted as MOVEMENTS rather than as five independent snapshots, because a
    fixture whose weeks are all the same shape would satisfy every test above
    and still exercise nothing.
    """
    by = {w: {r["stage"]: r["count"] for r in evo["byStage"] if r["week"] == w}
          for w in WEEKS}
    counts = {p["week"]: p["metrics"]["pipeline_case_count"] for p in evo["periods"]}
    # ENTRY: the population grows twice.
    assert counts["2026-05-01"] < counts["2026-05-08"] < counts["2026-05-15"]
    # PROGRESSION: OFFER builds while KFI drains.
    assert by["2026-05-29"]["OFFER"] > by["2026-05-01"]["OFFER"]
    assert by["2026-05-29"]["KFI"] < by["2026-05-01"]["KFI"]
    # COMPLETION and WITHDRAWAL appear only once cases reach them.
    assert "COMPLETED" not in by["2026-05-15"] and "WITHDRAWN" not in by["2026-05-15"]
    assert by["2026-05-22"]["COMPLETED"] == 2 and by["2026-05-22"]["WITHDRAWN"] == 1
    # STASIS: the population is flat across the last three weeks even though the
    # stage mix is not — a series that only ever grows proves less.
    assert counts["2026-05-15"] == counts["2026-05-22"] == counts["2026-05-29"]
    assert by["2026-05-22"] != by["2026-05-29"]


def test_the_fixture_needs_no_production_branch():
    """Nothing in production may name this fixture."""
    repo = Path(__file__).resolve().parent.parent
    offenders = []
    for path in (repo / "mi_agent_api").rglob("*.py"):
        if "test" in path.as_posix():
            continue
        if "pipeline_history_5w" in path.read_text():
            offenders.append(path.name)
    assert not offenders, offenders
