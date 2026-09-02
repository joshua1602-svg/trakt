"""A static pool is only a pool once its vintage has stopped forming.

Reported defect: the 2025 vintage was first SEEN at 2025-10 with 33 loans, so
the pool anchored there. Loans originated in November joined afterwards, the
"surviving" count rose 33 → 73 — the one thing a static pool may never do — and
balance retention read 211.6%, which the panel's own footnote then attributed to
interest roll-up.

The pool now anchors at formation completion. Periods before it are still
reported, so a reader watches the vintage build, but they are marked ``forming``
and carry no retention: a survival rate against a pool that is still admitting
loans is not a survival rate.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mi_agent_api import cohorts as C  # noqa: E402

BALANCE = "current_outstanding_balance"


def _frame(loan_ids, per_loan_balance, origination_dates):
    return pd.DataFrame({
        "loan_id": list(loan_ids),
        BALANCE: [per_loan_balance] * len(loan_ids),
        "origination_date": pd.to_datetime(list(origination_dates)),
    })


@pytest.fixture
def still_forming_vintage():
    """33 loans by October, 40 more originated in November, followed to 2026-06."""
    early = [f"L{i:03d}" for i in range(33)]
    late = [f"L{i:03d}" for i in range(33, 73)]
    dates_all = ["2025-06-01"] * 33 + ["2025-11-15"] * 40
    return [
        {"reporting_date": "2025-10-31",
         "df": _frame(early, 4_200_000 / 33, ["2025-06-01"] * 33)},
        {"reporting_date": "2025-11-30",
         "df": _frame(early + late, 8_900_000 / 73, dates_all)},
        {"reporting_date": "2026-06-30",
         "df": _frame(early + late, 8_900_000 / 73, dates_all)},
    ]


def test_the_pool_anchors_to_the_complete_vintage(still_forming_vintage):
    out = C.cohort_static_pool(still_forming_vintage, vintage="2025", grain="Y")
    assert out["originalLoanCount"] == 73, "the pool is the whole vintage, not the part seen first"
    assert out["formationEnd"] == "2025-12-31"
    assert out["poolAnchored"] is True


def test_the_surviving_count_never_rises_once_the_pool_is_fixed(still_forming_vintage):
    out = C.cohort_static_pool(still_forming_vintage, vintage="2025", grain="Y")
    fixed = [p for p in out["periods"] if not p["forming"]]
    counts = [p["survivingLoanCount"] for p in fixed]
    assert counts == sorted(counts, reverse=True), counts


def test_no_retention_above_one_hundred_percent_from_late_formation(still_forming_vintage):
    out = C.cohort_static_pool(still_forming_vintage, vintage="2025", grain="Y")
    assert all((p["balanceRetention"] or 0) <= 1.0 for p in out["periods"])


def test_forming_periods_are_reported_but_carry_no_retention(still_forming_vintage):
    out = C.cohort_static_pool(still_forming_vintage, vintage="2025", grain="Y")
    forming = [p for p in out["periods"] if p["forming"]]
    assert [p["period"] for p in forming] == ["2025-10", "2025-11"]
    assert out["formingPeriods"] == 2
    for p in forming:
        assert p["balanceRetention"] is None and p["loanRetention"] is None
        # A loan that has not completed yet is not an exit.
        assert p["exitsInPeriod"] == 0 and p["cumulativeExits"] == 0
    # The balances themselves are still reported — nothing is hidden.
    assert [p["currentBalance"] for p in forming] == [4_200_000.0, 8_900_000.0]


def test_a_settled_vintage_is_unaffected():
    """A vintage whose formation closed before the first reporting period keeps
    its previous behaviour exactly: anchored at first sight, retention from the
    first row, exits counted."""
    loans = [f"L{i:03d}" for i in range(10)]
    frames = [
        {"reporting_date": "2026-01-31",
         "df": _frame(loans, 100_000.0, ["2024-06-01"] * 10)},
        {"reporting_date": "2026-02-28",
         "df": _frame(loans[:8], 100_000.0, ["2024-06-01"] * 8)},
    ]
    out = C.cohort_static_pool(frames, vintage="2024", grain="Y")
    assert out["formingPeriods"] == 0
    assert out["originalLoanCount"] == 10
    assert out["periods"][0]["balanceRetention"] == 1.0
    assert out["periods"][1]["survivingLoanCount"] == 8
    assert out["periods"][1]["exitsInPeriod"] == 2
    assert out["periods"][1]["balanceRetention"] == 0.8


@pytest.mark.parametrize("vintage,grain,expected", [
    ("2025", "Y", "2025-12-31"),
    ("2025-Q3", "Q", "2025-09-30"),
    ("2024-02", "M", "2024-02-29"),   # leap year
    ("2025-11", "M", "2025-11-30"),
    ("not-a-vintage", "Y", None),
])
def test_formation_end_per_grain(vintage, grain, expected):
    assert C._formation_end(vintage, grain) == expected


def test_an_entirely_unformed_vintage_reports_no_retention():
    """Every reporting period pre-dates formation: balances are shown, but no
    pool exists yet, so nothing claims to be a survival rate."""
    loans = [f"L{i:03d}" for i in range(5)]
    frames = [{"reporting_date": "2026-03-31",
               "df": _frame(loans, 100_000.0, ["2026-02-01"] * 5)}]
    out = C.cohort_static_pool(frames, vintage="2026", grain="Y")
    assert out["poolAnchored"] is False
    assert out["originalLoanCount"] is None
    assert all(p["forming"] and p["balanceRetention"] is None for p in out["periods"])
