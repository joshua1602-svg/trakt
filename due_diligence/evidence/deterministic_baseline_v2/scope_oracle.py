#!/usr/bin/env python3
"""Independent fixtures and truth for the four AnalyticalScope dimensions.

Imports nothing from the product calculation path. pandas, numpy and explicit
column names only — the same discipline `portfolio_truth_oracle` applies, and for
the same reason: asking the product to compute the expected answer would be the
product marking its own homework, and would have agreed with every scope defect
baseline v2 found.

WHAT THIS ADDS OVER portfolio_truth_oracle
------------------------------------------
That oracle's book is a single flat frame with no period, no portfolio lens and
no dataset identity, which is exactly why baseline v2 could only test the scope
dimensions NEGATIVELY — it could show a period was ignored, never that a period
was honoured. These fixtures carry the governed columns:

    reporting_date          three periods, materially different totals
    source_portfolio_type   Direct / Acquired, materially different totals

Both are real governed fields in `mi_agent/mi_semantics_field_registry.yaml`
(`reporting_date` -> canonical_field `reporting_date`; `source_portfolio_type` ->
canonical_field `source_portfolio_type`), so a scope request against them is
expressible in the existing contract rather than needing a new one.

ACCIDENTAL EQUALITY IS THE ENEMY
--------------------------------
A fixture whose period totals happen to coincide would let a scope-blind engine
pass. `assert_materially_distinct()` runs at import and fails loudly if any two
figures this bank relies on are within a tolerance of each other, so a false
pass cannot be built by accident.
"""
from __future__ import annotations

from itertools import combinations
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

BALANCE = "current_outstanding_balance"
LTV = "current_loan_to_value"
RATE = "current_interest_rate"
AGE = "youngest_borrower_age"
PERIOD = "reporting_date"
LENS = "source_portfolio_type"
REGION = "collateral_geography"

#: The three periods, newest first. CURRENT is what a plan stating no period and
#: a plan stating CURRENT should both mean.
CURRENT = "2026-06-30"
PREVIOUS = "2026-05-31"
HISTORICAL_A = "2026-03-31"
PERIODS = (HISTORICAL_A, PREVIOUS, CURRENT)

#: Periods the fixture deliberately does NOT carry.
UNAVAILABLE_PERIOD = "2019-12-31"      # plausible, simply not in the book
IMPOSSIBLE_PERIOD = "2026-02-30"       # not a date in any calendar
FUTURE_PERIOD = "2099-12-31"

DIRECT = "Direct"
ACQUIRED = "Acquired"
LENSES = (DIRECT, ACQUIRED)

_REGIONS = ("Scotland", "North West", "South East", "Wales")
_PRODUCTS = ("Lump Sum", "Drawdown")


def scoped_book(seed: int = 20260910) -> pd.DataFrame:
    """A book carrying a period and a portfolio lens on every row.

    Row counts differ per period ON PURPOSE — a book grows and redeems — so a
    period-blind sum is not merely slightly wrong, it is wrong by an amount no
    rounding could explain. Balances are scaled per period and per lens so that
    every figure this bank compares is far apart.
    """
    rng = np.random.default_rng(seed)
    frames: List[pd.DataFrame] = []
    # (period, lens) -> (rows, balance scale). Chosen so the twelve figures this
    # bank compares land roughly at 5/9/14/20/25/28/35/44/55/69/110/138 million,
    # every pair at least 3m apart. The first attempt at this fixture put the
    # March total within 333k of the current Acquired total and
    # `assert_materially_distinct` rejected it — which is the guard earning its
    # place, because a scope-blind engine would have passed on that coincidence.
    shape = {
        (HISTORICAL_A, DIRECT): (90, 1.000),
        (HISTORICAL_A, ACQUIRED): (25, 0.889),
        (PREVIOUS, DIRECT): (140, 1.111),
        (PREVIOUS, ACQUIRED): (40, 1.000),
        (CURRENT, DIRECT): (200, 1.222),
        (CURRENT, ACQUIRED): (50, 1.244),
    }
    n = 0
    for (period, lens), (rows, scale) in shape.items():
        frames.append(pd.DataFrame({
            "loan_identifier": [f"L{i:05d}" for i in range(n, n + rows)],
            PERIOD: period,
            LENS: lens,
            BALANCE: (rng.uniform(50_000, 400_000, rows) * scale).round(2),
            LTV: rng.uniform(12, 88, rows).round(1),
            RATE: rng.uniform(2.5, 9.5, rows).round(2),
            AGE: rng.integers(60, 95, rows),
            REGION: rng.choice(_REGIONS, rows),
            "erm_product_type": rng.choice(_PRODUCTS, rows),
            "borrower_type": rng.choice(("Single", "Joint"), rows),
        }))
        n += rows
    book = pd.concat(frames, ignore_index=True)
    return book


# --------------------------------------------------------------------------- #
# truth, computed longhand
# --------------------------------------------------------------------------- #

def _mask(frame: pd.DataFrame, *, period=None, lens=None,
          predicates: Sequence[Tuple[str, str, object]] = ()) -> pd.Series:
    keep = pd.Series(True, index=frame.index)
    if period is not None:
        keep &= frame[PERIOD] == period
    if lens is not None:
        keep &= frame[LENS] == lens
    for field, op, value in predicates:
        col = frame[field]
        if op == "eq":
            keep &= col == value
        elif op == "ne":
            keep &= col != value
        elif op == "gt":
            keep &= col > value
        elif op == "ge":
            keep &= col >= value
        elif op == "lt":
            keep &= col < value
        elif op == "le":
            keep &= col <= value
        else:
            raise ValueError(f"the scope oracle has no rule for {op!r}")
    return keep


def total(frame, column=BALANCE, *, period=None, lens=None, predicates=()):
    return float(frame.loc[_mask(frame, period=period, lens=lens,
                                 predicates=predicates), column].sum())


def rows(frame, *, period=None, lens=None, predicates=()):
    return int(_mask(frame, period=period, lens=lens,
                     predicates=predicates).sum())


def mean(frame, column, *, period=None, lens=None, predicates=()):
    sel = frame.loc[_mask(frame, period=period, lens=lens, predicates=predicates),
                    column]
    return float(sel.mean()) if len(sel) else float("nan")


def grouped(frame, by: Iterable[str], column=BALANCE, *, period=None, lens=None,
            predicates=()) -> Dict[Tuple[str, ...], float]:
    sel = frame.loc[_mask(frame, period=period, lens=lens, predicates=predicates)]
    out: Dict[Tuple[str, ...], float] = {}
    for key, chunk in sel.groupby(list(by), observed=True, dropna=False):
        k = key if isinstance(key, tuple) else (key,)
        out[tuple(str(x) for x in k)] = float(chunk[column].sum())
    return out


# --------------------------------------------------------------------------- #
# datasets — two genuinely different populations
# --------------------------------------------------------------------------- #

def dataset_pair(seed: int = 20260910) -> Dict[str, pd.DataFrame]:
    """Two datasets a plan could name, with deliberately far-apart totals.

    `funded` is the scoped book. `pipeline` is a different population entirely —
    fewer, larger cases — so a dataset-blind execution cannot coincide with the
    right answer.
    """
    funded = scoped_book(seed=seed)
    rng = np.random.default_rng(seed + 1)
    # 60 rows near 1.4m each puts `pipeline` around 84m — clear of every funded
    # figure. An earlier draft used 35 x 750k, which landed 83k from the Acquired
    # lens total and slipped past a guard that only compared datasets against
    # `funded` instead of against every figure. Both faults are fixed.
    rows_ = 60
    pipeline = pd.DataFrame({
        "loan_identifier": [f"P{i:05d}" for i in range(rows_)],
        PERIOD: CURRENT,
        LENS: DIRECT,
        BALANCE: (rng.uniform(1_200_000, 1_600_000, rows_)).round(2),
        LTV: rng.uniform(20, 60, rows_).round(1),
        RATE: rng.uniform(3.0, 8.0, rows_).round(2),
        AGE: rng.integers(60, 90, rows_),
        REGION: rng.choice(_REGIONS, rows_),
        "erm_product_type": rng.choice(_PRODUCTS, rows_),
        "borrower_type": rng.choice(("Single", "Joint"), rows_),
    })
    return {"funded": funded, "pipeline": pipeline}


# --------------------------------------------------------------------------- #
# the guard against a fixture that could produce a false pass
# --------------------------------------------------------------------------- #

def assert_materially_distinct(tolerance: float = 1_000_000.0) -> Dict[str, float]:
    """Every figure this bank compares must be far from every other.

    Without this a scope-blind engine could pass by coincidence: if the current
    total happened to equal the March total, "the period was ignored" and "the
    period was honoured" would be indistinguishable. Tolerance is deliberately
    coarse — these are millions apart by construction, and anything closer than
    a million means the fixture needs rebuilding, not the tolerance relaxing.
    """
    book = scoped_book()
    figures = {
        "all_periods_all_lenses": total(book),
        **{f"period_{p}": total(book, period=p) for p in PERIODS},
        **{f"lens_{l}": total(book, lens=l) for l in LENSES},
        **{f"period_{p}_lens_{l}": total(book, period=p, lens=l)
           for p in PERIODS for l in LENSES},
    }
    datasets = dataset_pair()
    figures["dataset_funded"] = total(datasets["funded"])
    figures["dataset_pipeline"] = total(datasets["pipeline"])
    # EVERY pair, datasets included. Comparing the two datasets only against each
    # other is how a 83k gap between `pipeline` and the Acquired lens total got
    # through the first draft of this guard.
    # `dataset_funded` IS the scoped book, so its equality with the all-periods
    # total is the fixture being consistent, not a coincidence that could mask a
    # defect: no test distinguishes those two figures from each other.
    intended_identities = {frozenset({"all_periods_all_lenses", "dataset_funded"})}
    clashes = [(a, b, round(abs(figures[a] - figures[b]), 2))
               for a, b in combinations(figures, 2)
               if abs(figures[a] - figures[b]) < tolerance
               and frozenset({a, b}) not in intended_identities]
    if clashes:
        raise AssertionError(
            "the scope fixture contains figures close enough to allow an "
            f"accidental pass: {clashes[:4]}")
    return figures


#: Runs at import. A fixture that could produce a false pass must not be usable.
DISTINCT_FIGURES = assert_materially_distinct()


if __name__ == "__main__":
    import json
    print(json.dumps({k: round(v, 2) for k, v in DISTINCT_FIGURES.items()},
                     indent=1))
    b = scoped_book()
    print(f"\nrows={len(b)}  periods={sorted(b[PERIOD].unique())}  "
          f"lenses={sorted(b[LENS].unique())}")
