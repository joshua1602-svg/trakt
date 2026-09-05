#!/usr/bin/env python3
"""Independently calculated portfolio truth. Imports nothing from the product.

WHY THIS EXISTS. Every acceptance layer this programme has built so far checks
that the product agreed with ITSELF: the spec named the right field, the receipt
recorded the right filter, the plan reconciled against the execution. All of
that is necessary and none of it is evidence that a NUMBER is right. The live
composition audit made the gap concrete — three correct figures, correct field
names, correct aggregations, computed over the whole book instead of the joint
one, and every check the estate had said complete.

So this module answers a different question:

    What is the mathematically correct answer on this dataset?

and it answers it with pandas and explicit column names. It does not import
`llm_query_parser`, `MIQuerySpec`, `QueryPlan`, the executor, the compiler or
the reconciler — asking the product to compute the expected answer would be the
product marking its own homework, and would have agreed with every historical
defect this estate has found.

IT IS DELIBERATELY SIMPLER THAN THE PRODUCT. No registry, no semantics, no
bucket engine, no percent-scale detection, no missing-value policy. A filter is
a boolean mask, a grouping is a `groupby`, an aggregation is a `sum` or a
weighted mean written out longhand. Where the product is cleverer than this,
that is exactly where the comparison is worth making.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

__all__ = ["canonical_book", "mask_for", "total", "weighted_average",
           "grouped", "row_count"]

BALANCE = "current_outstanding_balance"
LTV = "current_loan_to_value"
RATE = "current_interest_rate"
AGE = "youngest_borrower_age"


def canonical_book(n: int = 400, seed: int = 20260905) -> pd.DataFrame:
    """A deterministic book with pre-materialised buckets.

    The buckets are COLUMNS rather than something either side derives, so the
    comparison is about filtering, grouping and aggregation rather than about
    two bucket engines agreeing. Bucket derivation has its own tests; mixing the
    two would make a failure here ambiguous.
    """
    rng = np.random.default_rng(seed)
    ltv = rng.uniform(10, 90, n).round(1)
    age = rng.integers(60, 95, n)
    frame = pd.DataFrame({
        "loan_identifier": [f"L{i:05d}" for i in range(n)],
        BALANCE: rng.uniform(45_000, 520_000, n).round(2),
        LTV: ltv,
        RATE: rng.uniform(2.5, 9.5, n).round(2),
        AGE: age,
        "borrower_type": rng.choice(["Joint", "Single"], n),
        "collateral_geography": rng.choice(
            ["Scotland", "Wales", "North West", "London"], n),
        "erm_product_type": rng.choice(["Lump Sum", "Drawdown"], n),
        "broker_channel": rng.choice(["Alpha", "Beta", "Gamma"], n),
    })
    frame["ltv_bucket"] = pd.cut(
        ltv, bins=[0, 30, 40, 50, 60, 100],
        labels=["0-30%", "30-40%", "40-50%", "50-60%", "60%+"]).astype(str)
    frame["age_bucket"] = pd.cut(
        age, bins=[0, 70, 80, 90, 200],
        labels=["<70", "70-79", "80-89", "90+"]).astype(str)
    return frame


def mask_for(frame: pd.DataFrame,
             predicates: Sequence[Tuple[str, str, Any]]) -> pd.Series:
    """A boolean mask for ``(field, op, value)`` triples. Written longhand."""
    mask = pd.Series(True, index=frame.index)
    for field, op, value in predicates:
        column = frame[field]
        if op == "eq":
            mask &= column == value
        elif op == "ne":
            mask &= column != value
        elif op == "in":
            mask &= column.isin(list(value))
        elif op == "gt":
            mask &= column > value
        elif op == "ge":
            mask &= column >= value
        elif op == "lt":
            mask &= column < value
        elif op == "le":
            mask &= column <= value
        elif op == "between":
            low, high = value
            mask &= (column >= low) & (column <= high)
        else:
            raise ValueError(f"the oracle has no rule for {op!r}")
    return mask


def row_count(frame: pd.DataFrame, predicates=()) -> int:
    return int(mask_for(frame, predicates).sum())


def total(frame: pd.DataFrame, column: str, predicates=()) -> float:
    return float(frame.loc[mask_for(frame, predicates), column].sum())


def weighted_average(frame: pd.DataFrame, column: str, weight: str,
                     predicates=()) -> Optional[float]:
    """Σ(value × weight) / Σ(weight), written out rather than delegated."""
    rows = frame.loc[mask_for(frame, predicates)]
    weights = rows[weight].sum()
    if not weights:
        return None
    return float((rows[column] * rows[weight]).sum() / weights)


def grouped(frame: pd.DataFrame, by: Iterable[str], *,
            column: Optional[str] = None, how: str = "sum",
            predicates=()) -> Dict[Tuple[str, ...], float]:
    """``{group key tuple: figure}`` for one or more grouping columns.

    The key is always a tuple, including for a single axis, so a caller compares
    a 1-D and a 2-D result the same way.
    """
    rows = frame.loc[mask_for(frame, predicates)]
    keys = list(by)
    out: Dict[Tuple[str, ...], float] = {}
    for key, chunk in rows.groupby(keys, observed=True, dropna=False):
        key_tuple = key if isinstance(key, tuple) else (key,)
        key_tuple = tuple(str(k) for k in key_tuple)
        if how == "count":
            out[key_tuple] = float(len(chunk))
        elif how == "sum":
            out[key_tuple] = float(chunk[column].sum())
        elif how == "avg":
            out[key_tuple] = float(chunk[column].mean())
        elif how == "weighted_avg":
            weights = chunk[BALANCE].sum()
            out[key_tuple] = (float((chunk[column] * chunk[BALANCE]).sum()
                                    / weights) if weights else float("nan"))
        else:
            raise ValueError(f"the oracle has no rule for {how!r}")
    return out
