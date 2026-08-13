#!/usr/bin/env python3
"""``analytics_lib.stratify`` is vectorised, and must still give the old answer.

Why it was rewritten
--------------------
The original looped over groups in Python. That is O(groups) in interpreted
code, and the dimensions a *concentration* question asks about — borrower,
postcode, originator — are exactly the high-cardinality ones. Stratifying a
million loans by borrower (333,334 groups) took **61.7 seconds**; the
``concentration`` tool built on it took **166 seconds**. The slow path was the
common path.

Vectorised, the same call takes **2.06 seconds** — a 30x improvement, and the
concentration tool goes from 166 s to about 2 s.

Why this file exists
--------------------
A rewrite of a shared calculation is only safe if it is provably the same
calculation. ``_reference_stratify`` below is the ORIGINAL implementation,
preserved verbatim as an oracle, and every case here runs both and compares
whole frames. The awkward cases are the point: null dimension values, null
balances, zero weights, a metric the frame does not carry, a group whose weights
sum to zero. Each of those took a distinct branch in the loop, and a vectorised
rewrite is exactly where such a branch gets quietly dropped.

If a future change to ``stratify`` makes these diverge, the divergence is the
finding — not the test.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from analytics_lib.concentration import top_n_concentration
from analytics_lib.stratify import _apply_filters, stratify


# --------------------------------------------------------------------------- #
# The oracle: the implementation this replaced, verbatim
# --------------------------------------------------------------------------- #
def _reference_stratify(df, dimension, balance_col=None, *, count_col=None,
                        loan_id_col=None, filters=None, weighted_metrics=None,
                        weight_col=None, unknown_label="Unknown", dropna=False,
                        sort_by="balance_sum"):
    work = _apply_filters(df, filters).copy()
    dim = work[dimension]
    if dropna:
        work = work[dim.notna()]
    else:
        work[dimension] = dim.astype("object").where(dim.notna(), unknown_label)

    has_balance = balance_col is not None
    weight_col = weight_col or balance_col
    rows: List[Dict[str, Any]] = []

    for value, grp in work.groupby(dimension, dropna=False, sort=False):
        if loan_id_col and loan_id_col in grp.columns:
            loan_count = int(grp[loan_id_col].nunique())
        elif count_col and count_col in grp.columns:
            loan_count = int(pd.to_numeric(grp[count_col],
                                           errors="coerce").fillna(0).sum())
        else:
            loan_count = int(len(grp))

        row: Dict[str, Any] = {dimension: value, "loan_count": loan_count}
        if has_balance:
            bal = pd.to_numeric(grp[balance_col], errors="coerce")
            balance_sum = float(bal.sum())
            row["balance_sum"] = balance_sum
            row["avg_balance"] = balance_sum / loan_count if loan_count else 0.0

        for metric in (weighted_metrics or []):
            if metric not in grp.columns:
                row[f"{metric}_weighted_avg"] = float("nan")
                continue
            val = pd.to_numeric(grp[metric], errors="coerce")
            wt = (pd.to_numeric(grp[weight_col], errors="coerce")
                  if weight_col and weight_col in grp.columns else None)
            if wt is not None and float(wt.fillna(0).sum()) != 0:
                paired = pd.DataFrame({"v": val, "w": wt}).dropna()
                wsum = float(paired["w"].sum())
                row[f"{metric}_weighted_avg"] = (
                    float((paired["v"] * paired["w"]).sum()) / wsum
                    if wsum else float("nan"))
            else:
                row[f"{metric}_weighted_avg"] = float(val.mean())
        rows.append(row)

    result = pd.DataFrame(rows)
    if result.empty:
        return result
    if has_balance:
        total = float(result["balance_sum"].sum())
        result["balance_share"] = (result["balance_sum"] / total
                                   if total else 0.0)
        ordered = [dimension, "loan_count", "balance_sum", "balance_share",
                   "avg_balance"]
        ordered += [c for c in result.columns if c not in ordered]
        result = result[ordered]
    if sort_by in result.columns:
        result = result.sort_values(by=[sort_by, dimension],
                                    ascending=[False, True],
                                    kind="mergesort").reset_index(drop=True)
    else:
        result = result.sort_values(by=[dimension],
                                    kind="mergesort").reset_index(drop=True)
    return result


# --------------------------------------------------------------------------- #
# A frame built to hit every branch the loop had
# --------------------------------------------------------------------------- #
def _awkward_frame(n: int = 4_000) -> pd.DataFrame:
    rng = np.random.default_rng(11)
    frame = pd.DataFrame({
        "loan_identifier": [f"LN{i:06d}" for i in range(n)],
        "borrower_identifier": [f"BR{i // 3:06d}" for i in range(n)],
        # A null dimension value, which buckets to "Unknown" unless dropna.
        "region": [["SE", "LDN", "WAL", None][i % 4] for i in range(n)],
        "status": ["performing" if i % 5 else "arrears" for i in range(n)],
        "bal": np.where(rng.random(n) < 0.05, np.nan,
                        rng.uniform(1e4, 9e5, n)),      # null balances
        "ltv": np.where(rng.random(n) < 0.10, np.nan,
                        rng.uniform(20, 95, n)),        # null metric values
        "zero_w": 0.0,                                   # weights summing to zero
        "cnt": rng.integers(1, 4, n),
    })
    frame.loc[frame.index[:50], "bal"] = 0.0             # zero, not null
    return frame


CASES = [
    pytest.param(dict(dimension="region", balance_col="bal",
                      loan_id_col="loan_identifier", weighted_metrics=["ltv"]),
                 id="nulls-in-dimension-and-metric"),
    pytest.param(dict(dimension="region", balance_col="bal",
                      weighted_metrics=["ltv", "not_on_this_frame"]),
                 id="a-metric-the-frame-lacks"),
    pytest.param(dict(dimension="borrower_identifier", balance_col="bal",
                      loan_id_col="loan_identifier"),
                 id="high-cardinality"),
    pytest.param(dict(dimension="status", balance_col=None,
                      loan_id_col="loan_identifier"),
                 id="no-balance-column"),
    pytest.param(dict(dimension="status", balance_col="bal", count_col="cnt"),
                 id="explicit-count-column"),
    pytest.param(dict(dimension="region", balance_col="bal",
                      weighted_metrics=["ltv"], weight_col="zero_w"),
                 id="weights-sum-to-zero-falls-back-to-simple-mean"),
    pytest.param(dict(dimension="region", balance_col="bal", dropna=True,
                      loan_id_col="loan_identifier"),
                 id="dropna"),
    pytest.param(dict(dimension="region", balance_col="bal",
                      filters={"status": "performing"},
                      weighted_metrics=["ltv"]),
                 id="filtered"),
    pytest.param(dict(dimension="status", balance_col="bal",
                      sort_by="loan_count"),
                 id="alternate-sort"),
]


@pytest.mark.parametrize("kwargs", CASES)
def test_the_vectorised_form_gives_the_same_answer(kwargs):
    frame = _awkward_frame()
    expected = _reference_stratify(frame, **kwargs)
    actual = stratify(frame, **kwargs)
    pd.testing.assert_frame_equal(expected, actual, check_dtype=False,
                                  rtol=1e-12)


def test_an_empty_frame_is_still_empty():
    frame = _awkward_frame(0)
    assert stratify(frame, "region", "bal").empty


def test_a_single_group_still_holds_the_whole_balance():
    frame = _awkward_frame(500)
    frame["only"] = "one"
    table = stratify(frame, "only", "bal")
    assert len(table) == 1
    assert table.loc[0, "balance_share"] == pytest.approx(1.0)


def test_concentration_inherits_the_same_answer():
    """analytics_lib.concentration builds on stratify, so it has to move with
    it — and it is the caller that made the slow path matter."""
    frame = _awkward_frame()
    outcome = top_n_concentration(frame, "borrower_identifier", "bal", n=10,
                                  loan_id_col="loan_identifier")
    reference = _reference_stratify(frame, "borrower_identifier", "bal",
                                    loan_id_col="loan_identifier")
    total_count = float(reference["loan_count"].sum())
    reference["count_share"] = reference["loan_count"] / total_count
    top = reference.head(10)
    assert outcome["n_groups"] == len(reference)
    assert outcome["balance_concentration"] == pytest.approx(
        float(top["balance_share"].sum()))
    assert outcome["count_concentration"] == pytest.approx(
        float(top["count_share"].sum()))


# --------------------------------------------------------------------------- #
# The measurement that justified the rewrite
# --------------------------------------------------------------------------- #
def test_the_vectorised_form_is_materially_faster_on_a_wide_dimension(capsys):
    """The dimensions a concentration question asks about are the
    high-cardinality ones, so the O(groups) Python loop was the common path, not
    the edge case.

    Threshold deliberately loose (3x) — shared CI hardware, and the point is the
    order of the difference.
    """
    n = 60_000
    rng = np.random.default_rng(5)
    frame = pd.DataFrame({
        "borrower_identifier": [f"BR{i // 3:08d}" for i in range(n)],
        "loan_identifier": [f"LN{i:08d}" for i in range(n)],
        "bal": rng.uniform(1e4, 9e5, n),
        "ltv": rng.uniform(20, 95, n),
    })
    kwargs = dict(dimension="borrower_identifier", balance_col="bal",
                  loan_id_col="loan_identifier", weighted_metrics=["ltv"])

    t = time.perf_counter(); reference = _reference_stratify(frame, **kwargs)
    loop_ms = (time.perf_counter() - t) * 1000
    t = time.perf_counter(); actual = stratify(frame, **kwargs)
    vector_ms = (time.perf_counter() - t) * 1000

    pd.testing.assert_frame_equal(reference, actual, check_dtype=False,
                                  rtol=1e-12)
    with capsys.disabled():
        print(f"\n    {len(reference):,} groups over {n:,} rows"
              f"\n    per-group loop : {loop_ms:8.0f} ms"
              f"\n    vectorised     : {vector_ms:8.0f} ms"
              f"\n    improvement    : {loop_ms / max(vector_ms, 1e-6):8.1f}x")

    assert vector_ms < loop_ms / 3, (
        f"gained only {loop_ms / max(vector_ms, 1e-6):.1f}x")
