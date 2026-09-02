#!/usr/bin/env python3
"""An aggregation over no rows refuses in words; it does not raise TypeError.

WHY THIS FILE EXISTS. Captured live on 2026-09-02, running the accepted bank
against the Direct book:

    File "mi_agent/mi_query_executor.py", line 293, in aggregate_series
        return float(vals.mean())
    TypeError: float() argument must be a string or a real number,
               not 'NAType'

`coerce_numeric` returns a pandas NULLABLE dtype. A nullable series that is
empty — or all-null — aggregates to `pd.NA`, not to `nan`, and `float(pd.NA)`
raises `TypeError`. A TypeError is not an `MIQueryExecutionError`, so it slipped
past the executor's own handling and surfaced as "The MI Agent could not
complete this query": a crash where the truth was a FACT about the book.

The question was "how much outstanding balance do we have where borrower age
exceeds 75 and LTV is over 40%?" The Direct portfolio holds no loans above 40%
LTV — established separately — so the filter legitimately matched nothing. Its
near-identical sibling refused properly and named the filter, because it
happened to route somewhere that checked first. Same question, two phrasings,
one crash.

The message says "no rows" on purpose: `mi_service._error_code_for` reads that
phrase out of the validation errors and classifies the outcome
NO_MATCHING_RECORDS — what happened — rather than CALCULATION_FAILED, which is
what a fault looks like.
"""
from __future__ import annotations

import unittest

import pandas as pd

from mi_agent.mi_query_executor import MIQueryExecutionError, aggregate_series

#: A frame shaped like the governed tape: nullable dtypes, which is what makes
#: an empty aggregate `pd.NA` instead of `nan`.
EMPTY = pd.DataFrame({"balance": pd.array([], dtype="Float64"),
                      "weight": pd.array([], dtype="Float64")})
ALL_NULL = pd.DataFrame({"balance": pd.array([None, None], dtype="Float64"),
                         "weight": pd.array([None, None], dtype="Float64")})
REAL = pd.DataFrame({"balance": pd.array([10.0, 20.0, 30.0], dtype="Float64"),
                     "weight": pd.array([1.0, 1.0, 2.0], dtype="Float64")})


class TestNoRowsIsARefusalNotACrash(unittest.TestCase):

    def test_the_aggregation_that_crashed_the_live_run(self):
        with self.assertRaises(MIQueryExecutionError) as caught:
            aggregate_series(EMPTY, "balance", "avg")
        self.assertIn("no rows", str(caught.exception),
                      "the message must carry 'no rows' — mi_service reads it "
                      "to classify the outcome NO_MATCHING_RECORDS")

    def test_every_scalar_aggregation_is_covered(self):
        """A guard on four of five sites is the shape this defect had."""
        for aggregation in ("avg", "median", "min", "max"):
            with self.subTest(aggregation=aggregation):
                with self.assertRaises(MIQueryExecutionError):
                    aggregate_series(EMPTY, "balance", aggregation)

    def test_a_column_that_is_present_but_all_null(self):
        """Rows exist; the measure does not. Still undefined, still not a crash."""
        for aggregation in ("avg", "median", "min", "max"):
            with self.subTest(aggregation=aggregation):
                with self.assertRaises(MIQueryExecutionError):
                    aggregate_series(ALL_NULL, "balance", aggregation)

    def test_it_is_never_a_bare_TypeError(self):
        """The distinction that let this escape: TypeError is not handled upstream."""
        try:
            aggregate_series(EMPTY, "balance", "avg")
        except MIQueryExecutionError:
            pass
        except TypeError as exc:            # pragma: no cover - the defect
            self.fail("still raising TypeError: %s" % exc)


class TestRealDataIsUntouched(unittest.TestCase):
    """The guard must change nothing about an answer that had rows."""

    def test_the_arithmetic_is_unchanged(self):
        self.assertEqual(aggregate_series(REAL, "balance", "sum"), 60.0)
        self.assertEqual(aggregate_series(REAL, "balance", "avg"), 20.0)
        self.assertEqual(aggregate_series(REAL, "balance", "median"), 20.0)
        self.assertEqual(aggregate_series(REAL, "balance", "min"), 10.0)
        self.assertEqual(aggregate_series(REAL, "balance", "max"), 30.0)

    def test_a_count_over_no_rows_is_still_zero(self):
        """Counting nothing IS zero. Averaging nothing is not."""
        self.assertEqual(aggregate_series(EMPTY, None, "count"), 0)

    def test_an_empty_sum_keeps_its_existing_answer(self):
        """pandas sums an empty series to 0, and that behaviour is not changed
        here: this fix converts a crash, it does not relitigate a convention."""
        self.assertEqual(aggregate_series(EMPTY, "balance", "sum"), 0.0)

    def test_weighted_average_over_no_rows_keeps_its_own_guard(self):
        """It already returned nan on a zero denominator; that path is untouched."""
        out = aggregate_series(EMPTY, "balance", "weighted_avg",
                               weight_col="weight")
        self.assertNotEqual(out, out)      # nan


if __name__ == "__main__":
    unittest.main()
