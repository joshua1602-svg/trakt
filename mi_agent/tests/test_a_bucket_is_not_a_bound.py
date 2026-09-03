"""A bucket NAME is not a threshold.

THE DEFECT, from the 2026-09-03 replay of live questions. "How many loans are
in the 60-70% LTV bucket" was refused with "the threshold was not applied to
the calculation" — while the executor's own ledger recorded
``parsed_filters: [ltv_bucket]`` and ``applied_filters: [ltv_bucket]``. The
narrowing HAD run and the answer was right; the receipt threw it away.

`_PCT_BOUND_RE` exists for "a 75% LTV cap", where a bound is written with no
comparator. It was also matching the UPPER EDGE of a bucket label — the "70%"
of "60-70%" — and raising a cap nobody asked for. Nothing could then satisfy
it, because the engine had (correctly) filtered on a bucket rather than
applied a comparison.

The fix is to stop detecting the bound, not to teach the guard a new way to
prove one: the reader never asked for a cap, so there is nothing to reconcile.
Two marks in the question itself say so — the range dash before the number,
and a bucket word after it.
"""
from __future__ import annotations

import unittest

from mi_agent import execution_receipt as R


def _thresholds(q):
    return [f.label for f in R._detect_thresholds(q)]


class TestABucketIsNotABound(unittest.TestCase):

    def test_the_upper_edge_of_a_range_is_not_a_cap(self):
        self.assertEqual(_thresholds("How many loans are in the 60-70% LTV bucket"), [])

    def test_a_range_without_the_word_bucket_is_still_not_a_cap(self):
        self.assertEqual(_thresholds("balance for 60-70% LTV"), [])

    def test_a_bucket_word_after_a_single_percentage_is_not_a_cap(self):
        self.assertEqual(_thresholds("Show me the 70% LTV bucket"), [])

    def test_band_and_bracket_read_the_same_way(self):
        self.assertEqual(_thresholds("the 60-70% LTV band"), [])
        self.assertEqual(_thresholds("80-90% loan-to-value bracket"), [])

    def test_an_en_dash_range_reads_the_same_way(self):
        self.assertEqual(_thresholds("loans in the 60–70% LTV bucket"), [])


class TestARealBoundStillRefuses(unittest.TestCase):
    """The reason this detector exists. Removing the false positive must not
    remove the true one, or a cap the reader asked for goes unreconciled."""

    def test_a_written_cap_is_still_a_threshold(self):
        self.assertEqual(_thresholds("loans with a 75% LTV cap"),
                         ["LTV bound of 75%"])

    def test_a_comparator_bound_is_still_a_threshold(self):
        self.assertTrue(_thresholds("loans above 70% LTV"))

    def test_a_bare_percentage_of_ltv_is_still_a_threshold(self):
        self.assertEqual(_thresholds("balance at 80% LTV"),
                         ["LTV bound of 80%"])

    def test_a_three_digit_bound_is_not_truncated(self):
        self.assertEqual(_thresholds("loans at 110% LTV"),
                         ["LTV bound of 110%"])


if __name__ == "__main__":
    unittest.main()
