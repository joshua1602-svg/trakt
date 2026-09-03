"""A grain the series does not have is a limit, not a substitution.

THE DEFECT, from the 2026-09-03 replay. "Has pipeline progression improved
month on month?" was refused outright. The pipeline is published WEEKLY, the
route said so correctly (`seriesGrain: week`), and the receipt then treated the
mismatch the way it treats a spatial one — as a substitution that blocks. But
there is no monthly pipeline series that was passed over in favour of a weekly
one. The weekly movement was computed, correct, and thrown away, and the reader
got nothing where they could have had the real figure and a sentence saying
what period it covers.

The two grains are now different kinds, because they deserve different
verdicts:

    KIND_GRANULARITY   place   "by postcode" answered at ITL3 area level is a
                               DIFFERENT NUMBER for a different question — blocks
    KIND_TIME_GRAIN    time    a weekly series answering a monthly question is
                               the REAL number over a stated window — discloses

What must not change is that the reader is told. A weekly figure delivered
silently to a monthly question is the failure the blocking rule was protecting
against, and it is still a failure.
"""
from __future__ import annotations

import unittest

from mi_agent import execution_receipt as R


def _facet(kind, asked, reported, *, applied=False):
    """A facet as the RECONCILER leaves it.

    `assess` reads a reconciled receipt; it does not reconcile. So the status
    and the reason are stamped here exactly as the granularity branch stamps
    them, and the first test below pins that this is what the branch produces.
    """
    if applied:
        return R.RequestedFacet(kind=kind, label=asked,
                                concepts=(asked, reported), status=R.APPLIED)
    return R.RequestedFacet(
        kind=kind, label=asked, concepts=(asked, reported),
        status=R.UNSUPPORTED,
        reason="this answer is reported at %s level, not by %s" % (reported,
                                                                   asked))


def _assess(facets):
    return R.assess(R.ExecutionReceipt(facets=list(facets)))


class TestTheTwoGrainsAreDifferentKinds(unittest.TestCase):

    def test_the_time_axis_builder_makes_a_time_grain(self):
        facet = R.time_axis_disclosure(
            "month", "evolution_pipeline_stage",
            envelope={"metadata": {"seriesGrain": "week"}})
        self.assertIsNotNone(facet)
        self.assertEqual(facet.kind, R.KIND_TIME_GRAIN)
        self.assertEqual(facet.concepts, ("month", "week"))

    def test_a_place_grain_is_still_the_blocking_kind(self):
        self.assertIn(R.KIND_GRANULARITY, R.NUMBER_OR_SUBJECT_FACETS)
        self.assertNotIn(R.KIND_TIME_GRAIN, R.NUMBER_OR_SUBJECT_FACETS)


class TestAWeeklySeriesAnswersAndSaysSo(unittest.TestCase):

    def test_a_monthly_question_on_a_weekly_series_is_not_refused(self):
        verdict, message = _assess([_facet(R.KIND_TIME_GRAIN, "month", "week")])
        self.assertEqual(verdict, R.VERDICT_PARTIAL)
        self.assertNotEqual(verdict, R.VERDICT_REFUSE)

    def test_the_reader_is_told_which_grain_they_got(self):
        """Silently handing a weekly figure to a monthly question is the
        failure the blocking rule existed to stop, and still is."""
        _, message = _assess([_facet(R.KIND_TIME_GRAIN, "month", "week")])
        self.assertIn("week", message)
        self.assertIn("month", message)

    def test_a_matching_grain_says_nothing_at_all(self):
        verdict, message = _assess(
            [_facet(R.KIND_TIME_GRAIN, "week", "week", applied=True)])
        self.assertEqual(verdict, R.VERDICT_OK)
        self.assertIsNone(message)

    def test_the_disclosure_is_the_one_the_reconciler_stamps(self):
        """Not a string invented by this test: the branch that adjudicates a
        spatial grain now adjudicates a temporal one, and this is its wording."""
        import inspect
        source = inspect.getsource(R)
        self.assertIn("elif facet.kind in (KIND_GRANULARITY, KIND_TIME_GRAIN):",
                      source)
        self.assertIn('"this answer is reported at %s level, not by %s"', source)


class TestAPlaceGrainStillRefuses(unittest.TestCase):
    """Removing one refusal must not remove the other. A postcode question
    answered at area level IS a different number."""

    def test_a_spatial_mismatch_still_blocks(self):
        verdict, message = _assess(
            [_facet(R.KIND_GRANULARITY, "postcode", "ITL3 area")])
        self.assertEqual(verdict, R.VERDICT_REFUSE)
        self.assertIn("not substituted a broader figure", message)

    def test_a_spatial_mismatch_beside_a_time_one_still_blocks(self):
        verdict, _ = _assess([_facet(R.KIND_TIME_GRAIN, "month", "week"),
                              _facet(R.KIND_GRANULARITY, "postcode", "ITL3 area")])
        self.assertEqual(verdict, R.VERDICT_REFUSE)


if __name__ == "__main__":
    unittest.main()
