"""A grain the series does not have is a different question, not a limit.

REVERSED 2026-09-04, and the reasoning it replaces is kept below because it was
a real trade rather than an oversight.

THE RULING NOW. Time grain is a MATERIAL semantic facet. A response may remain
PARTIAL only where the omitted element cannot change the population, the
measure, the comparison basis, the period or the economic interpretation. A
grain changes two of them: "month on month" and "week on week" compare
different spans across different boundaries, and a monthly improvement is not a
weekly one. So a requested grain the series cannot express REFUSES.

WHAT THIS FILE USED TO ARGUE, and what was measured for it. From the 2026-09-03
replay, "Has pipeline progression improved month on month?" was refused
outright. The pipeline is published WEEKLY, the route said so correctly
(`seriesGrain: week`), and the receipt treated the mismatch the way it treats a
spatial one — as a substitution that blocks. The objection was that no monthly
pipeline series had been passed over in favour of a weekly one: the weekly
movement was computed, correct, and thrown away, and the reader got nothing
where they could have had the real figure and a sentence saying what period it
covers. `KIND_TIME_GRAIN` was moved to `SHAPE_FACETS` on that basis.

WHY THAT DID NOT HOLD. The 2026-09-04 bank shows the consequence: the question
came back ANSWERED, verdict `partial`, with "month — this answer is reported at
week level, not by month" printed UNDERNEATH the figure. The disclosure was
real and the answer still answered a question nobody asked. Disclosure is not
honouring — the same rule already settled for populations and for periods.

WHAT MUST NOT CHANGE, and is pinned below: a question asking for the grain the
series publishes still answers, silently and completely. Refusing sound weekly
questions to make monthly ones honest would trade one wrong answer for many
missing ones.

    KIND_GRANULARITY   place  "by postcode" answered at ITL3 area level is a
                              different number for a different question — blocks
    KIND_TIME_GRAIN    time   a weekly series answering a monthly question is a
                              different comparison basis — blocks
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

    def test_both_grains_are_now_the_blocking_kind(self):
        """They are still DIFFERENT kinds — the builders and the wording differ
        — but they now earn the same verdict."""
        self.assertIn(R.KIND_GRANULARITY, R.NUMBER_OR_SUBJECT_FACETS)
        self.assertIn(R.KIND_TIME_GRAIN, R.NUMBER_OR_SUBJECT_FACETS)
        self.assertNotIn(R.KIND_TIME_GRAIN, R.SHAPE_FACETS)


class TestAWeeklySeriesRefusesAndSaysWhy(unittest.TestCase):

    def test_a_monthly_question_on_a_weekly_series_is_refused(self):
        verdict, _ = _assess([_facet(R.KIND_TIME_GRAIN, "month", "week")])
        self.assertEqual(verdict, R.VERDICT_REFUSE)

    def test_the_reader_is_told_which_grain_they_could_not_have(self):
        """The refusal must still name both grains. "I cannot answer that" with
        no reason is a worse outcome than the partial it replaces."""
        _, message = _assess([_facet(R.KIND_TIME_GRAIN, "month", "week")])
        self.assertIn("week", message)
        self.assertIn("month", message)
        self.assertIn("not substituted a broader figure", message)

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
