#!/usr/bin/env python3
"""Calibration: equivalent wording reaching the SAME existing construction.

CALIBRATION B — a nominal grouping request is the same request as `by <dim>`.
    "Show Application-stage pipeline BY LTV BAND" answered with five bands.
    "What is the LTV DISTRIBUTION of pipeline at Application?" answered
    `Weighted-average Current LTV: 55.9%` — one scalar for a question about a
    shape, with no warning that the axis had been dropped.

CALIBRATION A — an adjectival qualifier is a narrowing, not an axis.
    "What share of Offer-STAGE pipeline is joint borrowers?" was refused with
    "'share borrowers' is not a governed measure", while the identical question
    written "Offer pipeline" answered 59.68%. `share`, `percentage`,
    `proportion` and `fraction` failed together and worked together, which is
    what identifies the QUALIFIER rather than the share vocabulary as the
    cause: `pipeline_stage` became the grouping axis of a sentence that had
    already narrowed to one of its own values, and with a dimension set the
    share branch is never reached.

The share vocabulary needed no change and received none. These tests pin that:
all four words answer, and they answer the same figure.

Half of this file is about what must NOT move — an explicit `by <dimension>`
grouping, a two-dimensional grouping, the percentage-valued measures, and the
whole-book share denominator.

The oracle is arithmetic on the governed prepared frame — see
``scripts/prove_multivariate_pipeline_fixture.py``:

    Offer total       4,960,000 · 10 cases      Offer joint      2,960,000 · 6
    Application total 4,345,000 · 12 cases      App joint WA LTV    53.671%
    Offer WA LTV         58.383%                Offer by region  six regions
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mi_agent_api.tests.test_multivariate_two_defect_fix import (  # noqa: E402
    answer, ask)

#: The same business question, four ways of naming a ratio of two populations.
SHARE_WORDS = ("share", "percentage", "proportion", "fraction")


# --------------------------------------------------------------------------- #
# CALIBRATION A — the qualifier
# --------------------------------------------------------------------------- #
class TestAdjectivalQualifierIsANarrowing(unittest.TestCase):

    def test_every_share_word_answers_the_governed_figure(self):
        """The vocabulary was never the gap, and this is the evidence."""
        for word in SHARE_WORDS:
            with self.subTest(word=word):
                text = answer("What %s of Offer-stage pipeline is joint "
                              "borrowers?" % word)
                self.assertIn("59.7", text)
                self.assertNotIn("19.2", text)

    def test_the_qualifier_spelling_does_not_change_the_answer(self):
        """"Offer pipeline", "Offer-stage pipeline" and "Offer stage pipeline"
        are one question. Before the calibration only the first answered."""
        figures = set()
        for population in ("Offer pipeline", "Offer-stage pipeline",
                           "Offer stage pipeline"):
            text = answer("What share of %s is joint borrowers?" % population)
            self.assertIn("59.7", text)
            figures.add("59.7" in text)
        self.assertEqual(figures, {True})

    def test_the_qualified_field_binds_as_a_filter_not_as_an_axis(self):
        """`pipeline_stage` is the narrowing. A degenerate one-group axis over
        the field the sentence already filtered on is not a breakdown."""
        text = answer("What is the Offer-stage balance for joint borrowers?")
        self.assertIn("£3.0MM", text)
        self.assertNotIn("grouped by Pipeline Stage", text)

    def test_the_denominator_is_the_qualified_population(self):
        text = answer("What percentage of Offer-stage pipeline is joint "
                      "borrowers?")
        self.assertIn("Population Total: 10", text)
        self.assertNotIn("Population Total: 40", text)


class TestTheQualifierRuleIsGovernedNotHardCoded(unittest.TestCase):

    def test_no_stage_name_appears_in_the_calibration(self):
        """PORTABILITY. A lender that spells its stages differently gets the
        same behaviour, because the rule asks the value catalogue."""
        source = (_REPO_ROOT / "mi_agent" / "llm_query_parser.py").read_text(
            encoding="utf-8")
        start = source.index("def _qualified_by_own_value")
        body = source[start:source.index("\ndef ", start + 10)]
        for spelling in ("offer", "kfi", "application", "completed",
                         "withdrawn", "london", "joint"):
            with self.subTest(spelling=spelling):
                self.assertNotIn('"%s"' % spelling, body.lower())
                self.assertNotIn("'%s'" % spelling, body.lower())

    def test_a_qualifier_from_another_dimension_does_not_collapse_the_axis(self):
        """The word in front must be a value OF THAT SAME FIELD. "London" is a
        region, so it cannot make `pipeline_stage` a qualifier."""
        from mi_agent.llm_query_parser import _qualified_by_own_value

        values = {"pipeline_stage": {"offer": "OFFER"},
                  "geographic_region_obligor": {"london": "London"}}
        self.assertTrue(_qualified_by_own_value(
            "offer-stage pipeline", "stage", "pipeline_stage", values))
        self.assertFalse(_qualified_by_own_value(
            "london stage pipeline", "stage", "pipeline_stage", values))
        self.assertFalse(_qualified_by_own_value(
            "pipeline by stage", "stage", "pipeline_stage", values))


# --------------------------------------------------------------------------- #
# WHAT MUST NOT MOVE
# --------------------------------------------------------------------------- #
class TestExplicitGroupingIsUntouched(unittest.TestCase):

    def test_a_term_after_an_axis_marker_still_groups(self):
        """"by stage" is a grouping clause; the qualifier rule never reads it."""
        text = answer("Show pipeline by stage and borrower type.")
        self.assertIn("grouped by Pipeline Stage and Borrower Type", text)

    def test_a_two_dimensional_grouping_keeps_both_dimensions(self):
        env = ask("Show me total balance by region and product.")
        self.assertTrue(env.get("ok"))
        self.assertIn("and", env.get("answer") or "")

    def test_an_explicit_by_dimension_grouping_still_groups(self):
        text = answer("Break down Offer pipeline by region.")
        self.assertIn("grouped by Region", text)

    def test_a_qualified_stage_still_groups_by_the_dimension_asked_for(self):
        """The qualifier is dropped; the REQUESTED axis is not."""
        text = answer("Show Offer-stage pipeline by region.")
        self.assertIn("grouped by Region", text)
        self.assertIn("Pipeline Stage = OFFER", text)


class TestPercentageValuedMeasuresAreNotStolen(unittest.TestCase):

    def test_weighted_average_ltv_is_still_a_weighted_average(self):
        text = answer("What is WA LTV for Offer-stage pipeline?")
        self.assertIn("58.4", text)
        self.assertNotIn("53.1", text)   # the UNWEIGHTED mean
        self.assertIn("Weighted-average", text)

    def test_the_hardest_working_construction_is_unchanged(self):
        """stage + borrower type + balance-weighted LTV, all three bound."""
        text = answer("What is WA LTV for joint borrowers in Application?")
        self.assertIn("53.7", text)
        self.assertIn("Borrower Type = Joint", text)
        self.assertIn("Pipeline Stage = APPLICATION", text)

    def test_an_interest_rate_is_not_read_as_a_share(self):
        env = ask("What percentage is the interest rate?")
        self.assertNotIn("Share Pct", env.get("answer") or "")

    def test_the_conversion_rate_keeps_its_own_route(self):
        env = ask("What is the conversion rate?")
        self.assertEqual((env.get("metadata") or {}).get("route"),
                         "cohort_conversion")


class TestWholeBookShareDenominatorIsUnchanged(unittest.TestCase):

    def test_a_whole_book_share_still_divides_by_the_whole_book(self):
        text = answer("What proportion of the book is drawdown?")
        self.assertIn("Population Total: 640", text)

    def test_a_whole_book_value_share_is_unchanged(self):
        text = answer("What share of the balance is in Scotland?")
        self.assertIn("Population Total: 640", text)

    def test_a_whole_book_threshold_share_is_unchanged(self):
        text = answer("What proportion of the book is above 60% LTV?")
        self.assertIn("Population Total: 640", text)


# --------------------------------------------------------------------------- #
# CALIBRATION B — the shape noun
# --------------------------------------------------------------------------- #
class TestNominalGroupingReachesTheExistingConstruction(unittest.TestCase):

    def test_a_distribution_is_grouped_not_scalar(self):
        text = answer("What is the LTV distribution of pipeline currently at "
                      "Application?")
        self.assertIn("grouped by LTV Bucket", text)
        self.assertIn("Pipeline Stage = APPLICATION", text)

    def test_it_reaches_the_same_axis_the_explicit_wording_reaches(self):
        """"LTV distribution" and "by LTV band" are one request."""
        for question in ("What is the LTV distribution of pipeline at "
                         "Application?",
                         "Show Application-stage pipeline by LTV band."):
            with self.subTest(question=question):
                self.assertIn("grouped by LTV Bucket", answer(question))

    def test_the_axis_is_a_governed_bucket_dimension(self):
        """Dimension-grounded: the promotion is the existing governed map, and
        a measure with no bucket dimension is never given a manufactured one."""
        from mi_agent.llm_query_parser import _NUMERIC_AXIS_BUCKET

        self.assertEqual(_NUMERIC_AXIS_BUCKET["ltv"], "ltv_bucket")
        self.assertEqual(_NUMERIC_AXIS_BUCKET["age"], "age_bucket")


class TestTheShapeNounDoesNotBroaden(unittest.TestCase):

    def test_a_sentence_that_names_its_own_axis_is_left_alone(self):
        """"by region" and "across regions" already say where to group. The
        shape noun must not displace or duplicate the axis the reader asked
        for, so it never fires when an axis marker is present."""
        for question in ("What is the balance distribution by region?",
                         "What is the distribution of balance across regions?"):
            with self.subTest(question=question):
                env = ask(question)
                self.assertNotIn("LTV Bucket", env.get("answer") or "")
                self.assertNotIn("Ticket", env.get("answer") or "")

    def test_spread_is_not_a_shape_noun(self):
        """MEASURED, AND THE REASON THE LIST IS SHORT. In lending a spread is a
        governed rate concept; the calibration is not entitled to the word."""
        from mi_agent.llm_query_parser import _SHAPE_NOUNS

        self.assertNotIn("spread", _SHAPE_NOUNS)
        env = ask("Show the LTV spread of Offer pipeline.")
        self.assertNotIn("grouped by", env.get("answer") or "")

    def test_a_scalar_measure_is_still_a_scalar(self):
        """No shape noun, no axis. This is the line the calibration walks."""
        for question in ("What is the average LTV?",
                         "What is WA LTV for Offer-stage pipeline?"):
            with self.subTest(question=question):
                text = answer(question)
                self.assertNotIn("grouped by", text)
                self.assertIn("Weighted-average", text)

    def test_the_shape_noun_must_stand_directly_after_the_measure(self):
        """Only a determiner may intervene, so an unrelated later "split"
        cannot reach back and claim a measure."""
        from mi_agent.llm_query_parser import _asks_for_a_shape

        self.assertTrue(_asks_for_a_shape("what is the ltv distribution?", "ltv"))
        self.assertTrue(_asks_for_a_shape("show the ltv breakdown", "ltv"))
        self.assertFalse(_asks_for_a_shape(
            "what is the ltv for loans we split out?", "ltv"))
        self.assertFalse(_asks_for_a_shape("what is the average ltv?", "ltv"))


if __name__ == "__main__":
    unittest.main()
