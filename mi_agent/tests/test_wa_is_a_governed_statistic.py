#!/usr/bin/env python3
""""WA" is the estate's own abbreviation for a weighted average. Nobody owned it.

THE DEFECT, and it is the quietest kind: a correct answer for no reason.

    "WA LTV"  ->  weighted_avg(current_loan_to_value)   correct

but not because anything read "WA". `statistic_named("WA LTV")` returned None,
`_aggregation_intent("wa ltv")` returned None, and the measure owner matched only
"ltv". The weighted average arrived because it is the REGISTRY DEFAULT for a
percent metric — the same answer "LTV" alone produces. Twenty-seven questions in
the estate's own corpus use the abbreviation, and the statistic they name was
never accounted for.

WHY THAT MATTERS EVEN THOUGH THE FIGURE IS RIGHT. Nothing raises a statistic
facet, so nothing can reconcile the requested statistic against the executed one.
The day a field's default changes, or the abbreviation is used on a field whose
default is a plain mean, the answer changes and no guard notices — the reader
asked for a weighted average and the receipt has no record that they did. A
correct answer by coincidence is semantic debt, not a passing test.

THE OWNER. `mi_agent.statistic` holds the governed statistic vocabulary, and
"WA" belongs beside "weighted average" and "exposure-weighted" in it. It is a
lexical alias for a statistic, so:

  * it occupies the STATISTIC role and nothing else — the metric still comes
    from the measure owner, so "WA LTV" is LTV and "WA rate" is the rate;
  * it matches on word boundaries only, never as a substring, so "Wales",
    "warehouse" and "software" are untouched;
  * it is case-insensitive, because readers write "WA", "wa" and "Wa".

Deliberately NOT added: any other abbreviation. This one is in the estate's own
question corpus twenty-seven times; inventing a table of plausible abbreviations
to raise the answer rate is how vocabulary stops meaning anything.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import mi_agent.statistic as statistic                                  # noqa: E402


class TestWAIsWeightedAverage(unittest.TestCase):

    def test_the_statistic_owner_reads_it(self):
        for text in ("WA LTV", "wa ltv", "Wa LTV", "What is WA LTV?",
                     "Show WA LTV by region."):
            with self.subTest(text=text):
                self.assertEqual(statistic.statistic_named(text), "weighted_avg")

    def test_it_agrees_with_the_spelled_out_form(self):
        """The abbreviation and the phrase are one request, so they must produce
        one statistic."""
        for short, long in (("WA LTV", "weighted average LTV"),
                            ("WA rate", "weighted average rate")):
            with self.subTest(short=short):
                self.assertEqual(statistic.statistic_named(short),
                                 statistic.statistic_named(long))


class TestItIsATokenNotASubstring(unittest.TestCase):
    """The negative controls. A two-letter alias is exactly the kind that
    silently eats other words if it is matched loosely."""

    def test_it_never_matches_inside_another_word(self):
        for text in ("balance in Wales", "warehouse loans", "software vendor",
                     "Swansea balance", "the Warwick broker", "wallet share",
                     "how many loans await completion"):
            with self.subTest(text=text):
                self.assertIsNone(statistic.statistic_named(text),
                                  "a substring was read as a statistic")

    def test_ordinary_prose_names_no_statistic(self):
        for text in ("give me the total balance", "how many loans are there",
                     "balance by region", "show me the book"):
            with self.subTest(text=text):
                self.assertIsNone(statistic.statistic_named(text))


class TestItOccupiesTheStatisticRoleOnly(unittest.TestCase):
    """"WA" says HOW to aggregate. It must not become part of the measure, the
    dimension or the population."""

    def test_the_measure_still_comes_from_the_measure_owner(self):
        from mi_agent.llm_query_parser import _detect_metric
        from mi_agent.mi_query_validator import load_mi_semantics

        semantics = load_mi_semantics(
            str(_REPO_ROOT / "mi_agent" / "mi_semantics_field_registry.yaml"))
        self.assertEqual(_detect_metric("wa ltv", semantics)[0],
                         "current_loan_to_value")

    def test_the_statistic_is_read_and_the_answer_is_still_the_weighted_average(self):
        """End to end, so the alias is proven to reach a spec rather than only a
        vocabulary table."""
        import mi_agent.execution_receipt as receipt
        from mi_agent.llm_query_parser import _deterministic_parse
        from mi_agent.mi_query_validator import load_mi_semantics
        from mi_agent.tests import portfolio_truth_oracle as truth

        semantics = load_mi_semantics(
            str(_REPO_ROOT / "mi_agent" / "mi_semantics_field_registry.yaml"))
        book = truth.canonical_book()
        spec, _meta = _deterministic_parse(
            "WA LTV", semantics, available_columns=receipt.book_columns(book),
            available_values=receipt.book_values(book, semantics))
        self.assertEqual(spec.metric, "current_loan_to_value")
        self.assertEqual(spec.aggregation, "weighted_avg")


if __name__ == "__main__":
    unittest.main()
