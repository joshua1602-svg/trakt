#!/usr/bin/env python3
""""Balance over time" names the balance. It is not a predicate about balance.

THE REGRESSION, found by comparing this branch against MAIN rather than against
its own parent.

    "Compare balance over time"
        main   -> current_outstanding_balance
        branch -> no metric, and the question refuses

Every governed measure was affected, not just the balance: `ltv over time`,
`interest rate over time`, `exposure over time`, `loan count over time` and
`weighted average ltv over time` all stopped naming a measure.

HOW IT HAPPENED, and neither half was wrong on its own.

`is_filter_subject` owns the question "is this span the subject of a predicate".
Its own docstring states the contract — "both its patterns require a comparator,
a `<>=` symbol or a leading digit" — and lists the measured cases. But one of its
alternatives matches a COMPARATOR WITH NOTHING AFTER IT, so "over" alone was
enough:

    "balance over 50%"     the comparator has a bound   -> a predicate    ✓
    "balance over time"    the comparator has no bound  -> a predicate    ✗

That over-claim is on main too. It was DORMANT there, because `_detect_metric`
never asked the role check: only `_measure_hits` did. Wiring the check into
`_detect_metric` was right — it is what stopped "balance by ltv bucket" reading
the AXIS as the measure — and it is what made the latent over-claim visible.

So the repair belongs to the role owner, not to the measure owner and not to the
sentence: a comparator marks a filter subject only where a BOUND follows it.
"Time" is not a bound; it is the axis the measure is plotted against.

THE DISTINCTION THAT MUST SURVIVE. A question naming a measure and a temporal
axis names that measure. A question naming NO measure still defaults, and the
default is still recorded as a substitution — the two readings are different
requests and this file pins both.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mi_agent.llm_query_parser import _detect_metric                    # noqa: E402
from mi_agent.mi_query_validator import load_mi_semantics               # noqa: E402
from question_interpretation.lexical import is_filter_subject           # noqa: E402

_SEMANTICS = load_mi_semantics(
    str(_REPO_ROOT / "mi_agent" / "mi_semantics_field_registry.yaml"))

#: Governed measures, named the way a reader names them. The expectation is
#: never written out: it is whatever the measure resolves to WITHOUT the
#: temporal phrase, which is what makes this a property rather than a table.
MEASURES = ("balance", "ltv", "interest rate", "exposure", "loan count",
            "valuation", "weighted average ltv")

#: Temporal language that follows a measure. None of it is a bound.
TEMPORAL = ("over time", "over the last six months", "over the last year",
            "month by month")


class TestTemporalLanguageDoesNotConsumeTheMeasure(unittest.TestCase):

    def test_the_measure_is_the_same_with_and_without_the_temporal_phrase(self):
        """Compared as (metric, aggregation), not metric alone: a COUNT is a
        governed measure whose `metric` is legitimately None — what identifies
        it is the aggregation — and asserting a non-None metric would exclude
        the very measure the trend work was about."""
        for measure in MEASURES:
            plain = _detect_metric(measure, _SEMANTICS)
            self.assertTrue(
                plain[2], f"{measure!r} matches no term; the fixture is wrong")
            for temporal in TEMPORAL:
                with self.subTest(measure=measure, temporal=temporal):
                    resolved = _detect_metric(f"{measure} {temporal}", _SEMANTICS)
                    self.assertTrue(resolved[2],
                                    "temporal language consumed the measure")
                    self.assertEqual(resolved[:2], plain[:2],
                                     "temporal language changed the measure")

    def test_the_sentences_from_the_regression(self):
        for question in ("Compare balance over time",
                         "Show balance over time",
                         "Balance over time",
                         "Compare balance over time for direct and acquired",
                         "Compare balances over time for direct and acquired"):
            with self.subTest(question=question):
                self.assertEqual(_detect_metric(question.lower(), _SEMANTICS)[0],
                                 "current_outstanding_balance")

    def test_a_count_over_time_is_still_a_count(self):
        for question in ("show loan count over time", "loan count over time"):
            with self.subTest(question=question):
                self.assertEqual(_detect_metric(question, _SEMANTICS)[:2],
                                 _detect_metric("loan count", _SEMANTICS)[:2])


class TestABoundIsStillABound(unittest.TestCase):
    """The negative controls. The role check exists to stop a THRESHOLD SUBJECT
    being read as the measure, and every one of those must still be caught."""

    PREDICATES = ("balance over 50000", "balance over £100k", "ltv over 50%",
                  "ltv above 40%", "balance below 25000",
                  "rate of more than 7%", "ltv greater than or equal to 60%",
                  "age over 80", "balance over 1m")

    def test_a_measure_followed_by_a_bound_is_a_filter_subject(self):
        for text in self.PREDICATES:
            subject = text.split(" ")[0]
            with self.subTest(text=text):
                self.assertTrue(
                    is_filter_subject(text, 0, len(subject)),
                    "a threshold subject stopped being recognised as one")

    def test_and_therefore_does_not_become_the_measure(self):
        for text in self.PREDICATES:
            with self.subTest(text=text):
                self.assertIsNone(
                    _detect_metric(text, _SEMANTICS)[0],
                    "a threshold subject was read as the requested measure")

    def test_the_axis_is_still_not_the_measure(self):
        """The case the role check was wired into `_detect_metric` for: a
        grouping axis must not be read as the measure."""
        self.assertEqual(_detect_metric("balance by ltv bucket", _SEMANTICS)[0],
                         "current_outstanding_balance")


class TestAMeasurelessTrendStillDefaults(unittest.TestCase):
    """The other half of the distinction. A question that names NO measure still
    gets the governed default, and the substitution is still RECORDED — a
    defaulted measure must never be indistinguishable from a named one."""

    def test_a_question_naming_no_measure_resolves_none_here(self):
        for question in ("over time", "month by month", "how has it moved"):
            with self.subTest(question=question):
                self.assertIsNone(_detect_metric(question, _SEMANTICS)[0])

    def test_a_named_measure_is_not_recorded_as_defaulted(self):
        """§4. "Balance over time" supplies its measure; nothing may mark it
        substituted merely because temporal language follows it."""
        import mi_agent.execution_receipt as receipt
        from mi_agent.llm_query_parser import _deterministic_parse
        from mi_agent.tests import portfolio_truth_oracle as truth

        book = truth.canonical_book()
        book["origination_date"] = "2026-01-31"
        columns = receipt.book_columns(book)
        values = receipt.book_values(book, _SEMANTICS)
        for question in ("Show balance over time", "Compare balance over time"):
            with self.subTest(question=question):
                spec, _meta = _deterministic_parse(
                    question, _SEMANTICS, available_columns=columns,
                    available_values=values)
                self.assertEqual(spec.metric, "current_outstanding_balance")
                self.assertFalse(
                    getattr(spec, "metric_defaulted", False),
                    "a measure the reader named was recorded as substituted")


if __name__ == "__main__":
    unittest.main()
