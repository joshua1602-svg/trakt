#!/usr/bin/env python3
"""What "by" means, and what asks for a relationship.

THE INVARIANT THIS REPLACES, and it was a deliberate one:

    OLD   two NUMERIC concepts after "by" → loan-level bubble
    NEW   "by" owns GROUPING semantics;
          explicit relationship/plot language owns scatter/bubble semantics

The old rule made "by" mean different things depending on the DATATYPE of the
words after it:

    "balance by LTV by region"   → grouped matrix   (one categorical axis)
    "balance by LTV by age"      → bubble           (both numeric)

Same grammar, same request shape, two different analyses — and the second lost
its measure and its aggregation on the way to `loan_level`. A reader asking for
a breakdown has no way to know that the answer depends on how the estate happens
to store the two concepts they named.

That is a PRESENTATION heuristic standing in for a business-semantic rule.
Retired here, deliberately, and replaced by one that reads the grammar:

    by X            grouping
    by X by Y       two grouping dimensions
    by X and Y      two grouping dimensions
    vs / against / scatter / bubble / sized by / plot
                    a loan-level relationship

THE CAPABILITY IS NOT DELETED, it is addressed by name. Every bubble and scatter
test that used "balance by ltv by age" as its vehicle now says what it means, so
those paths are still proven rather than quietly abandoned. See the migration
note on each.

NOT EVERY NUMBER GETS BINNED. A natural numeric concept resolves to a grouped
representation only where governed semantics already define one — LTV →
ltv_bucket, age → age_bucket, rate → interest_rate_bucket. Where the registry
defines no grouping for a concept, none is invented.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mi_agent.llm_query_parser import _deterministic_parse           # noqa: E402
from mi_agent.mi_query_validator import load_mi_semantics            # noqa: E402

_SEMANTICS = load_mi_semantics(
    str(_REPO_ROOT / "mi_agent" / "mi_semantics_field_registry.yaml"))

BALANCE = "current_outstanding_balance"
COLUMNS = ["loan_identifier", BALANCE, "current_loan_to_value",
           "youngest_borrower_age", "current_interest_rate", "ltv_bucket",
           "age_bucket", "interest_rate_bucket", "collateral_geography",
           "erm_product_type", "broker_channel", "borrower_type"]


def parse(question: str):
    spec, _meta = _deterministic_parse(question, _SEMANTICS,
                                       available_columns=COLUMNS)
    return spec


def axes(spec):
    return list(spec.dimensions or []) or ([spec.dimension] if spec.dimension else [])


class TestByOwnsGrouping(unittest.TestCase):
    """One axis, two axes, and the three ways a reader writes two."""

    def test_one_dimension(self):
        for question, expected in (
                ("balance by ltv", "ltv_bucket"),
                ("balance by age", "age_bucket"),
                ("balance by rate", "interest_rate_bucket"),
                ("balance by region", "collateral_geography"),
                ("balance by product", "erm_product_type")):
            with self.subTest(question=question):
                spec = parse(question)
                self.assertEqual(axes(spec), [expected])
                self.assertEqual(spec.metric, BALANCE)
                self.assertEqual(spec.aggregation, "sum")

    #: The same two axes, written three ways. All must agree.
    TWO_AXES = (
        "balance by ltv by age",
        "balance by ltv and age",
        # A comma-joined list ("by ltv, age") is deliberately NOT claimed. The
        # comma is already a qualifier boundary — it is how "in Scotland,
        # balance by product" is segmented — and giving it a second meaning
        # inside a grouping phrase would put two readings on one mark. "by X by
        # Y" and "by X and Y" are the governed forms.
    )

    def test_two_dimensions_however_they_are_joined(self):
        for question in self.TWO_AXES:
            with self.subTest(question=question):
                spec = parse(question)
                self.assertEqual(set(axes(spec)), {"ltv_bucket", "age_bucket"})
                self.assertEqual(spec.metric, BALANCE)
                self.assertEqual(spec.aggregation, "sum")


class TestDatatypeDoesNotChangeWhatByMeans(unittest.TestCase):
    """THE POINT OF THE MIGRATION, stated directly.

    Whether the two concepts after "by" happen to be stored as numbers or as
    categories is an implementation fact about the book. It may not decide what
    the reader's sentence means.
    """

    PAIRS = (
        ("balance by ltv by age", {"ltv_bucket", "age_bucket"}),                # num × num
        ("balance by ltv by region", {"ltv_bucket", "collateral_geography"}),   # num × cat
        ("balance by region by product",
         {"collateral_geography", "erm_product_type"}),                         # cat × cat
        ("balance by rate by age", {"interest_rate_bucket", "age_bucket"}),     # num × num
    )

    def test_every_pairing_is_a_grouped_breakdown(self):
        for question, expected in self.PAIRS:
            with self.subTest(question=question):
                spec = parse(question)
                self.assertEqual(set(axes(spec)), expected)
                self.assertEqual(spec.aggregation, "sum")
                self.assertEqual(spec.metric, BALANCE)
                self.assertNotEqual(spec.chart_type, "bubble")

    def test_the_aggregation_survives_the_second_dimension(self):
        """Adding a breakdown changes the breakdown, not the calculation."""
        one = parse("balance by ltv")
        two = parse("balance by ltv by age")
        self.assertEqual((one.metric, one.aggregation),
                         (two.metric, two.aggregation))


class TestRelationshipLanguageKeepsTheScatter(unittest.TestCase):
    """The capability the old invariant was protecting, addressed by name."""

    BUBBLE = ("bubble chart of ltv vs age sized by balance",
              "scatter ltv against age sized by balance")

    SCATTER = ("plot ltv vs age",)

    def test_bubble_wording_still_produces_a_bubble(self):
        for question in self.BUBBLE:
            with self.subTest(question=question):
                spec = parse(question)
                self.assertEqual(spec.chart_type, "bubble")
                self.assertTrue(spec.x and spec.y and spec.size)
                self.assertEqual(len({spec.x, spec.y, spec.size}), 3)

    def test_scatter_wording_still_produces_a_scatter(self):
        for question in self.SCATTER:
            with self.subTest(question=question):
                spec = parse(question)
                self.assertEqual(spec.chart_type, "scatter")
                self.assertTrue(spec.x and spec.y)

    def test_relationship_language_is_not_a_grouping(self):
        for question in self.BUBBLE + self.SCATTER:
            with self.subTest(question=question):
                self.assertEqual(axes(parse(question)), [])


class TestOnlyGovernedGroupingsAreUsed(unittest.TestCase):
    """No bins are invented for a concept the registry does not group."""

    def test_a_governed_bucket_is_used_where_one_exists(self):
        for question, bucket in (("balance by ltv", "ltv_bucket"),
                                 ("balance by age", "age_bucket"),
                                 ("balance by rate", "interest_rate_bucket")):
            with self.subTest(question=question):
                self.assertEqual(axes(parse(question)), [bucket])

    def test_a_grouping_axis_is_always_a_registry_field(self):
        """Whatever a `by` phrase resolves to must be a governed field — the
        guard against inventing a binning for an ungoverned concept."""
        from mi_agent.llm_query_parser import _fields

        known = set(_fields(_SEMANTICS))
        for question in ("balance by ltv by age", "balance by rate by region",
                         "balance by product and broker",
                         "balance by ltv by age by region"):
            with self.subTest(question=question):
                for axis in axes(parse(question)):
                    self.assertIn(axis, known)


class TestFiltersAndDimensionsStayOrthogonal(unittest.TestCase):
    """Adding a population may not cost an axis, and vice versa."""

    def test_a_filter_does_not_remove_a_dimension(self):
        plain = parse("balance by ltv by age")
        filtered = parse("for joint borrowers, balance by ltv by age")
        self.assertEqual(set(axes(plain)), set(axes(filtered)))
        self.assertEqual(filtered.filters.get("borrower_type"), "Joint")

    def test_a_dimension_does_not_remove_a_filter(self):
        one = parse("for joint borrowers, balance by ltv")
        two = parse("for joint borrowers, balance by ltv by age")
        self.assertEqual(one.filters, two.filters)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
