#!/usr/bin/env python3
""""The Alpha broker" is one broker, not every broker.

THE DEFECT. A governed value written next to its own field's name was read as a
request to break down BY that field, and the value itself was dropped:

    "total balance for Alpha"              broker_channel = Alpha      ✓
    "total balance for the Alpha broker"   breakdown by broker, no filter
    "total balance for lump sum loans"     erm_product_type = Lump Sum ✓
    "total balance for Lump Sum products"  breakdown by product, no filter

Nothing wrong reached the reader — the receipt refused, naming the narrowing it
could not see applied, which is the fail-closed design doing its job. But "the
Alpha broker" is how a person says "Alpha", and it could not be answered.

IT WORKED FOR ONE FIELD. "in the Scotland region" resolved correctly, because
the prepositional pattern carries a fixed list of trailing nouns and `region` is
on it while `broker` and `products` are not. One construction, and whether it
worked depended on which field the reader was talking about — a list standing in
for a rule.

THE RULE. A value phrase may be followed by the name of the field that value
belongs to, and that is a narrowing to the value, not a breakdown by the field.
It is checked against the registry and the book's own catalogue rather than a
noun list: the trailing words must name a governed field, and the leading words
must resolve to a value OF THAT SAME FIELD. "The London broker" therefore does
not resolve — London is a place, not a broker — and no filter is invented.

WHAT THIS FILE ASSERTS is the equivalence, not a list of repaired sentences: for
every governed categorical field the fixture carries, naming a value with its
field's name means what naming the value alone means.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mi_agent.llm_query_parser import _deterministic_parse             # noqa: E402
from mi_agent.mi_query_validator import load_mi_semantics              # noqa: E402

_SEMANTICS = load_mi_semantics(
    str(_REPO_ROOT / "mi_agent" / "mi_semantics_field_registry.yaml"))

BALANCE = "current_outstanding_balance"

COLUMNS = ["loan_identifier", BALANCE, "current_loan_to_value",
           "youngest_borrower_age", "current_interest_rate", "ltv_bucket",
           "age_bucket", "interest_rate_bucket", "collateral_geography",
           "erm_product_type", "broker_channel", "borrower_type"]

VALUES = {"collateral_geography": ["Scotland", "Wales", "North West", "London"],
          "erm_product_type": ["Lump Sum", "Drawdown"],
          "broker_channel": ["Alpha", "Beta", "Gamma"],
          "borrower_type": ["Joint", "Single"]}

#: ``(field, a value the book carries, the field's own name as a reader writes
#: it)``. The names are the registry's own synonyms for the field — nothing here
#: is new vocabulary.
FIELD_NAMES = (
    ("broker_channel", "Alpha", "broker"),
    ("broker_channel", "Beta", "broker channel"),
    ("erm_product_type", "Lump Sum", "product"),
    ("erm_product_type", "Drawdown", "product type"),
    ("collateral_geography", "Scotland", "region"),
    ("collateral_geography", "Wales", "geography"),
)


def parse(question: str):
    spec, _meta = _deterministic_parse(question, _SEMANTICS,
                                       available_columns=COLUMNS,
                                       available_values=VALUES)
    return spec


def axes(spec):
    return list(spec.dimensions or []) or ([spec.dimension] if spec.dimension else [])


def shape(spec):
    return (tuple(axes(spec)), spec.aggregation,
            tuple(sorted((k, repr(v)) for k, v in (spec.filters or {}).items())))


class TestAValueWithItsFieldNameIsThatValue(unittest.TestCase):

    def test_it_narrows_and_does_not_break_down(self):
        for field, value, name in FIELD_NAMES:
            question = f"What is the total balance for the {value} {name}?"
            with self.subTest(question=question):
                spec = parse(question)
                self.assertEqual((spec.filters or {}).get(field), value,
                                 "the value was dropped")
                self.assertEqual(axes(spec), [],
                                 "a breakdown was invented from the field name")

    def test_it_means_what_the_bare_value_means(self):
        """The equivalence, which is the whole point. A field name added after a
        value adds nothing to the request."""
        for field, value, name in FIELD_NAMES:
            with self.subTest(field=field, value=value, name=name):
                bare = parse(f"What is the total balance for {value}?")
                named = parse(f"What is the total balance for the {value} {name}?")
                self.assertEqual(shape(named), shape(bare),
                                 f"'{value} {name}' does not mean '{value}'")


class TestItCannotInventANarrowing(unittest.TestCase):
    """The guard. The trailing words name a field; the leading words must name a
    value OF THAT FIELD, or nothing is claimed."""

    def test_a_value_of_another_field_is_not_claimed(self):
        """"The London broker" — London is a place, and no broker is called it.
        The reading must not bind `broker_channel = London`, and it must not
        quietly bind `collateral_geography = London` either: the reader asked
        about a broker."""
        spec = parse("What is the total balance for the London broker?")
        self.assertNotEqual((spec.filters or {}).get("broker_channel"), "London")

    def test_a_field_name_with_no_value_is_still_a_breakdown(self):
        """Nothing here may cost an ordinary breakdown its axis."""
        for question, expected in (("Total balance by broker", "broker_channel"),
                                   ("Total balance by product", "erm_product_type"),
                                   ("Total balance by region", "collateral_geography")):
            with self.subTest(question=question):
                spec = parse(question)
                self.assertEqual(axes(spec), [expected])
                self.assertEqual(spec.filters or {}, {})

    def test_a_value_after_a_grouping_marker_is_still_the_axis(self):
        """"Balance by broker" names an axis; the rule fires on a QUALIFIER, and
        a term standing after the grouping marker is not one."""
        spec = parse("Total balance by broker for joint borrowers")
        self.assertEqual(axes(spec), ["broker_channel"])
        self.assertEqual((spec.filters or {}).get("borrower_type"), "Joint")


class TestTheNumberIsRightToo(unittest.TestCase):
    """From English to a figure, against a pandas oracle that imports nothing
    from the product.

    Reading the construction correctly is not the same as computing it
    correctly, and the population is exactly what this repair moves. Each row
    executes the question over the canonical book and compares both the total
    and the loan count with an independently calculated one — and asserts the
    figure is NOT the whole book's, so a row cannot pass on a narrowing that
    silently did nothing.
    """

    CASES = (
        ("What is the total balance for the Alpha broker?",
         ("broker_channel", "eq", "Alpha")),
        ("What is the total balance for Lump Sum products?",
         ("erm_product_type", "eq", "Lump Sum")),
        ("What is the total balance for the Drawdown product type?",
         ("erm_product_type", "eq", "Drawdown")),
        ("What is the total balance in the Scotland region?",
         ("collateral_geography", "eq", "Scotland")),
        ("How many loans are there for the Beta broker?",
         ("broker_channel", "eq", "Beta")),
    )

    @classmethod
    def setUpClass(cls):
        from mi_agent.mi_agent_workflow import run_mi_agent_query
        from mi_agent.tests import portfolio_truth_oracle as truth

        cls.run_query = staticmethod(run_mi_agent_query)
        cls.truth = truth
        cls.book = truth.canonical_book()

    def test_every_figure_matches_an_independently_computed_one(self):
        truth = self.truth
        for question, predicate in self.CASES:
            with self.subTest(question=question):
                result = self.run_query(question, self.book, _SEMANTICS)
                self.assertTrue(result.get("ok"),
                                f"not answered: {result.get('error')!r}")
                frame = result["query_result"].data
                self.assertEqual(len(frame), 1, "expected one filtered figure")
                row = frame.iloc[0]
                self.assertAlmostEqual(
                    float(row[f"{truth.BALANCE}_sum"]),
                    truth.total(self.book, truth.BALANCE, [predicate]), places=2)
                self.assertEqual(int(row["loan_count"]),
                                 truth.row_count(self.book, [predicate]))
                self.assertNotEqual(int(row["loan_count"]), len(self.book),
                                    "the narrowing did nothing")


if __name__ == "__main__":
    unittest.main()
