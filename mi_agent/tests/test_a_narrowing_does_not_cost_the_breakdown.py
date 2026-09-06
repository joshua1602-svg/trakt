#!/usr/bin/env python3
"""Naming a population must not delete the breakdown the reader also named.

THE DEFECT, found by trying to prove the agent was NOT ready.

    "Total balance by region"                        breakdown by region
    "Total balance by region for joint borrowers"    ONE NUMBER, no breakdown

Adding a narrowing to a grouped question destroyed the grouping. The governance
receipt caught it — the answer was refused, not silently widened, and that is
the fail-closed design working — but a refusal is what the reader got for one of
the most ordinary questions an MI function asks. The capability was absent, not
merely unproven.

WHY IT HAPPENED, and it is this estate's recurring shape: one concept, two
owners. A filtered-summary branch claimed the question on the STRENGTH OF A
PHRASE —

    is_balance_q = "how much" | "total balance"
    is_count_q   = the governed count-request phrase

— and then returned a summary spec, unconditionally, without ever asking whether
the same sentence also named a grouping axis. So the reading depended on which
words the reader happened to choose for the measure:

    "total balance by region for joint borrowers"    refused
    "total exposure by region for joint borrowers"   answered
    "sum of balance by region for joint borrowers"   answered
    "average balance by region for joint borrowers"  answered

Four spellings of one analytical request, one of them unanswerable. That is a
phrase-sensitive interpretation standing where a general rule belongs.

THE GENERAL RULE, and it is about grammar, not vocabulary:

    A filtered SUMMARY is the reading only where the reader named NO breakdown.
    A sentence that names a grouping axis is a filtered BREAKDOWN, and the
    grouped path owns it.

The narrowing is unaffected either way — it is the same population owner in both
readings — so this changes which SHAPE answers, never which rows.

WHAT THIS FILE ASSERTS. Not a list of repaired sentences: an equivalence. Every
spelling of "the summed balance" must produce the same analytical shape as every
other, under every kind of narrowing, at one and at two dimensions. A future
branch that claims a question by its phrasing fails here whatever phrase it
picks.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mi_agent.llm_query_parser import (                              # noqa: E402
    _categorical_narrowings, _deterministic_parse)
from mi_agent.mi_query_validator import load_mi_semantics            # noqa: E402

_SEMANTICS = load_mi_semantics(
    str(_REPO_ROOT / "mi_agent" / "mi_semantics_field_registry.yaml"))

BALANCE = "current_outstanding_balance"

COLUMNS = ["loan_identifier", BALANCE, "current_loan_to_value",
           "youngest_borrower_age", "current_interest_rate", "ltv_bucket",
           "age_bucket", "interest_rate_bucket", "collateral_geography",
           "erm_product_type", "broker_channel", "borrower_type"]

#: The book's own catalogue, as the live path supplies it.
VALUES = {"collateral_geography": ["Scotland", "Wales", "North West", "London"],
          "erm_product_type": ["Lump Sum", "Drawdown"],
          "broker_channel": ["Alpha", "Beta", "Gamma"],
          "borrower_type": ["Joint", "Single"]}


def parse(question: str):
    spec, _meta = _deterministic_parse(question, _SEMANTICS,
                                       available_columns=COLUMNS,
                                       available_values=VALUES)
    return spec


def axes(spec):
    return list(spec.dimensions or []) or ([spec.dimension] if spec.dimension else [])


def shape(spec):
    """What the parse MEANS, reduced to the three things a reader would notice."""
    return (tuple(axes(spec)), spec.aggregation,
            tuple(sorted((k, repr(v)) for k, v in (spec.filters or {}).items())))


#: Every stated narrowing this bank exercises — one of each kind the population
#: owner resolves, so a repair that fixes one kind and not the others cannot
#: pass. Each is a bare value, because a value written NEXT TO its own field's
#: name ("for the Alpha broker") is a separate question — whether the words are
#: an axis or a population — and it has its own bank:
#: `test_a_value_beside_its_field_name_is_a_population`.
NARROWINGS = (
    ("for joint borrowers", ("borrower_type", "'Joint'")),
    ("in Scotland", ("collateral_geography", "'Scotland'")),
    ("for lump sum loans", ("erm_product_type", "'Lump Sum'")),
    ("for Alpha", ("broker_channel", "'Alpha'")),
)


class TestANarrowingDoesNotCostTheBreakdown(unittest.TestCase):
    """The headline defect, stated once per narrowing kind and per axis count."""

    def test_one_dimension_survives_every_kind_of_narrowing(self):
        for phrase, (field, value) in NARROWINGS:
            question = f"Total balance by region {phrase}"
            with self.subTest(question=question):
                spec = parse(question)
                self.assertEqual(axes(spec), ["collateral_geography"],
                                 "the breakdown the reader named was dropped")
                self.assertEqual(spec.metric, BALANCE)
                self.assertEqual(spec.aggregation, "sum")
                self.assertEqual(repr((spec.filters or {}).get(field)), value,
                                 "the narrowing the reader named was dropped")

    def test_two_dimensions_survive_a_narrowing(self):
        spec = parse("For joint borrowers, total balance by ltv by age")
        self.assertEqual(set(axes(spec)), {"ltv_bucket", "age_bucket"})
        self.assertEqual(spec.metric, BALANCE)
        self.assertEqual(spec.aggregation, "sum")
        self.assertEqual((spec.filters or {}).get("borrower_type"), "Joint")

    def test_a_count_keeps_its_breakdown_and_its_narrowing(self):
        for question in ("How many loans in Scotland by product?",
                         "Loan count by product in Scotland"):
            with self.subTest(question=question):
                spec = parse(question)
                self.assertEqual(axes(spec), ["erm_product_type"])
                self.assertEqual(spec.aggregation, "count")
                self.assertEqual((spec.filters or {}).get("collateral_geography"),
                                 "Scotland")


class TestOneRequestHasOneShapeHoweverItIsSpelled(unittest.TestCase):
    """The general property. Synonymous measure wordings are synonymous.

    This is the assertion that makes the repair general rather than a patch: it
    never names the branch that was wrong, or the phrase that triggered it. It
    says that four ways of asking for the summed balance must agree, and any
    future reading that keys off one of them breaks it.
    """

    #: Every one of these denotes "the summed balance". None of them is the
    #: canonical spelling; they are peers.
    SUMMED_BALANCE = ("total balance", "sum of balance", "total exposure",
                      "how much balance")

    def _shapes(self, template: str):
        return {measure: shape(parse(template.format(measure=measure)))
                for measure in self.SUMMED_BALANCE}

    def test_the_spelling_of_the_measure_does_not_change_the_shape(self):
        for template in ("{measure} by region for joint borrowers",
                         "{measure} by product in Scotland",
                         "For joint borrowers, {measure} by ltv by age",
                         "{measure} by broker for loans with LTV over 50%"):
            with self.subTest(template=template):
                shapes = self._shapes(template)
                distinct = set(shapes.values())
                self.assertEqual(
                    len(distinct), 1,
                    "one request, several readings, decided by wording: "
                    + "; ".join(f"{m!r} -> {s}" for m, s in sorted(shapes.items())))

    def test_the_narrowing_is_the_only_difference_from_the_unnarrowed_question(self):
        """Adding a population changes the FILTERS and nothing else.

        Stated as a difference rather than as an expected value, so it holds
        whatever the governed axis for "region" or "product" happens to be.
        """
        for measure in self.SUMMED_BALANCE:
            for phrase, (field, value) in NARROWINGS:
                plain = parse(f"{measure} by region")
                narrowed = parse(f"{measure} by region {phrase}")
                with self.subTest(measure=measure, narrowing=phrase):
                    self.assertEqual(axes(narrowed), axes(plain),
                                     "the narrowing changed the breakdown")
                    self.assertEqual(narrowed.aggregation, plain.aggregation,
                                     "the narrowing changed the aggregation")
                    self.assertEqual(narrowed.metric, plain.metric,
                                     "the narrowing changed the measure")
                    self.assertEqual(repr((narrowed.filters or {}).get(field)),
                                     value)


class TestWhereTheNarrowingSitsDoesNotDecideWhetherItCounts(unittest.TestCase):
    """The second defect the same headline sentence exposed.

    A qualifier is ended by a comma and by another qualifier's opener, and the
    resolver reads each short segment on its own. It was NOT ended by the word
    that begins a breakdown, so a scope stated before the breakdown ran on into
    it and resolved to nothing at all:

        "In Scotland, how many loans by product?"    Scotland   applied
        "Loan count by product in Scotland"          Scotland   applied
        "How many loans in Scotland by product?"     whole book  LOST

    Three spellings of one question; the one a person is most likely to type is
    the one that answered over the whole book. A grouping marker ends a
    qualifier for the same reason a comma does, and the same owner
    (`axis_marker_alternation`) names those markers for the grouping paths.
    """

    #: One scope, one breakdown, said three ways.
    SPELLINGS = ("How many loans in Scotland by product?",
                 "In Scotland, how many loans by product?",
                 "Loan count by product in Scotland")

    def test_every_spelling_resolves_the_same_scope_and_the_same_breakdown(self):
        shapes = {q: shape(parse(q)) for q in self.SPELLINGS}
        self.assertEqual(
            len(set(shapes.values())), 1,
            "where the scope sits changed what it means: "
            + "; ".join(f"{q!r} -> {s}" for q, s in shapes.items()))
        for spec in (parse(q) for q in self.SPELLINGS):
            self.assertEqual(axes(spec), ["erm_product_type"])
            self.assertEqual((spec.filters or {}).get("collateral_geography"),
                             "Scotland")

    def test_the_same_holds_for_a_measure(self):
        for question in ("Total balance in Scotland by product",
                         "Total balance by product in Scotland"):
            with self.subTest(question=question):
                spec = parse(question)
                self.assertEqual(axes(spec), ["erm_product_type"])
                self.assertEqual((spec.filters or {}).get("collateral_geography"),
                                 "Scotland")


class TestAxisTextIsNotOfferedToTheValueResolver(unittest.TestCase):
    """The guard on the boundary above, stated at the owner it constrains.

    Making a grouping marker a qualifier boundary creates a segment that BEGINS
    at that marker, and that segment is axis text. Handing it to the value
    resolver would read a breakdown as a narrowing — "show balance by lump sum"
    names the product axis with one of its values' own words — so the narrowing
    owner skips it.

    Asserted on `_categorical_narrowings` rather than on the finished spec
    deliberately. Whether the FULL parse of "balance by lump sum" ought to group
    or narrow is a live question this estate answers elsewhere (the value stands
    outside a grouping position, so the dimension reader drops it and
    `_parse_filters` claims it — the behaviour predates this boundary and is
    unchanged by it). What this file is entitled to assert is narrower and
    exact: the boundary introduced here manufactures no narrowing of its own.
    """

    def test_a_segment_that_begins_at_a_grouping_marker_yields_no_narrowing(self):
        for question in ("show balance by lump sum", "balance by scotland"):
            with self.subTest(question=question):
                self.assertEqual(
                    _categorical_narrowings(question, _SEMANTICS, COLUMNS, VALUES),
                    {}, "axis text was read as a population")

    def test_a_qualifier_inside_an_axis_segment_still_resolves(self):
        """The boundary must not be a wall. "by region for loans in Wales" is a
        regional breakdown NARROWED to Wales, and the qualifier opens after the
        marker."""
        self.assertEqual(
            _categorical_narrowings("balance by region for loans in wales",
                                    _SEMANTICS, COLUMNS, VALUES),
            {"collateral_geography": "Wales"})


class TestTheSummaryReadingIsStillTheReadingWithoutABreakdown(unittest.TestCase):
    """The other half of the rule, so the repair cannot be one-sided.

    A narrowed question that names NO axis is still a filtered summary — one
    number over the stated population. Nothing here asks for that to change, and
    a repair that turned every filtered question into a breakdown would be the
    same defect pointing the other way.
    """

    def test_a_narrowed_question_with_no_axis_is_one_number(self):
        for question in ("What is the total balance for joint borrowers?",
                         "How much balance is there in Scotland?",
                         "How many loans have LTV over 50%?"):
            with self.subTest(question=question):
                spec = parse(question)
                self.assertEqual(axes(spec), [],
                                 "a breakdown was invented for a question that "
                                 "named no axis")
                self.assertEqual(spec.intent, "summary")

    def test_an_unnarrowed_summary_is_untouched(self):
        spec = parse("What is the total balance?")
        self.assertEqual(axes(spec), [])
        self.assertEqual(spec.filters or {}, {})


if __name__ == "__main__":
    unittest.main()
