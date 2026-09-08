#!/usr/bin/env python3
"""A restriction naming SEVERAL values, on the axis it groups by.

    "Show balance by region for loans in Wales and Scotland."

A two-region breakdown. There is no other way to phrase it, and it refused —
having computed all five regions and then declined to publish them, because the
coverage gate saw that neither Wales nor Scotland had been applied. That is
fail-closed and it is not an answer.

THREE THINGS WERE WRONG, and they are separable:

1. `_grouping_segments` split every "and" as an axis separator, so
   "by region for loans in Wales and Scotland" read as the axes
   ["region for loans in wales", "scotland"]. An "and" inside a segment's own
   QUALIFIER coordinates VALUES, not axes.

2. Only the first value bound. The clause splitter cuts "... in wales and
   scotland" into "... in wales" and a bare " scotland", and a clause that is
   nothing but a governed value resolved to nothing. Measured on the
   already-working axis: "Show balance by broker for loans in Wales and
   Scotland." bound `collateral_geography = Wales` and lost Scotland entirely —
   caught by the gate, so nothing wrong was published, but it is the same
   defect one axis over.

3. `_grouped_value_filters` dropped any filter whose field was the grouping
   dimension. That rule protects a real case — "show balance by lump sum" names
   an axis with a value's own words, and filtering to that value would answer a
   one-row question the reader did not ask — but it fired on a restriction that
   arrived through a qualifier, which is a different thing entirely.

The discriminator is WHERE THE VALUE'S WORDS ARE: inside the axis phrase it is
the axis; after the qualifier that introduced it, it is a restriction.
`test_the_axis_phrase_s_own_value_is_still_dropped` is the protection, and it
passes on the unfixed code too.

The single-value case answers now rather than refusing — one row, which is what
"by region for loans in Wales" describes. It was recorded as defensible to
refuse; it is not defensible to refuse it while answering the two-value one, and
a reader who wanted the figure alone can already ask for it directly.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mi_agent import execution_receipt as R
from mi_agent import llm_query_parser as P
from mi_agent.mi_agent_workflow import run_mi_agent_query
from mi_agent.mi_query_validator import load_mi_semantics

_SEMANTICS_PATH = _REPO_ROOT / "mi_agent" / "mi_semantics_field_registry.yaml"
_BALANCE = "current_outstanding_balance"
_GEO = "collateral_geography"
_BROKER = "broker_channel"
_PRODUCT = "erm_product_type"


def _frame():
    rng = np.random.default_rng(20260904)
    n = 400
    return pd.DataFrame({
        _BALANCE: rng.uniform(60_000, 480_000, n).round(2),
        _BROKER: rng.choice(["Alpha", "Beta"], n),
        _PRODUCT: rng.choice(["Lump Sum", "Drawdown"], n),
        _GEO: rng.choice(["North", "South East", "Midlands", "Wales",
                          "Scotland"], n),
        "data_cut_off_date": ["2026-06-30"] * n,
    })


class _Fixture(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.semantics = load_mi_semantics(_SEMANTICS_PATH)
        cls.frame = _frame()
        cls.values = R.book_values(cls.frame, cls.semantics)
        cls.columns = set(cls.frame.columns)

    def _filters(self, question, exclude):
        return P._grouped_value_filters(
            question.lower(), self.semantics, self.columns,
            exclude_dims=exclude, available_values=self.values)[0]

    def _ask(self, question):
        return run_mi_agent_query(question, self.frame, self.semantics)

    def _rows(self, result):
        self.assertTrue(result.get("ok"), result.get("error"))
        qr = result.get("query_result")
        return [dict(r) for r in ((qr.to_dict().get("data") or [])
                                  if qr is not None else [])]

    def _sum(self, mask, by):
        return {str(k): round(float(v), 2) for k, v in
                self.frame.loc[mask].groupby(by)[_BALANCE].sum().items()}


class TestTheReading(_Fixture):
    """The three parts, each on its own owner."""

    def test_an_and_inside_a_qualifier_is_not_an_axis_separator(self):
        _metric, segments = P._grouping_segments(
            "show balance by region for loans in wales and scotland.")
        self.assertEqual(len(segments), 1, segments)
        self.assertIn("wales", segments[0])
        self.assertIn("scotland", segments[0])

    def test_a_genuine_second_axis_still_separates(self):
        _metric, segments = P._grouping_segments("balance by region and broker")
        self.assertEqual(segments, ["region", "broker"])

    def test_several_values_bind_as_one_membership_predicate(self):
        parsed = P._parse_filters(
            "show balance by broker for loans in wales and scotland.",
            self.semantics, self.columns, available_values=self.values)
        self.assertIn(_GEO, parsed)
        condition = parsed[_GEO]
        self.assertEqual(condition.get("op"), "in")
        self.assertEqual(sorted(condition.get("value")), ["Scotland", "Wales"])

    def test_a_restriction_on_the_grouped_axis_survives(self):
        condition = self._filters(
            "show balance by region for loans in wales and scotland.",
            [_GEO]).get(_GEO)
        self.assertIsNotNone(condition)
        self.assertEqual(sorted(condition.get("value")), ["Scotland", "Wales"])

    def test_the_axis_phrase_s_own_value_is_still_dropped(self):
        """THE PROTECTION, and it passes on the unfixed code too. "by lump sum"
        names the product axis with a value's own words; filtering to that value
        would answer a one-row question nobody asked."""
        self.assertEqual(self._filters("show balance by lump sum", [_PRODUCT]), {})


class TestTheAnswer(_Fixture):
    """Reconciled to the frame, never to a previous run."""

    def test_two_regions_on_a_different_axis(self):
        rows = self._rows(self._ask(
            "Show balance by broker for loans in Wales and Scotland."))
        got = {str(r[_BROKER]): round(float(r[_BALANCE + "_sum"]), 2)
               for r in rows}
        self.assertEqual(got, self._sum(
            self.frame[_GEO].isin(["Wales", "Scotland"]), _BROKER))

    def test_two_regions_on_the_region_axis(self):
        rows = self._rows(self._ask(
            "Show balance by region for loans in Wales and Scotland."))
        got = {str(r[_GEO]): round(float(r[_BALANCE + "_sum"]), 2) for r in rows}
        self.assertEqual(got, self._sum(
            self.frame[_GEO].isin(["Wales", "Scotland"]), _GEO))

    def test_one_region_on_the_region_axis_is_one_row(self):
        rows = self._rows(self._ask(
            "Show balance by region for loans in Wales."))
        got = {str(r[_GEO]): round(float(r[_BALANCE + "_sum"]), 2) for r in rows}
        self.assertEqual(got, self._sum(self.frame[_GEO] == "Wales", _GEO))


class TestWhatMustNotMove(_Fixture):

    def test_an_unrestricted_breakdown_is_the_whole_book(self):
        rows = self._rows(self._ask("Show balance by region."))
        self.assertEqual(len(rows), 5)

    def test_a_breakdown_by_a_value_named_axis_covers_every_value(self):
        rows = self._rows(self._ask("Show balance by product type"))
        self.assertEqual({str(r[_PRODUCT]) for r in rows},
                         {"Lump Sum", "Drawdown"})

    def test_a_single_value_filter_with_no_grouping_is_unchanged(self):
        rows = self._rows(self._ask("What is the total balance in Wales?"))
        self.assertEqual(round(float(rows[0][_BALANCE + "_sum"]), 2),
                         round(float(self.frame.loc[self.frame[_GEO] == "Wales",
                                                    _BALANCE].sum()), 2))

    def test_a_place_the_book_does_not_carry_still_refuses(self):
        result = self._ask("What is the average LTV in Atlantis?")
        self.assertFalse(result.get("ok"))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
