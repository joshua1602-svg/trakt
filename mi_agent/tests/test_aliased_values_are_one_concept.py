#!/usr/bin/env python3
"""A value several ALIASED fields carry is one claim, not an ambiguous one.

WHAT WAS MEASURED. On the live tape, `collateral_geography` and
`geographic_region_obligor` hold the same region spellings. `value_field`
refuses any value more than one governed field claims, so every region FILTER
resolved to nothing and the reader was told ``unknown category: 'london'``
about a region the book plainly carries. Grouping was unaffected — the axis
owner picks a field by preference and never binds a value — which is why
"Show balance by region." answered while "balance in London" did not.

THE RULE THAT REFUSED IT IS RIGHT AND STAYS. Two fields claiming one value is
normally a genuine ambiguity: binding "lump sum" to geography because geography
spoke first is how a product type became a place. What was missing is that some
fields are not competing claims at all. Every region field declares
``value_domain: uk_region``; fields that draw their values from ONE declared
domain are ALIASES of one concept, and the domain's preference order — the
SAME order the grouping owner walks — says which of them a term binds to.

So the seam is the declared domain, and the test is two-sided:

  * hits that share one ``value_domain`` resolve, to the preferred field;
  * hits that span different domains, or any field that declares none, stay
    ambiguous and are disclosed exactly as before.

Falsified against the unfixed code: every test below fails on the parent of
the commit that adds them, and `test_a_value_two_domains_claim_is_still_ambiguous`
passes there — which is the point of keeping it.
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

from mi_agent import categorical_spans as CS
from mi_agent.mi_agent_workflow import run_mi_agent_query
from mi_agent.mi_query_validator import load_mi_semantics

_SEMANTICS_PATH = _REPO_ROOT / "mi_agent" / "mi_semantics_field_registry.yaml"
_BALANCE = "current_outstanding_balance"
_GEO = "collateral_geography"
_OBLIGOR = "geographic_region_obligor"
_REPORTING = "canonical_region_reporting"


def _semantics():
    return load_mi_semantics(_SEMANTICS_PATH)


class TestOneDomainIsOneConcept(unittest.TestCase):
    """`value_field`, on the book's own catalogue."""

    def setUp(self):
        self.semantics = _semantics()

    def test_a_single_claimant_is_unchanged(self):
        self.assertEqual(
            CS.value_field("london", {_GEO: ["London"]}, self.semantics),
            (_GEO, "London"))

    def test_two_aliases_of_one_domain_resolve_to_the_preferred_field(self):
        """The reproduction from the handover, verbatim."""
        self.assertEqual(
            CS.value_field("london", {_GEO: ["London"], _OBLIGOR: ["London"]},
                           self.semantics),
            (_GEO, "London"))

    def test_the_preference_is_the_grouping_owner_s_own_order(self):
        """Where the harmonised column is present it wins — for the FILTER as
        well as the axis, so a question that groups and filters on region binds
        one field rather than two."""
        from mi_agent.llm_query_parser import _REGION_PREFERENCE

        self.assertEqual(_REGION_PREFERENCE[0], _REPORTING)
        self.assertEqual(
            CS.value_field("london",
                           {_GEO: ["London"], _OBLIGOR: ["London"],
                            _REPORTING: ["London"]},
                           self.semantics),
            (_REPORTING, "London"))

    def test_a_value_two_domains_claim_is_still_ambiguous(self):
        """The protection the rule was written for. A product type carried by a
        geography field is not a region, and must not be resolved by preference.

        The two-argument form of this assertion passes on the unfixed code
        too, deliberately: it is the behaviour that must NOT move."""
        self.assertIsNone(
            CS.value_field("lump sum",
                           {"product_type": ["Lump Sum"], _GEO: ["Lump Sum"]},
                           self.semantics))

    def test_a_field_that_declares_no_domain_stays_ambiguous(self):
        """Silence is not agreement. Two fields that declare nothing about
        where their values come from have not been shown to be aliases."""
        self.assertIsNone(
            CS.value_field("drawdown",
                           {"product_type": ["Drawdown"],
                            "erm_sub_product_type": ["Drawdown"]},
                           self.semantics))

    def test_without_semantics_the_owner_keeps_its_strict_rule(self):
        """The signature grew a parameter; the old call is the old behaviour.
        A caller that cannot say which domain a field draws from has not
        established that its claimants are aliases."""
        self.assertIsNone(
            CS.value_field("london", {_GEO: ["London"], _OBLIGOR: ["London"]}))


class TestTheRegionFilterExecutes(unittest.TestCase):
    """The defect as a reader meets it: a filter, on a book carrying both."""

    @classmethod
    def setUpClass(cls):
        cls.semantics = _semantics()
        rng = np.random.default_rng(20260903)
        n = 400
        geo = rng.choice(["London", "South East", "Wales", "Scotland",
                          "North West"], n)
        cls.base = {
            _BALANCE: rng.uniform(40_000, 500_000, n).round(2),
            "current_loan_to_value": rng.uniform(15.0, 88.0, n).round(3),
            "youngest_borrower_age": rng.integers(58, 96, n),
            "account_status": rng.choice(["Active", "Redeemed"], n, p=[.94, .06]),
            "data_cut_off_date": ["2026-06-30"] * n,
            _GEO: geo,
        }
        cls.one_column = pd.DataFrame(cls.base)
        cls.two_columns = pd.DataFrame({**cls.base, _OBLIGOR: geo})

    def _london_total(self, frame):
        return float(frame.loc[frame[_GEO] == "London", _BALANCE].sum())

    def _ask(self, frame):
        return run_mi_agent_query("What is the total balance in London?",
                                  frame, self.semantics)

    def _figure(self, result):
        """The BALANCE the answer carried, by name. The row also holds a loan
        count, and reading position 0 would reconcile the wrong number."""
        qr = result.get("query_result")
        rows = (qr.to_dict().get("data") or []) if qr is not None else []
        self.assertTrue(rows, "the answer carried no row")
        row = dict(rows[0])
        key = next((k for k in row if str(k).startswith(_BALANCE)), None)
        self.assertIsNotNone(key, "no balance in %s" % sorted(row))
        return float(row[key])

    def test_one_region_column_answers_as_it_always_did(self):
        result = self._ask(self.one_column)
        self.assertTrue(result.get("ok"), result.get("error"))
        self.assertAlmostEqual(self._figure(result),
                               self._london_total(self.one_column), places=2)

    def test_two_aliased_region_columns_answer_the_same_question(self):
        """Byte-identical content in a second registered region column must not
        turn an answerable question into ``unknown category: 'london'``."""
        result = self._ask(self.two_columns)
        self.assertTrue(result.get("ok"), result.get("error"))
        self.assertAlmostEqual(self._figure(result),
                               self._london_total(self.two_columns), places=2)

    def test_the_filter_is_recorded_against_one_named_field(self):
        spec = self._ask(self.two_columns).get("spec") or {}
        filters = (spec.get("filters") if isinstance(spec, dict)
                   else getattr(spec, "filters", None)) or {}
        unavailable = (spec.get("unavailable_filters") if isinstance(spec, dict)
                       else getattr(spec, "unavailable_filters", None)) or []
        self.assertEqual(filters, {_GEO: "London"})
        self.assertEqual([n for n in unavailable if "unknown category" in str(n)],
                         [])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
