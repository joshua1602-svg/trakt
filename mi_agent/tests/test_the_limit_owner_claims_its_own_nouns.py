#!/usr/bin/env python3
"""The risk-limit owner's nouns are claimed by it, not left for geography.

MEASURED, on the shipped path: "What is the largest geographic concentration
versus limit?" is refused with

    No loans in this book match that filter ('concentration versus limit')

The question names no category at all. `_CATEGORICAL_FILTER_RE` reads
"geographic X" as "the place X", so the analytic phrase after "geographic" is
offered as a candidate value; `_claimed_by_an_owner` is the guard that stops a
candidate no owner claims being RECORDED as a category the book does not carry,
and it stopped two words out of three. "concentration" is analytical framing and
"versus" is a grouping marker; "limit" — the noun an exposure is measured
AGAINST, and the subject of the route that claims this very question — was
claimed by nothing.

WHY NOT THE FRAMING LIST. `_ANALYTICAL_FRAMING_WORDS` already carries
"concentration", "coverage" and "exposure", and "limit" reads like their kin.
But that set has a second job: a word in it is not metric RESIDUE, so adding
"limit" would take "Show the limit by region" from *"'limit' is not a governed
measure in this dataset; no substitute was used"* to a balance breakdown — the
silent substitution the estate exists to prevent. That refusal is pinned below.

SO THE OWNER IS ASKED INSTEAD. `_RISK_LIMIT_NOUNS` is the risk-limit
vocabulary's own nouns, read only by `_claimed_by_an_owner`, and every one of
them is asserted to appear in `_RISK_LIMIT_RE` so the two cannot drift into two
vocabularies.

The guard stays conservative by construction: it requires EVERY word of a
candidate to be claimed, so "Headroom for platinum loans" still records
`unknown category: 'platinum'`.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import pandas as pd

from mi_agent import execution_receipt as R
from mi_agent import llm_query_parser as P
from mi_agent.mi_agent_workflow import run_mi_agent_query
from mi_agent.mi_query_validator import load_mi_semantics

_SEMANTICS_PATH = _REPO_ROOT / "mi_agent" / "mi_semantics_field_registry.yaml"
CONCENTRATION = "what is the largest geographic concentration versus limit?"


class _Fixture(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.semantics = load_mi_semantics(_SEMANTICS_PATH)
        rng = np.random.default_rng(20260904)
        n = 200
        cls.frame = pd.DataFrame({
            "current_outstanding_balance": rng.uniform(60_000, 480_000, n).round(2),
            "collateral_geography": rng.choice(["North", "Wales", "Scotland"], n),
            "broker_channel": rng.choice(["Alpha", "Beta"], n),
            "data_cut_off_date": ["2026-06-30"] * n,
        })
        cls.values = R.book_values(cls.frame, cls.semantics)
        cls.columns = set(cls.frame.columns)

    def _notes(self, question):
        notes = []
        P._parse_categorical_filter(question.lower(), self.semantics,
                                    self.columns, self.values, unresolved=notes)
        return notes


class TestTheOwnerIsAsked(_Fixture):

    def test_every_noun_is_the_owner_s_own(self):
        """ONE vocabulary. A noun here that the recogniser does not carry would
        be a second opinion about what a limit question says."""
        pattern = P._RISK_LIMIT_RE.pattern
        for noun in P._RISK_LIMIT_NOUNS:
            with self.subTest(noun=noun):
                self.assertIn(noun, pattern)

    def test_the_limit_noun_is_claimed(self):
        self.assertTrue(P._claimed_by_an_owner(
            "limit", self.semantics, self.columns, self.values))

    def test_the_analytic_phrase_records_no_unknown_category(self):
        self.assertEqual(self._notes(CONCENTRATION), [])

    def test_the_question_is_no_longer_refused_for_a_category(self):
        result = run_mi_agent_query(CONCENTRATION, self.frame, self.semantics)
        spec = result.get("spec") or {}
        self.assertEqual(P.unknown_category_names(
            spec.get("unavailable_filters") or []), [])
        self.assertNotIn("No loans in this book match that filter",
                         str(result.get("error") or ""))


class TestWhatMustNotMove(_Fixture):

    def test_limit_is_still_not_a_measure(self):
        """The reason this is not in `_ANALYTICAL_FRAMING_WORDS`."""
        result = run_mi_agent_query("Show the limit by region", self.frame,
                                    self.semantics)
        self.assertFalse(result.get("ok"))
        self.assertIn("not a governed measure", str(result.get("error") or ""))

    def test_a_real_unknown_category_in_a_limit_question_is_still_recorded(self):
        self.assertEqual(P.unknown_category_names(
            self._notes("Headroom for platinum loans")), ["'platinum'"])

    def test_a_place_the_book_does_not_carry_is_still_recorded(self):
        self.assertEqual(P.unknown_category_names(
            self._notes("what is the average ltv in atlantis?")), ["'atlantis'"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
