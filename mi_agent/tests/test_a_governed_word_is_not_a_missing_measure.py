"""Three words the reader owns, and the estate was reading as missing measures.

Definitions given by the product owner, 2026-09-04, and the shape each needs:

    funded         the PORTFOLIO — i.e. not the pipeline. A dataset word.
    withdrawals    the pipeline stage WITHDRAWN. A governed VALUE.
    amount         defaults to current outstanding balance, count as the
                   fallback, unless the reader says which.

All three were refused by the metric-residue guard, which exists for a good
reason — "show me the unicorn ratio by region" must not answer as balance by
region — and was catching governed vocabulary with it. Measured on the deployed
build by the 115-question replay:

    Show me the funded loan book summary by region
        -> 'funded' is not a governed measure in this dataset
    What stage had the most withdrawals?
        -> 'withdrawals' is not a governed measure in this dataset
    What is the current pipeline amount?
        -> 'amount' could mean more than one governed measure (Balance or Valuation)

EACH IS FIXED AT ITS OWN OWNER, none by widening the residue guard itself:

* `funded` joins `pipeline`, `book` and `portfolio` in the analytical framing
  vocabulary. The estate had already decided this once: `pipeline_stage_
  vocabulary` DROPS the tape spelling "funded" because it "names the governed
  DATASET" in a sentence, and the framing set carried its twin `pipeline` for
  exactly the reason this fixes.
* `withdrawals` is a question-side spelling of a governed stage, so it goes in
  the ONE stage vocabulary; and the residue guard now asks that vocabulary, on
  the same principle it already applies to the book's own category values —
  a word a governed owner claims is not a measure this dataset lacks.
* `amount` is answered rather than refused, and DISCLOSED: `metric_defaulted`
  is the existing shape for "the model chose the measure", so a reader can see
  the choice was made for them.

The residue guard's own job is pinned in each class: an invented measure still
refuses, naming the term.
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

from mi_agent import llm_query_parser as P
from mi_agent.mi_agent_workflow import run_mi_agent_query
from mi_agent.mi_query_validator import load_mi_semantics
from question_interpretation.lexical import pipeline_stage_vocabulary

_SEMANTICS_PATH = _REPO_ROOT / "mi_agent" / "mi_semantics_field_registry.yaml"
_BALANCE = "current_outstanding_balance"
INVENTED = "show me the unicorn ratio by region"


class _Fixture(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.semantics = load_mi_semantics(_SEMANTICS_PATH)
        rng = np.random.default_rng(20260904)
        n = 300
        cls.frame = pd.DataFrame({
            _BALANCE: rng.uniform(60_000, 480_000, n).round(2),
            "current_valuation_amount": rng.uniform(200_000, 900_000, n).round(2),
            "pipeline_stage": rng.choice(
                ["KFI", "APPLICATION", "OFFER", "COMPLETED", "WITHDRAWN"], n),
            "collateral_geography": rng.choice(["London", "Wales"], n),
            "data_cut_off_date": ["2026-06-30"] * n,
        })

    def _ask(self, question):
        return run_mi_agent_query(question, self.frame, self.semantics)

    def _refused_as_missing_measure(self, question):
        return "is not a governed measure" in str(self._ask(question).get("error") or "")

    def test_an_invented_measure_still_refuses(self):
        """The guard's own job, asserted in every class that touches it."""
        result = self._ask(INVENTED)
        self.assertFalse(result.get("ok"))
        self.assertIn("unicorn", str(result.get("error") or ""))


class TestFundedNamesThePortfolio(_Fixture):

    def test_the_funded_loan_book_summary_answers(self):
        for question in ("Show me the funded loan book summary by region",
                         "Show me the Funded loan book summary by region"):
            with self.subTest(question=question):
                self.assertFalse(self._refused_as_missing_measure(question))
                self.assertTrue(self._ask(question).get("ok"),
                                self._ask(question).get("error"))

    def test_funded_balance_still_names_the_balance(self):
        """`funded balance` is a measure phrase and must keep resolving — this
        change may only stop `funded` being read as a measure ON ITS OWN."""
        spec = self._ask("What is the total funded balance?").get("spec") or {}
        self.assertEqual(spec.get("metric"), _BALANCE)


class TestWithdrawalsNameTheStage(_Fixture):

    def test_the_vocabulary_knows_the_noun(self):
        vocabulary = pipeline_stage_vocabulary()
        self.assertEqual(vocabulary.get("withdrawals"), "WITHDRAWN")
        self.assertEqual(vocabulary.get("withdrawn"), "WITHDRAWN")

    def test_the_singular_is_dropped_by_the_fragment_rule(self):
        """Not an oversight, and worth pinning so nobody "fixes" it.

        The vocabulary drops any spelling that is a PREFIX of a longer one for
        the same stage — the rule that stops `complete` (a fragment of
        `completed`) turning five data-completeness questions into a COMPLETED
        stage. `withdrawal` is a prefix of `withdrawals`, so it goes the same
        way. It stays in the data map, where it normalises a tape cell.
        """
        from mi_agent_api.pipeline_prep import _STAGE_CANON

        self.assertEqual(_STAGE_CANON.get("withdrawal"), "WITHDRAWN")
        self.assertIsNone(pipeline_stage_vocabulary().get("withdrawal"))

    def test_funded_is_still_not_a_stage(self):
        """The adjustment this joins: a spelling that collides with a governed
        VIEW name is dropped, so "funded" never acquires a COMPLETED stage."""
        self.assertIsNone(pipeline_stage_vocabulary().get("funded"))

    def test_the_withdrawals_question_is_not_a_missing_measure(self):
        self.assertFalse(
            self._refused_as_missing_measure("What stage had the most withdrawals?"))


class TestAmountDefaultsToBalance(_Fixture):

    def test_amount_answers_on_the_balance(self):
        result = self._ask("What is the current pipeline amount?")
        self.assertTrue(result.get("ok"), result.get("error"))
        self.assertEqual((result.get("spec") or {}).get("metric"), _BALANCE)

    def test_the_choice_is_disclosed(self):
        """A measure chosen FOR the reader is published as chosen for them."""
        spec = self._ask("What is the current pipeline amount?").get("spec") or {}
        self.assertTrue(spec.get("metric_defaulted"))

    def test_value_stays_ambiguous(self):
        """Only `amount` was given a default. `value` still asks."""
        self.assertIn("value", P._AMBIGUOUS_MEASURE_WORDS)
        self.assertNotIn("amount", P._AMBIGUOUS_MEASURE_WORDS)

    def test_a_named_measure_still_wins(self):
        spec = self._ask("What is the total valuation amount?").get("spec") or {}
        self.assertEqual(spec.get("metric"), "current_valuation_amount")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
