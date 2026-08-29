"""GOVERNED SPAN OWNERSHIP — one span, one semantic claim.

The defect this closes was measured on a real book. With a broker called
**Gamma Direct**:

    "How many Gamma Direct loans do we have?"   ->   104

The categorical parser claimed ``Gamma Direct`` as a value of ``broker_channel``
(147 loans). The portfolio-lens resolver, reading the SAME raw question with its
own qualifier/noun grammar, independently matched ``Direct loans`` and narrowed
the population to the ``direct_001`` book as well — 104 of those 147. Two
resolvers, one span, and neither could see the other.

The invariant:

    Once a contiguous span has been claimed as a governed categorical value, the
    tokens INSIDE that span must not independently create another semantic
    claim, unless the grammar explicitly establishes a second meaning.

These tests are generic. Not one of them names "Gamma Direct" as a special case:
the values come from a catalogue in the shape ``execution_receipt.book_values``
produces, and the same rule is exercised on an unrelated vocabulary.
"""

import unittest

from mi_agent import categorical_spans as CS
from mi_agent import portfolio_lens as PL


BOOK = {
    "broker_channel": {"gamma direct": "Gamma Direct",
                       "acquired partners llp": "Acquired Partners LLP",
                       "beta": "Beta"},
    "product_type": {"lump sum": "Lump Sum", "drawdown": "Drawdown"},
    # The scope owner's OWN fields. A value here is the scope reading itself.
    "source_portfolio_id": {"direct_001": "direct_001",
                            "acquired_001": "acquired_001"},
    "source_portfolio_label": {"direct book": "Direct Book"},
}


class TestSpanOwnership(unittest.TestCase):
    def test_a_value_claims_its_own_span(self):
        q = "How many Gamma Direct loans do we have?"
        spans = CS.value_spans(q, BOOK)
        self.assertEqual([q[a:b] for a, b in spans], ["Gamma Direct"])

    def test_an_underscored_identifier_is_one_word_and_claims_nothing(self):
        # `direct_002` is one token spelled with a separator. Counting it as two
        # let a book value mask an explicit cohort id the reader typed.
        book = {"portfolio_cohort": {"direct_002": "direct_002"}}
        self.assertEqual(CS.value_spans("balance for direct_002?", book), ())

    def test_an_ambiguous_value_claims_nothing(self):
        # Two governed fields both carry it -> it has not been claimed by
        # anything, so it may not silence anything either.
        book = {"a": {"lump sum": "Lump Sum"}, "b": {"lump sum": "Lump Sum"}}
        self.assertEqual(CS.value_spans("lump sum loans", book), ())

    def test_a_single_word_value_claims_nothing(self):
        self.assertEqual(CS.value_spans("Beta loans", BOOK), ())

    def test_the_longest_value_wins(self):
        book = {"broker_channel": {"gamma direct": "Gamma Direct",
                                   "gamma direct nominees": "Gamma Direct Nominees"}}
        q = "How many Gamma Direct Nominees loans?"
        self.assertEqual([q[a:b] for a, b in CS.value_spans(q, book)],
                         ["Gamma Direct Nominees"])

    def test_masking_preserves_offsets(self):
        q = "How many Gamma Direct loans do we have?"
        masked = CS.mask_value_spans(q, BOOK)
        self.assertEqual(len(masked), len(q))
        self.assertNotIn("Direct", masked)
        self.assertIn("loans do we have?", masked)

    def test_a_value_of_the_asking_owners_own_field_is_excluded(self):
        q = "show direct book balance"
        self.assertTrue(CS.value_spans(q, BOOK))          # it IS a book value
        self.assertEqual(CS.mask_value_spans(
            q, BOOK, exclude_fields=("source_portfolio_label",)), q)


class TestTheLensStopsReadingAClaimedSpan(unittest.TestCase):
    """The collision itself, at the resolver that caused it."""

    def test_a_categorical_value_no_longer_creates_a_scope_claim(self):
        q = "How many Gamma Direct loans do we have?"
        self.assertEqual(PL.resolve_lens(q).name, PL.LENS_DIRECT)   # unguarded
        self.assertEqual(PL.resolve_lens(q, available_values=BOOK).name,
                         PL.LENS_TOTAL)

    def test_the_precedence_gate_stops_too(self):
        # `mentions_portfolio` decides whether the QUESTION overrides the
        # caller's selection. Left unguarded it would hand the question to
        # `resolve_lens` in place of the caller's scope even once `resolve_lens`
        # itself had stopped reading the span that way.
        q = "How many Gamma Direct loans do we have?"
        self.assertTrue(PL.mentions_portfolio(q))
        self.assertFalse(PL.mentions_portfolio(q, available_values=BOOK))

    def test_the_grammar_may_still_establish_a_second_meaning(self):
        # The second "Direct" is OUTSIDE the claimed span. The question really
        # does carry both claims and both must survive.
        q = "How many Gamma Direct loans are in the Direct book?"
        self.assertEqual(PL.resolve_lens(q, available_values=BOOK).name,
                         PL.LENS_DIRECT)
        self.assertTrue(PL.mentions_portfolio(q, available_values=BOOK))

    def test_an_ordinary_scope_question_is_untouched(self):
        for q, expected in (("How many loans are in the direct book?", PL.LENS_DIRECT),
                            ("How many loans are in the acquired book?", PL.LENS_ACQUIRED),
                            ("What is the total balance?", PL.LENS_TOTAL)):
            self.assertEqual(PL.resolve_lens(q, available_values=BOOK).name, expected)
            self.assertEqual(PL.resolve_lens(q).name, expected)

    def test_the_scope_owners_own_label_still_resolves(self):
        # "Direct Book" is a `source_portfolio_label` value. Masking it would
        # blind the scope owner to its own vocabulary.
        self.assertEqual(PL.resolve_lens("show direct book balance",
                                         available_values=BOOK).name,
                         PL.LENS_DIRECT)

    def test_an_explicit_cohort_id_still_wins(self):
        self.assertEqual(
            PL.resolve_lens("total balance for acquired_001?",
                            available_values=BOOK).cohort_id, "acquired_001")

    def test_no_values_means_no_change(self):
        for q in ("How many Gamma Direct loans do we have?",
                  "How many loans are in the direct book?",
                  "the acquired portfolio"):
            self.assertEqual(PL.resolve_lens(q, available_values=None).name,
                             PL.resolve_lens(q).name)


class TestTheRuleIsGenericNotAPatch(unittest.TestCase):
    """An unrelated vocabulary, same rule, no code that knows about it."""

    def test_a_product_value_containing_a_scope_word_is_claimed_once(self):
        book = {"product_type": {"acquired estate": "Acquired Estate"}}
        q = "How many Acquired Estate loans do we have?"
        self.assertEqual(PL.resolve_lens(q).name, PL.LENS_ACQUIRED)   # unguarded
        self.assertEqual(PL.resolve_lens(q, available_values=book).name,
                         PL.LENS_TOTAL)

    def test_nothing_in_the_owner_names_a_business_value(self):
        """No vocabulary lives in the owner. Structural, not a substring scan.

        Every string CONSTANT in the module is read from its AST with the
        docstrings removed, so a business word surviving in executable code is
        the only thing that can fail this.
        """
        import ast
        import pathlib

        tree = ast.parse(pathlib.Path("mi_agent/categorical_spans.py").read_text())
        docstrings = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef,
                                 ast.ClassDef)):
                doc = ast.get_docstring(node, clean=False)
                if doc is not None:
                    docstrings.add(doc)
        literals = [n.value.lower() for n in ast.walk(tree)
                    if isinstance(n, ast.Constant) and isinstance(n.value, str)
                    and n.value not in docstrings]
        blob = " ".join(literals)
        for token in ("gamma", "direct", "acquired", "lump", "broker", "region"):
            self.assertNotIn(token, blob,
                             f"{token!r} is a business value in the owner's code")


class TestTheLivePathAnswersTheBrokerNotTheBook(unittest.TestCase):
    """End to end, through the workflow that produced the wrong number.

    The unit tests above prove the rule; this proves the WIRING. Without it the
    owner is inert and the live answer is still 104.
    """

    @staticmethod
    def _frame():
        import pandas as pd

        # Two brokers. "Gamma Direct" straddles both books on purpose: the
        # broker answer (3) differs from the broker-and-direct answer (2), so a
        # scope claim leaking out of the value span changes the number.
        return pd.DataFrame({
            "loan_identifier": ["L1", "L2", "L3", "L4", "L5"],
            "current_outstanding_balance": [100.0, 200.0, 300.0, 400.0, 500.0],
            "broker_channel": ["Gamma Direct", "Gamma Direct", "Gamma Direct",
                               "Delta Partners", "Delta Partners"],
            "source_portfolio_id": ["direct_001", "direct_001", "acquired_001",
                                    "direct_001", "acquired_001"],
            "source_portfolio_type": ["direct", "direct", "acquired",
                                      "direct", "acquired"],
            "source_portfolio_label": ["Direct Book", "Direct Book",
                                       "Acquired Portfolio 1", "Direct Book",
                                       "Acquired Portfolio 1"],
            "portfolio_cohort": ["direct_001", "direct_001", "acquired_001",
                                 "direct_001", "acquired_001"],
        })

    def _run(self, question):
        import pathlib
        import sys

        repo = pathlib.Path(__file__).resolve().parents[1]
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))
        from mi_agent.mi_agent_workflow import run_mi_agent_query

        return run_mi_agent_query(
            question, self._frame(),
            str(repo / "mi_agent" / "mi_semantics_field_registry.yaml"))

    def test_the_broker_question_no_longer_narrows_to_the_book(self):
        r = self._run("How many Gamma Direct loans do we have?")
        self.assertEqual((r.get("spec_obj").filters or {}).get("broker_channel"),
                         "Gamma Direct")
        self.assertNotIn("source_portfolio_id", r.get("spec_obj").filters or {})
        self.assertEqual((r.get("portfolio_lens") or {}).get("name"), "total")

    def test_both_claims_survive_when_the_grammar_states_both(self):
        r = self._run("How many Gamma Direct loans are in the Direct book?")
        self.assertEqual((r.get("spec_obj").filters or {}).get("broker_channel"),
                         "Gamma Direct")
        self.assertEqual((r.get("spec_obj").filters or {}).get("source_portfolio_id"),
                         ["direct_001"])

    def test_a_plain_scope_question_is_untouched(self):
        r = self._run("How many loans are in the direct book?")
        self.assertEqual((r.get("spec_obj").filters or {}).get("source_portfolio_id"),
                         ["direct_001"])


if __name__ == "__main__":
    unittest.main()
