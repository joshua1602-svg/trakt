"""A grouped ranking is ordered in the direction the question asked for.

Measured on the shipped path before this was fixed, with a broker column the
geographic route never touches:

    "Which broker channel has the smallest balance?"
        Delta Advisers   £49,050,182     <- the LARGEST, first
        Alpha Network    £41,654,473
        Gamma Direct     £40,884,938
        Beta Partners    £40,465,954     <- the answer, last

The spec had already resolved `sort_direction='asc'`; the grouped execution path
sorted `ascending=False` unconditionally, so "smallest" and "largest" returned
byte-identical results led by the largest group. The loan-level ranking path and
`_apply_top_n` both honoured the field — only the grouped path did not.

These tests are the truth check Mutation 4 targets: reverse the direction and
they fail.
"""

import sys
import unittest
from pathlib import Path

import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from mi_agent.mi_agent_workflow import run_mi_agent_query

_SEM = str(_REPO / "mi_agent" / "mi_semantics_field_registry.yaml")


def _frame():
    # Four brokers with strictly ordered balances, so first-row identity is a
    # fact about direction and not about a tie-break.
    return pd.DataFrame({
        "loan_identifier": [f"L{i}" for i in range(1, 9)],
        "current_outstanding_balance": [400.0, 400.0, 300.0, 300.0,
                                        200.0, 200.0, 100.0, 100.0],
        "broker_channel": ["Delta", "Delta", "Charlie", "Charlie",
                           "Bravo", "Bravo", "Alpha", "Alpha"],
        "reporting_date": ["2026-06-30"] * 8,
    })


def _rows(question):
    result = run_mi_agent_query(question, _frame(), _SEM)
    qr = result.get("query_result")
    assert qr is not None, result.get("error")
    return qr.to_dict().get("data") or []


class TestTheDirectionIsHonoured(unittest.TestCase):
    def test_largest_leads_with_the_largest(self):
        rows = _rows("Which broker channel has the largest balance?")
        self.assertEqual(rows[0]["broker_channel"], "Delta")
        self.assertEqual(rows[0]["current_outstanding_balance_sum"], 800.0)

    def test_smallest_leads_with_the_smallest(self):
        rows = _rows("Which broker channel has the smallest balance?")
        self.assertEqual(rows[0]["broker_channel"], "Alpha")
        self.assertEqual(rows[0]["current_outstanding_balance_sum"], 200.0)

    def test_the_two_directions_are_not_the_same_answer(self):
        """The regression in one line: they used to be identical."""
        self.assertNotEqual(
            _rows("Which broker channel has the largest balance?")[0]["broker_channel"],
            _rows("Which broker channel has the smallest balance?")[0]["broker_channel"])

    def test_every_group_is_still_returned_in_both_directions(self):
        """Ordering, not filtering: a direction never drops a group."""
        for question in ("Which broker channel has the largest balance?",
                         "Which broker channel has the smallest balance?"):
            rows = _rows(question)
            self.assertEqual({r["broker_channel"] for r in rows},
                             {"Alpha", "Bravo", "Charlie", "Delta"}, question)

    def test_a_plain_breakdown_is_unchanged(self):
        """`sort_direction` defaults to descending, so a question that asked for
        no ordering is presented exactly as it always was."""
        rows = _rows("Show balance by broker channel.")
        self.assertEqual([r["broker_channel"] for r in rows],
                         ["Delta", "Charlie", "Bravo", "Alpha"])

    def test_the_share_of_total_does_not_depend_on_row_order(self):
        """`concentration_pct` is a per-row share, so ascending output carries
        the same shares as descending — the reason reordering is safe here."""
        asc = {r["broker_channel"]: round(r.get("concentration_pct", 0), 6)
               for r in _rows("Which broker channel has the smallest balance?")}
        desc = {r["broker_channel"]: round(r.get("concentration_pct", 0), 6)
                for r in _rows("Which broker channel has the largest balance?")}
        self.assertEqual(asc, desc)


class TestTheAnswerNamesTheGroup(unittest.TestCase):
    """A ranking question is answered by naming the group, not by "7 groups"."""

    def _answer(self, question):
        from mi_agent_api.adapters import adapt_workflow_result

        result = run_mi_agent_query(question, _frame(), _SEM)
        payload = adapt_workflow_result(result, as_of="2026-06-30")
        return payload.get("answer") or ""

    def test_largest_names_the_largest_group(self):
        answer = self._answer("Which broker channel has the largest balance?")
        self.assertIn("Delta", answer)
        self.assertIn("highest", answer.lower())

    def test_smallest_names_the_smallest_group(self):
        answer = self._answer("Which broker channel has the smallest balance?")
        self.assertIn("Alpha", answer)
        self.assertIn("lowest", answer.lower())

    def test_a_plain_breakdown_keeps_the_neutral_lead(self):
        answer = self._answer("Show balance by broker channel.")
        self.assertNotIn("highest", answer.lower())
        self.assertIn("group", answer.lower())


if __name__ == "__main__":
    unittest.main()
