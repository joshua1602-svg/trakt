#!/usr/bin/env python3
"""The arm may change whether Trakt answers; not what it answers.

THE REGRESSION, from the 115-question replay against the deployed build:
"Where was the greatest pipeline attrition?" had ANSWERED, and came back

    parsed dimension(s) neither applied nor rejected: pipeline_stage.
    Refusing to answer with a silently dropped dimension.

The parse did not move — the same question at both commits, over 48 column and
value combinations, produces byte-identical specs, and the deterministic parser
builds a loan-level ranking with no dimension. The concept-merge arm proposed
`pipeline stage` as a dimension, `_apply_to_spec` filled the empty slot, and a
loan-level result has no group columns for it to land in. The invariant was
right to refuse; the axis should never have reached the contract.

Stubbed, so this needs no model, no credit and no network — the arm's one
outbound call is replaced, exactly as `test_the_language_understanding_step_is_
the_variable` replaces it to reproduce the availability refusal.

The rule itself lives with the merge, beside the `share` entry that has always
had it: `question_interpretation/tests/test_a_loan_level_row_is_not_a_group.py`.
"""
from __future__ import annotations

import os
import unittest

from mi_agent import llm_query_parser as LQ
from mi_agent_api.tests.test_stage_movement_query import ask

QUESTION = "Where was the greatest pipeline attrition?"

#: A well-formed reply proposing the axis the deployed arm proposed.
PROPOSES_AN_AXIS = ('{"concepts": [{"kind": "dimension", "term": "pipeline stage",'
                    ' "covers": "pipeline attrition"}]}')


class _Arm:
    """The arm switched on, with its one outbound call replaced."""

    def __init__(self, reply):
        self._reply = reply

    def __enter__(self):
        self._saved = {k: os.environ.get(k) for k in
                       ("MI_AGENT_CONCEPT_MERGE", "ANTHROPIC_API_KEY")}
        os.environ["MI_AGENT_CONCEPT_MERGE"] = "on"
        os.environ["ANTHROPIC_API_KEY"] = "sk-not-used-the-call-is-replaced"
        self._original = LQ._call_llm
        LQ._call_llm = lambda *a, **k: (self._reply, {}, False)
        return self

    def __exit__(self, *exc):
        LQ._call_llm = self._original
        for key, value in self._saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        return False


class TestALoanLevelAnswerKeepsItsShape(unittest.TestCase):

    def test_the_question_answers_with_the_arm_off(self):
        """The baseline the replay recorded, and what a reader had."""
        envelope = ask(QUESTION)
        self.assertTrue(envelope.get("ok"), envelope.get("error"))
        self.assertEqual((envelope.get("spec") or {}).get("aggregation"), "loan_level")
        self.assertIsNone((envelope.get("spec") or {}).get("dimension"))

    def test_a_proposed_axis_does_not_reach_a_loan_level_contract(self):
        with _Arm(PROPOSES_AN_AXIS):
            envelope = ask(QUESTION)
        spec = envelope.get("spec") or {}
        self.assertIsNone(spec.get("dimension"))
        self.assertFalse(spec.get("dimensions"))
        self.assertTrue(envelope.get("ok"), envelope.get("error"))
        self.assertNotIn("neither applied nor rejected",
                         str(envelope.get("error") or ""))

    def test_the_arm_still_ran_and_said_what_it_declined(self):
        """Not silence. The arm reports; the merge declines the role; the
        estate can count both."""
        with _Arm(PROPOSES_AN_AXIS):
            envelope = ask(QUESTION)
        evidence = (envelope.get("metadata") or {}).get("conceptMerge") or {}
        self.assertEqual(evidence.get("status"), "no_change")
        self.assertTrue(evidence.get("proposed"))
        self.assertEqual(evidence.get("applied") or [], [])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
