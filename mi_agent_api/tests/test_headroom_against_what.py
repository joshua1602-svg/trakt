#!/usr/bin/env python3
"""Headroom is a relative noun, and two owners answered it.

THE OBSERVATION, from the 100-question atomic perimeter run. F044:

    "What is the current NNEG headroom on the funded book?"
        → route `risk_limits`
        → *"4 passed, 0 warning(s), 7 breach(es), 1 need review, 3 unavailable.
           Nearest to limit: Top 3 brokers (-55.0 pp headroom). Largest
           concentration: Top 3 brokers at 100.0%."*

That is a governing-document concentration report, delivered confidently, to a
question about collateral shortfall on a lifetime book. It is not a wrong
number — it is a wrong SUBJECT, and it is the worst row in that bank for
exactly that reason: a reader cannot detect it from the answer, because the
answer is internally correct. F045 ("percentage NNEG headroom") behaved
identically.

THE CAUSE. `_RISK_LIMIT_RE` contained a bare `\\bheadroom\\b`. Headroom means
"distance to a bound", and the question is always *against what*: the
risk-limit route means distance to a Schedule 8 concentration limit; NNEG
headroom means the equity above the balance. One word, two governed subjects,
and the recogniser at priority 100 took every sentence containing it.

WHAT CHANGED, AND WHAT DID NOT. The risk-limit route keeps every question that
is about limits, INCLUDING every bare-headroom phrasing the 115-question bank
exercises — that vocabulary is untouched, and the class below asserts it. What
it no longer does is claim a headroom question whose subject is a governed
concept it does not own.

THE SECOND DEFECT, in the same family. `mi_agent_workflow._UNSUPPORTED_CONCEPTS`
declined NNEG on the basis of a field named `nneg_flag` — which exists nowhere
in this estate. The registered field is `negative_equity_guarantee`; no tape,
contract or registry has ever carried `nneg_flag`. So the refusal was
unconditional and its stated reason ("this book does not report it") was
unfalsifiable: it would have said the same on a tape that reported it perfectly.
The gate is a DATA-AVAILABILITY gate, so it is pointed at the field that exists.

WHAT IS DELIBERATELY NOT BUILT HERE. No NNEG economics, and no grouped NNEG.
`evolution._nneg_metrics` and `snapshots._risk_tile` already own the aggregate
point-in-time definitions; exposing them on the query surface is a capability
decision, recorded in migration_phase0/MI_ATOMIC_PERIMETER_PHASE0.md and not
taken in Phase 1. Phase 1's requirement is only this: never answer an NNEG
question with a concentration report.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mi_agent_api.tests.test_stage_movement_query import ask     # noqa: E402

#: Routes that answer about the governing document's concentration limits.
LIMIT_ROUTES = frozenset({"risk_limits", "concentration_analysis"})

NNEG_QUESTIONS = (
    "What is the current NNEG headroom on the funded book?",
    "What is the percentage NNEG headroom?",
    "What is the NNEG headroom by product type?",
    "What is the no-negative-equity headroom?",
)


def route_of(envelope) -> str:
    return (envelope.get("metadata") or {}).get("route")


class TestANnegQuestionIsNeverAnsweredByTheLimitMonitor(unittest.TestCase):
    """The acceptance the brief set for F044: a governed NNEG answer, or an
    explicit governed refusal. Never concentration output."""

    def test_no_nneg_question_reaches_a_limit_route(self):
        for question in NNEG_QUESTIONS:
            with self.subTest(question=question):
                self.assertNotIn(route_of(ask(question)), LIMIT_ROUTES)

    def test_no_nneg_question_is_answered_with_a_limit_report(self):
        """Belt and braces on the WORDS, not only the route name: the failing
        answer named breaches and concentrations, and no NNEG question may
        produce those however it is routed."""
        for question in NNEG_QUESTIONS:
            with self.subTest(question=question):
                answer = (ask(question).get("answer") or "").lower()
                for tell in ("breach", "concentration limit", "nearest to limit"):
                    self.assertNotIn(tell, answer)

    def test_the_refusal_that_stands_is_explicit(self):
        """Phase 1 leaves NNEG unanswered. That is allowed — but it has to be a
        refusal a reader can act on, not a figure about something else."""
        envelope = ask("What is the current NNEG headroom on the funded book?")
        self.assertFalse(envelope.get("ok"), envelope.get("answer"))
        self.assertFalse(envelope.get("artifacts"))


class TestTheLimitRouteKeepsItsOwnQuestions(unittest.TestCase):
    """The regression risk this fix carries, asserted directly. `headroom` is a
    limit word in every one of these, and the route must still claim them."""

    LIMIT_QUESTIONS = (
        "How much headroom do we have on the concentration limits?",
        "What is the headroom on the largest broker limit?",
        "Which limits are we closest to breaching?",
        "Are we within our Schedule 8 limits?",
        "Show me the risk limit tests.",
    )

    def test_limit_questions_still_reach_the_limit_monitor(self):
        for question in self.LIMIT_QUESTIONS:
            with self.subTest(question=question):
                self.assertIn(route_of(ask(question)), LIMIT_ROUTES)


class TestTheGateNamesAFieldThatExists(unittest.TestCase):
    """`_UNSUPPORTED_CONCEPTS` declines a concept when its FIELD is absent from
    the data. That refusal is only checkable if the field is a name something
    reads — otherwise the gate says "this book does not report it" about a
    column no book could ever report, and would say it on a tape that reported
    the concept perfectly.

    The test is a READER CENSUS rather than a registry lookup, because the
    registry is the wrong oracle here: `days_in_arrears`, `accrued_interest` and
    `indexed_valuation_amount` are real tape columns that the curated MI
    registry does not carry, and a registry test would have condemned them too.
    Measured across the estate, excluding the gate's own line and the golden
    questions generated from it:

        arrears_balance 70   allocated_losses 53   default_amount 43
        indexed_loan_to_value 20   protected_equity_flag 18
        recoveries_in_period 12   days_in_arrears 10   accrued_interest 8
        indexed_valuation_amount 1
        nneg_flag 0          credit_score 0
    """

    #: `credit_score` is the same defect and is NOT fixed here: it is outside
    #: the six suspect rows this phase is scoped to, and unlike NNEG there is no
    #: evidence in hand about what it should name instead. Recorded so it is a
    #: decision someone makes rather than one that keeps happening.
    KNOWN_PHANTOM_NOT_IN_PHASE_1 = frozenset({"credit_score"})

    @staticmethod
    def _readers(field: str) -> int:
        """How many places outside the gate itself name this field."""
        import subprocess

        out = subprocess.run(
            ["grep", "-rn", r"\b%s\b" % field, "--include=*.py",
             "--include=*.yaml", str(_REPO_ROOT)],
            capture_output=True, text=True).stdout.splitlines()
        return len([l for l in out
                    if "mi_agent_workflow.py" not in l
                    and "golden_questions" not in l
                    and "/tests/" not in l])

    def test_the_nneg_gate_declines_on_a_field_something_reads(self):
        from mi_agent.mi_agent_workflow import _UNSUPPORTED_CONCEPTS

        nneg = [fields for _p, concept, fields in _UNSUPPORTED_CONCEPTS
                if concept == "NNEG"]
        self.assertTrue(nneg, "the NNEG gate entry disappeared")
        for field in nneg[0]:
            with self.subTest(field=field):
                self.assertNotEqual(field, "nneg_flag")
                self.assertGreater(self._readers(field), 0)

    def test_no_new_phantom_joins_the_gate(self):
        """The whole list, so the next one is caught when it is added rather
        than when a bank row surfaces it two years later."""
        from mi_agent.mi_agent_workflow import _UNSUPPORTED_CONCEPTS

        phantoms = {field for _p, _c, fields in _UNSUPPORTED_CONCEPTS
                    for field in fields if self._readers(field) == 0}
        self.assertEqual(phantoms, self.KNOWN_PHANTOM_NOT_IN_PHASE_1)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
