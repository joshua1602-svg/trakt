#!/usr/bin/env python3
"""The coverage gate declining is not the system failing, and the record must
say which.

MEASURED 2026-09-03 over 954 live questions: 540 ANSWERED, 125 REFUSED,
**289 ERROR (30.3%)**. That last figure was unreadable, because
`_enforce_semantic_coverage` marked its envelope with nothing that
`_classify_analytical_failure` recognised, so every coverage decline fell
through to CALCULATION_FAILED -- which `mi_query_telemetry._ERROR_CODES`
counts as an ERROR.

So the gate this whole build exists for -- refuse rather than answer over a
concept you cannot account for -- was recorded as the system breaking, every
single time it worked. The 289 is a mixture of genuine unhandled exceptions
and correct governed declines, and an operator reading it cannot tell which
they are looking at or whether it is getting better.

UNSUPPORTED_QUESTION is the existing CAPABILITY code for "I will not answer
that" and carries the same HTTP 200 CALCULATION_FAILED already did. What
changes is the label on the record, not what any caller receives.
"""
from __future__ import annotations

import unittest

from mi_agent_api import mi_service as MS
from mi_agent_api.datasets import load_mi_semantics, semantics_path
from trakt_core.errors import ErrorCode, http_status_for


def _refused_by_coverage(question="How many pipeline cases moved into Offer "
                                  "stage in the last week?"):
    env = {"ok": True, "answer": "...", "metadata": {}, "artifacts": [],
           "question": question}
    MS._stamp_semantic_coverage(env, question=question,
                                semantics=load_mi_semantics(semantics_path()),
                                frame=None)
    return MS._enforce_semantic_coverage(env)


class TestTheGateIsRecordedAsADecline(unittest.TestCase):

    def test_the_fixture_really_is_a_coverage_refusal(self):
        """Guard the guard: if this stops refusing, everything below passes
        vacuously and would report a fix that is no longer exercised."""
        env = _refused_by_coverage()
        self.assertFalse(env.get("ok"))
        self.assertIn("could not confirm it was applied", str(env.get("answer")))

    def test_it_classifies_as_a_capability_decline(self):
        self.assertEqual(MS._classify_analytical_failure(_refused_by_coverage()),
                         ErrorCode.UNSUPPORTED_QUESTION)

    def test_the_telemetry_no_longer_calls_it_an_error(self):
        from operations_control.mi_query_telemetry import _ERROR_CODES
        code = MS._classify_analytical_failure(_refused_by_coverage())
        self.assertNotIn(code, _ERROR_CODES)

    def test_no_caller_sees_a_different_status(self):
        """The point is the label, not the behaviour. Both codes are HTTP 200,
        so a client that renders `ok:false` renders exactly what it did."""
        self.assertEqual(http_status_for(ErrorCode.UNSUPPORTED_QUESTION),
                         http_status_for(ErrorCode.CALCULATION_FAILED))

    def test_the_marker_is_its_own_rather_than_a_borrowed_one(self):
        """`controlledUnsupported` means the estate's declared capability
        boundary. Overloading it would make two different decisions
        indistinguishable in the record for the sake of saving a key."""
        env = _refused_by_coverage()
        meta = env.get("metadata") or {}
        self.assertTrue(meta.get("semanticCoverageRefused"))
        self.assertIsNone(meta.get("controlledUnsupported"))


class TestARealBreakageIsStillABreakage(unittest.TestCase):
    """The boundary. If this fix swallowed genuine failures the 289 would drop
    and nothing would have improved -- the worse outcome, because it would look
    like progress."""

    def test_an_unclassifiable_failure_is_still_CALCULATION_FAILED(self):
        broken = {"ok": False, "error": "something exploded", "metadata": {}}
        self.assertEqual(MS._classify_analytical_failure(broken),
                         ErrorCode.CALCULATION_FAILED)

    def test_that_failure_is_still_reported_as_an_error(self):
        from operations_control.mi_query_telemetry import _ERROR_CODES
        broken = {"ok": False, "error": "something exploded", "metadata": {}}
        self.assertIn(MS._classify_analytical_failure(broken), _ERROR_CODES)

    def test_the_existing_classifications_are_untouched(self):
        cases = [
            ({"metadata": {"controlledUnsupported": True}},
             ErrorCode.UNSUPPORTED_QUESTION),
            ({"metadata": {"unmappedQuestion": True}},
             ErrorCode.AMBIGUOUS_QUESTION),
            ({"metadata": {}, "validation": {"errors": ["no rows matched"]}},
             ErrorCode.NO_MATCHING_RECORDS),
        ]
        for payload, expected in cases:
            with self.subTest(expected=expected):
                self.assertEqual(MS._classify_analytical_failure(payload),
                                 expected)


if __name__ == "__main__":
    unittest.main()
