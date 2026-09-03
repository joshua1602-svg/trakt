#!/usr/bin/env python3
"""A route that already refused keeps the reason it wrote.

WHY THIS FILE EXISTS. Measured on the live book, ten accepted questions refused
with a sentence that named nothing the reader could act on, while the route's
own diagnosis survived only as a code in a warnings array no channel renders:

    route wrote:  "I could not rank movement by region: no category moved that
                   way."
    reader saw:   "I understood that you asked for ranking by region and region,
                   but that could not be applied to the calculation..."
    warnings:     ["ranking unavailable: no_category_moved_that_way", ...]

The facet guard exists to stop a DELIVERED answer standing when something the
reader asked for never reached the calculation. Where the route has already
declined and said why, the guard was overwriting a specific cause with a general
one — and `'broker_channel' cannot be analysed here: the scope does not declare
a single governed asset class` is a sentence someone can act on, while the
generic form is not.

WHAT MUST NOT CHANGE, and is pinned below: the VERDICT. This changes which
sentence leads on an envelope that is already a refusal. It never turns a
refusal into an answer, never turns an answer into a refusal, and never
suppresses the facet message — that is kept as a warning.
"""
from __future__ import annotations

import unittest

from mi_agent_api import mi_service

ROUTE_REASON = ("I could not rank movement by region: no category moved that "
                "way. I have not ranked a different dimension instead.")


def _refused_by_route():
    return {"ok": False, "error": ROUTE_REASON, "answer": ROUTE_REASON,
            "controlledRefusal": True,
            "metadata": {"route": "period_change_analysis",
                         "controlledUnsupported": True},
            "warnings": ["ranking unavailable: no_category_moved_that_way"]}


class TestTheRoutesOwnReasonIsRecognised(unittest.TestCase):

    def test_a_route_refusal_is_detected(self):
        self.assertEqual(mi_service._route_stated_reason(_refused_by_route()),
                         ROUTE_REASON)

    def test_only_the_metadata_mark_is_enough(self):
        env = _refused_by_route()
        del env["controlledRefusal"]
        self.assertEqual(mi_service._route_stated_reason(env), ROUTE_REASON)


class TestNothingElseIsClaimed(unittest.TestCase):
    """The guard must not start deferring to answers, or to plain failures."""

    def test_a_delivered_answer_has_no_reason_to_keep(self):
        env = _refused_by_route()
        env["ok"] = True
        self.assertIsNone(mi_service._route_stated_reason(env))

    def test_an_unmarked_failure_is_not_a_route_refusal(self):
        """An uncontrolled fault must still get the guard's sentence."""
        env = {"ok": False, "error": "boom", "metadata": {"route": "x"}}
        self.assertIsNone(mi_service._route_stated_reason(env))

    def test_a_refusal_with_no_sentence_defers_to_the_guard(self):
        env = _refused_by_route()
        env["error"] = env["answer"] = ""
        self.assertIsNone(mi_service._route_stated_reason(env))

    def test_a_non_dict_is_not_a_refusal(self):
        self.assertIsNone(mi_service._route_stated_reason(None))
        self.assertIsNone(mi_service._route_stated_reason("refused"))


class TestTheGuardStillRefuses(unittest.TestCase):
    """The verdict is untouched: this decides wording, never outcome."""

    def test_the_source_only_guards_the_message_assignment(self):
        import pathlib
        src = pathlib.Path(mi_service.__file__).read_text()
        block = src.split("_own = _route_stated_reason(routed)", 1)[1][:400]
        # `ok` is set False BEFORE the branch, so a refusal stays a refusal
        # whichever sentence wins.
        self.assertIn('routed["error"] = message', block)
        self.assertNotIn('routed["ok"] = True', block)

    def test_the_facet_message_is_still_recorded(self):
        import pathlib
        src = pathlib.Path(mi_service.__file__).read_text()
        after = src.split("_own = _route_stated_reason(routed)", 1)[1][:900]
        self.assertIn('warnings"', after,
                      "the facet message must still reach the warnings, or the "
                      "guard's finding is lost rather than demoted")


if __name__ == "__main__":
    unittest.main()
