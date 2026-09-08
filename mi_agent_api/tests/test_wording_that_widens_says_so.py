#!/usr/bin/env python3
"""A question that widens the caller's chosen book must say it widened.

WHY THIS FILE EXISTS. Measured on the live book with 'Direct' selected, three of
the 166 accepted questions answered over the WHOLE platform and said nothing:

    Q06C  "Give me a concise overview of the funded portfolio."
    Q18C  "Give me a summary of how the funded book moved over the last month."
    Q23C  "When does the funded book reach the £100m milestone?"

Q06A and Q06B — the same question in other words — answered Direct-only. Same
reader, same selection, two populations, no warning. The governed envelope was
honest (`portfolioScope` said Total) and no reader parses the envelope; what
reaches a CFO is prose and a number.

The route did nothing wrong: a question's own words override the caller's lens
by design, and that is how "balance in the acquired book" works. But widening to
Total is the one override that returns a BIGGER population than the reader
chose, and it is the one that cannot be noticed from the answer itself.

So this discloses widening only. Re-pointing to another named book already
announces itself through the scope owner's own warning.
"""
from __future__ import annotations

import unittest

from mi_agent_api import chat_routing


def _run(question, selected, resolved_has_filters):
    """Drive the disclosure with a resolved lens we control."""
    from mi_agent import portfolio_lens as plens

    meta = {"route": "portfolio_summary", "lensApplied": True}
    envelope = {"ok": True, "metadata": meta, "warnings": []}
    real = chat_routing._resolve_lens

    def _fake(_q, _lens):
        return (plens.lens_from_selection("direct") if resolved_has_filters
                else plens.total_lens() if hasattr(plens, "total_lens")
                else plens.lens_from_selection("total"))

    chat_routing._resolve_lens = _fake
    try:
        return chat_routing._disclose_lens_scope(envelope, question, selected)
    finally:
        chat_routing._resolve_lens = real


class TestWideningIsDisclosed(unittest.TestCase):

    def test_the_question_that_answered_whole_book_silently(self):
        env = _run("Give me a concise overview of the funded portfolio.",
                   "direct", resolved_has_filters=False)
        joined = " ".join(env["warnings"])
        self.assertIn("Scope widened by the question", joined)
        self.assertIn("Direct", joined)
        self.assertIn("NOT Direct-only", joined)

    def test_the_warning_says_what_to_do_about_it(self):
        env = _run("Give me a concise overview of the funded portfolio.",
                   "direct", resolved_has_filters=False)
        self.assertIn("Name the book in the question",
                      " ".join(env["warnings"]))

    def test_the_answer_is_not_refused(self):
        """Disclosure, not refusal — the figures are right for what they cover."""
        env = _run("Give me a concise overview of the funded portfolio.",
                   "direct", resolved_has_filters=False)
        self.assertTrue(env["ok"])


class TestNothingElseIsAnnounced(unittest.TestCase):
    """The guard is narrow: it must not add noise to answers that were fine."""

    def test_a_lens_that_held_says_nothing(self):
        env = _run("Summarise the portfolio.", "direct",
                   resolved_has_filters=True)
        self.assertEqual(env["warnings"], [])

    def test_total_was_requested_so_there_is_nothing_to_widen_from(self):
        env = _run("Summarise the portfolio.", "total",
                   resolved_has_filters=False)
        self.assertEqual(env["warnings"], [])

    def test_no_selection_at_all_is_not_a_widening(self):
        env = _run("Summarise the portfolio.", None,
                   resolved_has_filters=False)
        self.assertEqual(env["warnings"], [])


if __name__ == "__main__":
    unittest.main()
