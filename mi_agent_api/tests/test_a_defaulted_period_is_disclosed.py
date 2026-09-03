"""A window nobody chose may be used, but never silently.

THE RULING AND ITS AMENDMENT. `period_change` refused any question naming no
comparison period, because every answer it gives is a delta between two dates
and which two is the meaning of the number. The rule was right about the
danger and wrong about the remedy for one case: the estate's OTHER half of the
same ruling — the measure — substitutes a governed default and records that it
did (`metric_defaulted`), refusing only the BARE question. Amended 2026-09-03
so the period reads the same way.

A RANKED DIMENSION IS A NAMED SUBJECT, the standing a grouping dimension
already has under the measure rule. "Which region grew the most?" says what to
analyse and omits only the window.

WHAT MUST NOT CHANGE, and is pinned below: the disclosure names BOTH DATES and
reaches the reader, not just the metadata. "The latest period" is not a
disclosure — a reader cannot check a window they cannot see — and a declaration
only a machine can read is the exact trace the measure half was faulted for not
leaving.
"""
from __future__ import annotations

import unittest

from mi_agent_api import period_change_route as PCR


_RESOLUTION = {
    "resolution_method": "latest_available_pair",
    "resolved_start_snapshot": {"snapshot_id": "s1",
                                "reporting_date": "2026-05-31"},
    "resolved_end_snapshot": {"snapshot_id": "s2",
                              "reporting_date": "2026-06-30"},
}


class TestTheDisclosureNamesBothDates(unittest.TestCase):

    def test_both_ends_are_named(self):
        said = PCR._defaulted_period_sentence(_RESOLUTION)
        self.assertIn("2026-05-31", said)
        self.assertIn("2026-06-30", said)

    def test_it_says_the_reader_did_not_choose_the_window(self):
        self.assertIn("did not name a period",
                      PCR._defaulted_period_sentence(_RESOLUTION))

    def test_it_reads_the_reference_the_resolution_actually_records(self):
        """`PeriodResolution.to_dict` records each end as a full provenance
        reference under `resolved_*_snapshot`. There is no flat
        `start_reporting_date`: reading one produced a sentence that still
        scanned as a disclosure while naming no dates at all."""
        self.assertEqual(PCR._period_date(
            {"reporting_date": "2026-06-30T00:00:00"}), "2026-06-30")
        self.assertIsNone(PCR._period_date(None))
        self.assertIsNone(PCR._period_date({"snapshot_id": "s"}))

    def test_a_resolution_with_no_dates_does_not_fake_one(self):
        said = PCR._defaulted_period_sentence({"resolution_method": "x"})
        self.assertIn("the latest period", said)
        self.assertNotIn("None", said)


class TestTheDisclosureTravelsWithTheAnswer(unittest.TestCase):
    """A channel that renders only the answer would otherwise show a
    comparison with no window on it at all."""

    def test_it_is_appended_to_the_answer(self):
        out = PCR._disclose_defaulted_period("Region A grew most.", _RESOLUTION)
        self.assertTrue(out.startswith("Region A grew most."))
        self.assertIn("2026-05-31", out)

    def test_the_answer_comes_first(self):
        """The reader asked a question; the first thing they read is its
        answer, not the caveat."""
        out = PCR._disclose_defaulted_period("Region A grew most.", _RESOLUTION)
        self.assertLess(out.index("Region A grew most."),
                        out.index("did not name a period"))

    def test_an_empty_answer_still_carries_the_disclosure(self):
        out = PCR._disclose_defaulted_period("", _RESOLUTION)
        self.assertIn("2026-05-31", out)

    def test_a_none_answer_does_not_become_the_string_none(self):
        out = PCR._disclose_defaulted_period(None, _RESOLUTION)
        self.assertNotIn("None", out)


if __name__ == "__main__":
    unittest.main()
