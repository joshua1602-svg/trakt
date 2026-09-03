"""The alignment probe must find a split region, and must never name one.

This probe is the only one that LOOKS AT category labels — it cannot detect
"LONDON and London are one region reported twice" without comparing them. So
the contract it has to keep is stricter than the others': it looks locally and
emits counts. These pin both halves, because a probe that finds the defect and
leaks the book is not a probe anyone can run.
"""
from __future__ import annotations

import io
import json
import os
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import alignment_probe as AP  # noqa: E402


class TestItFindsARegionReportedTwice(unittest.TestCase):

    def test_the_lenders_own_case_difference_is_caught(self):
        """Acquired in capitals, direct in sentence case, one combined view."""
        out = AP.case_collisions(["LONDON", "London", "Scotland", "SCOTLAND"])
        self.assertFalse(out["clean"])
        self.assertEqual(out["colliding_groups"], 2)
        self.assertEqual(out["distinct_case_insensitive"], 2)
        self.assertEqual(out["widest_collision"], 2)

    def test_a_clean_grouping_is_reported_clean(self):
        out = AP.case_collisions(["London", "Scotland", "South West"])
        self.assertTrue(out["clean"])
        self.assertEqual(out["colliding_groups"], 0)

    def test_whitespace_alone_is_also_a_split(self):
        self.assertFalse(AP.case_collisions(["London", "London "])["clean"])

    def test_empty_and_null_labels_are_not_counted_as_a_region(self):
        out = AP.case_collisions(["London", "", "  ", None, "nan"])
        self.assertTrue(out["clean"])
        self.assertEqual(out["distinct_case_insensitive"], 1)


class TestItNeverEmitsARegionName(unittest.TestCase):
    """The standing rule, on the one probe that has to read the values."""

    def test_the_collision_report_carries_no_label(self):
        out = AP.case_collisions(["LONDON", "London", "Peterborough"])
        blob = json.dumps(out).lower()
        for name in ("london", "peterborough"):
            self.assertNotIn(name, blob)
        self.assertTrue(all(isinstance(v, (int, bool)) for v in out.values()))

    def test_the_labels_are_stripped_before_a_result_is_reported(self):
        self.assertEqual(AP._strip_local({"ok": True, "_labels": ["London"]}),
                         {"ok": True})

    def test_a_figure_in_a_refusal_is_redacted_and_a_date_is_not(self):
        said = AP._redact("No loans match; the book stands at £562,900,000 "
                          "as at 2026-06-30")
        self.assertNotIn("562", said)
        self.assertIn("2026-06-30", said)


class TestItReadsWhatTheAnswerDeclares(unittest.TestCase):

    def test_the_reporting_date_comes_from_governed_provenance(self):
        found = AP._reporting_dates({
            "governance": {"snapshot": {"reporting_date": "2026-06-30"}},
            "metadata": {"asOfDate": "2026-06-30"},
            "sourceNotes": [{"detail": "Governed snapshots: 2026-05-31 → 2026-06-30"}]})
        self.assertEqual(found, ["2026-06-30", "2026-05-31"])

    def test_an_answer_declaring_nothing_yields_nothing(self):
        self.assertEqual(AP._reporting_dates({"answer": "as at 2026-06-30"}), [])

    def test_category_labels_come_off_the_grouped_axis(self):
        labels = AP._category_labels({"artifacts": [
            {"xKey": "region", "rows": [{"region": "LONDON", "value": 1},
                                        {"region": "London", "value": 2}]}]})
        self.assertEqual(labels, ["LONDON", "London"])


class TestTheVerdictsSayWhatWasFound(unittest.TestCase):

    def _run(self, responses):
        calls = iter(responses)

        def _fake(base, token, question, *, lens, portfolio, timeout):
            return next(calls)

        with mock.patch.object(AP, "_ask", side_effect=_fake):
            return AP.probe_cut_off_alignment("b", "t", "p", 1.0)

    def _res(self, dates):
        return {"ok": True, "http": 200, "transport_error": "",
                "error_code": None, "reason": "", "ms": 1,
                "parsed_filters": [], "applied_filters": [],
                "reporting_dates": dates, "_labels": []}

    def test_two_different_cut_offs_are_named_in_the_verdict(self):
        out = self._run([self._res(["2026-06-30"]),      # total
                         self._res(["2026-06-30"]),      # direct
                         self._res(["2026-01-12"])])     # acquired
        self.assertIn("DIFFERENT AS-OF DATES", out["verdict"])
        self.assertIn("2026-01-12", out["verdict"])

    def test_matching_cut_offs_say_so(self):
        out = self._run([self._res(["2026-06-30"])] * 3)
        self.assertIn("same as-of date", out["verdict"])

    def test_a_lens_that_declares_nothing_is_not_read_as_agreement(self):
        """The failure that would matter most: silence reported as a pass."""
        out = self._run([self._res([]), self._res([]), self._res([])])
        self.assertIn("not established", out["verdict"])


if __name__ == "__main__":
    unittest.main()
