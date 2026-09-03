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


class TestAChartAndATableAreNotTwoRegions(unittest.TestCase):
    """THE FALSE POSITIVE THIS PROBE PRODUCED ON ITS FIRST LIVE RUN.

    A grouped answer returns a chart AND a table over the same rows. Pooling
    their labels made every region appear twice and reported a clean book as
    "one region reported twice" — 20 labels, 10 distinct, widest exactly 2. A
    duplicate only means anything WITHIN one artefact.
    """

    def test_the_same_regions_in_a_chart_and_a_table_are_clean(self):
        payload = {"artifacts": [
            {"xKey": "region", "rows": [{"region": "London"},
                                        {"region": "Scotland"}]},
            {"xKey": "region", "rows": [{"region": "London"},
                                        {"region": "Scotland"}]}]}
        out = AP.worst_collision(AP._category_labels(payload))
        self.assertTrue(out["clean"])
        self.assertEqual(out["artifacts_read"], 2)

    def test_a_split_inside_one_table_is_still_caught(self):
        payload = {"artifacts": [
            {"xKey": "region", "rows": [{"region": "LONDON"},
                                        {"region": "London"}]},
            {"xKey": "region", "rows": [{"region": "Scotland"}]}]}
        out = AP.worst_collision(AP._category_labels(payload))
        self.assertFalse(out["clean"])
        self.assertEqual(out["colliding_groups"], 1)

    def test_an_answer_with_no_grouped_artefact_is_not_measured(self):
        out = AP.worst_collision(AP._category_labels({"artifacts": []}))
        self.assertFalse(out["measured"])



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
        self.assertEqual(labels, [["LONDON", "London"]])




class TestAHypothesisCanBeExonerated(unittest.TestCase):
    """A probe that can only confirm is not evidence.

    The user's stated hope is that the data cut-off turns out to be a large
    problem. These pin that the probe will say otherwise when the evidence
    says otherwise — including the outcome the code as read predicts, which is
    that a value nothing reads cannot block anything.
    """

    def test_the_three_verdicts_are_distinct_and_none_is_a_default(self):
        self.assertTrue(AP._verdict(True, False, "x").startswith("CONFIRMED"))
        self.assertTrue(AP._verdict(False, True, "x").startswith("EXONERATED"))
        self.assertTrue(AP._verdict(False, False, "x").startswith("NOT ESTABLISHED"))
        self.assertTrue(AP._verdict(True, True, "x").startswith("NOT ESTABLISHED"))

    def _cut_off(self, per_lens_ok, payload, surfaces):
        seen = {"n": 0}

        def _fake_ask(base, token, question, *, lens, portfolio, timeout,
                      keep_payload=False):
            seen["n"] += 1
            name = lens or "total"
            return {"ok": per_lens_ok.get(name, True), "http": 200,
                    "transport_error": "", "error_code": None, "reason": "",
                    "ms": 1, "parsed_filters": [], "applied_filters": [],
                    "reporting_dates": ["2026-06-30"], "_labels": [],
                    "_payload": dict(payload)}

        with mock.patch.object(AP, "_ask", side_effect=_fake_ask):
            return AP.hypothesis_data_cut_off("b", "t", "p", 1.0,
                                              fetch=lambda *a: dict(surfaces))

    def test_a_cut_off_nothing_reads_is_reported_as_not_blocking(self):
        """The outcome the source predicts, and the probe must be willing to
        return it."""
        out = self._cut_off({}, {"governance": {"snapshot": {
            "reporting_date": "2026-06-30"}}}, {})
        self.assertIn("NOT ESTABLISHED", out["verdict"])
        self.assertIn("NOT BLOCKING", out["verdict"])
        self.assertIn("DISCLOSURE failure", out["verdict"])

    def test_a_combined_failure_over_two_working_books_is_confirmation(self):
        out = self._cut_off({"total": False, "direct": True, "acquired": True},
                            {}, {})
        self.assertIn("CONFIRMED", out["verdict"])

    def test_an_aligned_book_exonerates_the_hypothesis(self):
        out = self._cut_off({}, {"governance": {"snapshot": {
            "data_cut_off_date": "2026-06-30"}}}, {})
        self.assertIn("EXONERATED", out["verdict"])


class TestRegionIsJudgedByComparison(unittest.TestCase):

    def _region(self, splits, ok=True, catalogue=None):
        def _fake_ask(base, token, question, *, lens, portfolio, timeout,
                      keep_payload=False):
            name = lens or "total"
            labels = [[l for l in splits.get(name, ["London", "Scotland"])]]
            return {"ok": ok, "http": 200, "transport_error": "",
                    "error_code": None, "reason": "", "ms": 1,
                    "parsed_filters": [], "applied_filters": ["collateral_geography"],
                    "reporting_dates": [], "_labels": labels}

        with mock.patch.object(AP, "_ask", side_effect=_fake_ask):
            return AP.hypothesis_region_vocabulary(
                "b", "t", "p", 1.0, fetch=lambda *a: (catalogue or {}))

    def test_a_split_only_when_combined_confirms_harmonisation(self):
        out = self._region({"total": ["LONDON", "London"],
                            "direct": ["London"], "acquired": ["LONDON"]})
        self.assertIn("CONFIRMED", out["verdict"])
        self.assertIn("COMBINED", out["verdict"])

    def test_a_clean_split_on_a_raw_binding_is_not_confirmation(self):
        """No region is reported twice — so case is not blocking anything,
        even though MI binds to a raw source column."""
        out = self._region({})
        self.assertIn("NOT ESTABLISHED", out["verdict"])
        self.assertFalse(out["bound_to_harmonised_field"])

    def test_a_harmonised_binding_with_no_split_exonerates(self):
        def _fake_ask(base, token, question, *, lens, portfolio, timeout,
                      keep_payload=False):
            return {"ok": True, "http": 200, "transport_error": "",
                    "error_code": None, "reason": "", "ms": 1,
                    "parsed_filters": [],
                    "applied_filters": ["canonical_region_reporting"],
                    "reporting_dates": [], "_labels": [["London"]]}

        with mock.patch.object(AP, "_ask", side_effect=_fake_ask):
            out = AP.hypothesis_region_vocabulary(
                "b", "t", "p", 1.0,
                fetch=lambda *a: {"fields": ["canonical_region_reporting"]})
        self.assertIn("EXONERATED", out["verdict"])


if __name__ == "__main__":
    unittest.main()
