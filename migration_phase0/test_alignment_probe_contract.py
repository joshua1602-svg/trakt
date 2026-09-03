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


class TestAnAbsenceOfEvidenceIsNotAVerdict(unittest.TestCase):
    """The other first-run defect: three case forms that all REFUSE agree
    perfectly, and the probe read that agreement as "case-insensitive"."""

    def _run(self, ok):
        def _fake(base, token, question, *, lens, portfolio, timeout):
            return {"ok": ok, "http": 200, "transport_error": "",
                    "error_code": None if ok else "CALCULATION_FAILED",
                    "reason": "", "ms": 1, "parsed_filters": [],
                    "applied_filters": ["region"], "reporting_dates": [],
                    "_labels": []}

        with mock.patch.object(AP, "_ask", side_effect=_fake):
            return AP.probe_case_on_the_way_in("b", "t", "p", 1.0)

    def test_three_refusals_that_agree_are_not_case_insensitivity(self):
        self.assertIn("not established", self._run(False)["verdict"])

    def test_three_answers_that_agree_are(self):
        self.assertEqual(self._run(True)["verdict"], "case-insensitive")


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


class TestItAsksWhenTheDataWasCutNotWhatItIsLabelled(unittest.TestCase):
    """THE THIRD FIRST-RUN DEFECT, and the worst of them.

    C compared `reporting_date` across the lenses and found them equal — which
    they are BY CONSTRUCTION. `_platform_reporting_date` picks the first column
    present from ("reporting_date", "data_cut_off_date", "cut_off_date"), so a
    tape carrying both never has the second read. The probe was measuring the
    chain's first preference and reporting it as the two books agreeing, while
    the lender's own tape has them cut six months apart.
    """

    def _run(self, answer_payload, surfaces):
        def _fake_ask(base, token, question, *, lens, portfolio, timeout,
                      keep_payload=False):
            return {"ok": True, "http": 200, "transport_error": "",
                    "error_code": None, "reason": "", "ms": 1,
                    "parsed_filters": [], "applied_filters": [],
                    "reporting_dates": ["2026-06-30"], "_labels": [],
                    "_payload": dict(answer_payload)}

        with mock.patch.object(AP, "_ask", side_effect=_fake_ask):
            return AP.probe_cut_off_alignment(
                "b", "t", "p", 1.0, fetch=lambda *a: dict(surfaces))

    def test_a_cut_off_hidden_behind_a_uniform_reporting_date_is_found(self):
        out = self._run({"governance": {"snapshot": {
            "reporting_date": "2026-06-30",
            "data_cut_off_date": "2025-11-30"}}}, {})
        self.assertIn("NOT CUT WHEN THE ANSWER SAYS", out["verdict"])
        self.assertIn("2025-11-30", out["verdict"])

    def test_no_cut_off_anywhere_is_the_finding_not_a_pass(self):
        """A uniform reporting date is not evidence the books are aligned."""
        out = self._run({"governance": {"snapshot": {
            "reporting_date": "2026-06-30"}}}, {})
        self.assertIn("NO DATA CUT-OFF IS SURFACED", out["verdict"])

    def test_it_looks_at_the_platform_surfaces_too(self):
        out = self._run({}, {"dataCutOffDate": "2026-05-20"})
        self.assertIn("2026-05-20", out["cut_off_dates_surfaced"])

    def test_it_finds_the_key_however_it_is_spelled_and_however_deep(self):
        found = AP._find_cut_off(
            {"a": {"b": [{"cutOffDate": "2025-11-30T00:00:00"}]}})
        self.assertEqual(found, [("cutOffDate", "2025-11-30")])

    def test_a_reporting_date_is_not_mistaken_for_a_cut_off(self):
        self.assertEqual(AP._find_cut_off({"reporting_date": "2026-06-30"}), [])

    def test_a_timestamped_date_is_not_invisible(self):
        """`\\b` does not exist between the "0" of a date and the "T" of a
        timestamp, so every timestamped date in the platform metadata was
        being read as absent."""
        self.assertEqual(AP._ISO_DATE.findall("2025-11-30T00:00:00Z"),
                         ["2025-11-30"])


class TestTheVerdictsSayWhatWasFound(unittest.TestCase):

    def _run(self, responses):
        calls = iter(responses)

        def _fake(base, token, question, *, lens, portfolio, timeout,
                  keep_payload=False):
            return {**next(calls), "_payload": {}}

        with mock.patch.object(AP, "_ask", side_effect=_fake):
            return AP.probe_cut_off_alignment("b", "t", "p", 1.0,
                                              fetch=lambda *a: {})

    def _res(self, dates):
        return {"ok": True, "http": 200, "transport_error": "",
                "error_code": None, "reason": "", "ms": 1,
                "parsed_filters": [], "applied_filters": [],
                "reporting_dates": dates, "_labels": []}

    def test_agreeing_reporting_dates_are_no_longer_read_as_alignment(self):
        """What C used to assert, kept as the thing that must NOT come back:
        three lenses agreeing on a reporting date proves only that the date
        chain has one first preference."""
        out = self._run([self._res(["2026-06-30"])] * 3)
        self.assertNotIn("same as-of date", out["verdict"])
        self.assertIn("NO DATA CUT-OFF IS SURFACED", out["verdict"])


if __name__ == "__main__":
    unittest.main()
