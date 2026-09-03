"""The audit must agree with the runtime, and must never decide anything.

TWO CONTRACTS. It resolves with the SAME functions `funded_prep` will call, so
it cannot pass while the runtime fails — an audit with its own copy of the rule
is worse than none, because it is believed. And it proposes without approving:
a suggestion is printed for a human and never written to the taxonomy, which is
the rule `region_taxonomy` already applies to its own LLM proposals.
"""
from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import unittest
from collections import Counter
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import region_vocabulary_audit as A  # noqa: E402
from engine import region_taxonomy as RT  # noqa: E402


def _tax():
    return RT.resolve_taxonomy(None)


class TestItMeasuresWhatTheRuntimeWillDo(unittest.TestCase):

    def test_it_resolves_through_the_taxonomy_itself(self):
        """Not a reimplementation: the audit's verdict for a value must be the
        taxonomy's own."""
        tax = _tax()
        out = A.audit_column(tax, Counter({"SOUTH WEST": 3, "Atlantis": 1}))
        self.assertEqual(out["by_method"]["exact"], 1)
        self.assertEqual(out["by_method"]["unresolved"], 1)
        self.assertEqual([e["key"] for e in out["unresolved_keys"]], ["atlantis"])

    def test_a_fully_governed_column_reads_one_hundred_percent(self):
        out = A.audit_column(_tax(), Counter({"London": 1, "Scotland": 1}))
        self.assertEqual(out["resolution_pct"], 100.0)
        self.assertEqual(out["unresolved_keys"], [])

    def test_the_gaps_are_ordered_biggest_first(self):
        out = A.audit_column(_tax(), Counter({"Atlantis": 2, "Narnia": 90}))
        self.assertEqual([e["key"] for e in out["unresolved_keys"]],
                         ["narnia", "atlantis"])

    def test_two_spellings_of_one_region_are_reported_as_one_key(self):
        """The lender's own case, after the ampersand fix: two spellings, one
        key, and the audit says so rather than counting them as two regions."""
        out = A.audit_column(_tax(), Counter({"Yorkshire & Humberside": 4,
                                              "YORKSHIRE AND HUMBERSIDE": 7}))
        self.assertEqual(out["resolution_pct"], 100.0)
        self.assertEqual(sorted(out["spellings_sharing_one_key"]),
                         ["yorkshire and humberside"])

    def test_a_null_token_is_not_a_region(self):
        import pandas as pd
        frame = pd.DataFrame({"region": ["London", "", "  ", "nan", None,
                                         "NULL"]})
        self.assertEqual(dict(A._distinct_values(frame, "region")),
                         {"London": 1})


class TestItProposesButNeverDecides(unittest.TestCase):

    def test_a_suggestion_is_by_token_overlap_and_nothing_else(self):
        self.assertEqual(A.suggest(_tax(), "west midlands region"),
                         "West Midlands")

    def test_it_offers_nothing_where_no_word_is_shared(self):
        """"Humberside" shares no token with "Yorkshire and The Humber", and
        the suggester must say so rather than reach for a nearby region."""
        self.assertIsNone(A.suggest(_tax(), "humberside"))
        self.assertIsNone(A.suggest(_tax(), "atlantis"))

    def test_the_taxonomy_file_is_never_written(self):
        path = Path(RT.DEFAULT_CONFIG_PATH)
        before = path.read_bytes()
        A.suggest(_tax(), "atlantis")
        A.audit_column(_tax(), Counter({"Atlantis": 1}))
        self.assertEqual(path.read_bytes(), before)


class TestItRefusesToPretendItMeasuredSomething(unittest.TestCase):
    """Every path that cannot measure says so and exits 3, rather than
    reporting a clean book it never read."""

    def _run(self, frame_csv, extra=()):
        with tempfile.TemporaryDirectory() as d:
            csv = Path(d) / "t.csv"
            csv.write_text(frame_csv)
            out = Path(d) / "o.json"
            err = io.StringIO()
            with redirect_stdout(io.StringIO()):
                try:
                    import contextlib
                    with contextlib.redirect_stderr(err):
                        rc = A.main(["--csv", str(csv), "--out", str(out),
                                     *extra])
                except SystemExit as exc:  # argparse
                    rc = int(exc.code or 0)
            return rc, err.getvalue()

    def test_a_dataset_with_no_region_column_is_not_a_clean_book(self):
        rc, err = self._run("loan_id,balance\n1,100\n")
        self.assertEqual(rc, 3)
        self.assertIn("no-op at runtime", err)

    def test_an_ungoverned_value_makes_the_run_fail(self):
        """So this can gate a pipeline: 'unresolved: 0' is the finish line and
        it is checked, not asserted."""
        rc, _ = self._run("region\nLondon\nAtlantis\n")
        self.assertEqual(rc, 1)

    def test_a_fully_governed_book_passes(self):
        rc, _ = self._run("region\nLondon\nSCOTLAND\nYorkshire & Humberside\n")
        self.assertEqual(rc, 0)


class TestItEmitsNoFigureFromTheBook(unittest.TestCase):

    def test_counts_only_drops_the_region_names(self):
        with tempfile.TemporaryDirectory() as d:
            csv = Path(d) / "t.csv"
            csv.write_text("region\nLondon\nAtlantis\n")
            out = Path(d) / "o.json"
            with redirect_stdout(io.StringIO()) as printed:
                A.main(["--csv", str(csv), "--out", str(out), "--counts-only"])
            written = json.loads(out.read_text())
        blob = json.dumps(written).lower() + printed.getvalue().lower()
        self.assertNotIn("atlantis", blob)
        self.assertNotIn("london", blob)

    def test_row_weights_are_off_unless_asked_for(self):
        with tempfile.TemporaryDirectory() as d:
            csv = Path(d) / "t.csv"
            csv.write_text("region\n" + "Atlantis\n" * 7)
            out = Path(d) / "o.json"
            with redirect_stdout(io.StringIO()) as printed:
                A.main(["--csv", str(csv), "--out", str(out)])
            self.assertNotIn("rows=7", printed.getvalue())


if __name__ == "__main__":
    unittest.main()
