#!/usr/bin/env python3
"""Gate 2 typing must recognise a text column however pandas labels it.

The ERE acquired tape reported two parse failures that were not parse
failures:

  * ``"0.00%"`` came back null. Zero protected equity is a real, known value.
  * a blank second-borrower date of birth was counted as a failed parse rather
    than an absent value.

Both had one cause. Every text branch in ``canonical_transform`` asked
``series.dtype == object``; pandas 3 reads a text column as the new ``str``
dtype, so none of those branches ran and the values went straight to a numeric
coercion with their per-cent signs still attached.

What makes this worth a test of its own is that neither symptom looks like a
dtype problem from the outside. A column read from CSV, from Excel, or built in
a test each arrive with a different dtype, and typing has to behave the same
for all of them — so the dtypes are enumerated here rather than left to
whichever one the test's own DataFrame constructor happens to produce.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from engine.gate_2_transform import canonical_transform as ct

#: How a text column can turn up. ``object`` is what a hand-built frame used to
#: give and the only one the old checks covered; ``str`` is what pandas 3 gives
#: for a column read from a file, which is every real delivery.
TEXT_DTYPES = ("object", "string", "str")


def _column(values, dtype):
    return pd.Series(values, dtype=dtype)


class TestTextDtypesAreAllRecognised(unittest.TestCase):

    def test_every_text_dtype_is_text(self):
        for dtype in TEXT_DTYPES:
            with self.subTest(dtype=dtype):
                self.assertTrue(ct._is_text(_column(["20.00%"], dtype)))

    def test_a_numeric_column_is_not_text(self):
        """The predicate must not widen: a number column takes the fast path."""
        self.assertFalse(ct._is_text(pd.Series([20.0, 30.0])))
        self.assertFalse(ct._is_text(pd.Series([1, 2], dtype="Int64")))


class TestZeroPercentSurvives(unittest.TestCase):

    def test_zero_percent_is_zero_whatever_the_dtype(self):
        for dtype in TEXT_DTYPES:
            with self.subTest(dtype=dtype):
                out = ct.to_percentage(_column(["0.00%", "20.00%"], dtype))
                self.assertEqual([float(v) for v in out.tolist()], [0.0, 20.0])

    def test_typing_a_read_column_reports_no_parse_failure(self):
        """End to end through ``apply_types``, on the dtype a real file gives."""
        df = pd.DataFrame({"protected_equity_percentage":
                           _column(["0.00%"], "str")})
        report = ct.apply_types(
            df, {"protected_equity_percentage": {"format": "percentage"}})
        field = report["fields"]["protected_equity_percentage"]
        self.assertEqual(df["protected_equity_percentage"].tolist(), [0.0])
        self.assertEqual(field["parse_failures"], 0)

    def test_a_malformed_percentage_still_fails(self):
        """Nothing here makes typing more forgiving about real bad data."""
        df = pd.DataFrame({"p": _column(["twenty percent"], "str")})
        report = ct.apply_types(df, {"p": {"format": "percentage"}})
        self.assertEqual(report["fields"]["p"]["parse_failures"], 1)


class TestBlankIsAbsentNotFailed(unittest.TestCase):

    def test_a_blank_cell_is_blank_whatever_the_dtype(self):
        for dtype in TEXT_DTYPES:
            with self.subTest(dtype=dtype):
                mask = ct._blank_token_mask(_column(["01/11/1935", "", "  "],
                                                    dtype))
                self.assertEqual(mask.tolist(), [False, True, True])

    def test_a_blank_date_of_birth_is_not_a_parse_failure(self):
        df = pd.DataFrame({"borrower_2_DOB":
                           _column(["01/11/1938", "", "  "], "str")})
        report = ct.apply_types(df, {"borrower_2_DOB": {"format": "date"}})
        field = report["fields"]["borrower_2_DOB"]
        self.assertEqual(df["borrower_2_DOB"].tolist()[0], "1938-11-01")
        self.assertEqual(field["parse_failures"], 0)
        self.assertEqual(field["blank_values"], 2)

    def test_an_unparseable_date_still_fails(self):
        df = pd.DataFrame({"borrower_1_DOB":
                           _column(["01/11/1935", "not-a-date"], "str")})
        report = ct.apply_types(df, {"borrower_1_DOB": {"format": "date"}})
        self.assertEqual(report["fields"]["borrower_1_DOB"]["parse_failures"], 1)

    def test_a_client_null_marker_is_still_content(self):
        """``N/A`` is something a source chose to write. It is not a blank, and
        voiding it would be a semantic decision this layer must not make."""
        df = pd.DataFrame({"borrower_1_DOB": _column(["N/A"], "str")})
        report = ct.apply_types(df, {"borrower_1_DOB": {"format": "date"}})
        self.assertEqual(report["fields"]["borrower_1_DOB"]["parse_failures"], 1)
        self.assertEqual(report["fields"]["borrower_1_DOB"]["blank_values"], 0)


class TestTheOtherTypedReadersAgree(unittest.TestCase):
    """The same predicate governs every typed reader, so none is left behind."""

    def test_an_amount_with_separators_parses(self):
        for dtype in TEXT_DTYPES:
            with self.subTest(dtype=dtype):
                out = ct.to_decimal(_column(["1,250,000.50"], dtype))
                self.assertEqual([float(v) for v in out.tolist()], [1250000.50])

    def test_a_yes_no_column_parses(self):
        for dtype in TEXT_DTYPES:
            with self.subTest(dtype=dtype):
                out = ct.to_bool_yn(_column(["Yes", "no", ""], dtype))
                self.assertEqual(out.tolist()[:2], ["Y", "N"])
                self.assertTrue(pd.isna(out.tolist()[2]))

    def test_a_currency_synonym_resolves(self):
        for dtype in TEXT_DTYPES:
            with self.subTest(dtype=dtype):
                out = ct.to_currency(_column(["ukp", "eur"], dtype))
                self.assertEqual(out.tolist(), ["GBP", "EUR"])

    def test_an_nd_code_is_stripped(self):
        for dtype in TEXT_DTYPES:
            with self.subTest(dtype=dtype):
                out = ct._strip_nd(_column(["ND1", "RHOS"], dtype))
                self.assertTrue(pd.isna(out.tolist()[0]))
                self.assertEqual(out.tolist()[1], "RHOS")


if __name__ == "__main__":
    unittest.main()
