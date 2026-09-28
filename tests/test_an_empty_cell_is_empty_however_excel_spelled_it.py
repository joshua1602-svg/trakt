"""549 empty cells were counted as 549 values, and nobody would have noticed.

The live pack's `LoanExtract One [LoanExtractOne]` carries 568 loans. Its
`Amount Of Repayment` column holds 19 amounts; the other 549 cells are blank —
but Excel handed them over as ``''`` rather than as blanks, because the cells
are text-formatted. ``dropna()`` keeps an empty string, so the column profiled
as::

    file_profiler   non_null=568/568  null_rate=0.0  type=string  min/max=''/''
    column_evidence null_count=0/568  null_rate=0.0  type=enum

A decimal column that is 96.6% empty was offered to the mapping stage as a
FULLY POPULATED CATEGORY. Read raw it is 3% numeric with twenty distinct
values in 568 rows, and that scores as a stage/status enum — so the wrong type
was not a slip, it was the correct conclusion from a miscounted column.

THIS ONE IS NOT A CRASH. Everything above completes, writes its artefacts and
reports success. It is a wrong answer that mapping decisions are made from,
and it reaches MI as a repayment field mapped as text: no minimum, no maximum,
no arithmetic, and a coverage figure claiming the lender supplied every value.
It was found only because a DIFFERENT fault in the same column — comparing
``''`` against a float — stopped the run loudly enough to look.

ONLY WHITESPACE IS NOTHING. ``"N/A"``, ``"n/k"`` and ``"-"`` are the lender
asserting that a value does not apply. That is a different fact from a cell
nobody filled in, and it stays visible as itself.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from engine.onboarding_agent import column_evidence as ce
from engine.onboarding_agent.file_profiler import blanks_as_null, profile_column


def _live_column():
    """19 amounts and 549 empty strings, as the delivery actually arrived."""
    return [1000.0 + i for i in range(19)] + [""] * 549


def _profile(series: pd.Series):
    return profile_column("", "", "", "Amount Of Repayment", series)


class TestTheLiveColumn:

    def test_the_profiler_counts_the_blanks_as_blank(self):
        p = _profile(pd.Series(_live_column()))
        assert p.non_null_count == 19
        assert p.null_rate == pytest.approx(0.967, abs=0.001)

    def test_the_profiler_reads_it_as_a_number_not_as_text(self):
        p = _profile(pd.Series(_live_column()))
        assert p.inferred_type in ("integer", "decimal")
        assert (p.min_value, p.max_value) == ("1000.0", "1018.0")

    def test_the_evidence_row_agrees_with_the_profile(self):
        """The two independent readings of one column must not disagree."""
        df = pd.DataFrame({"Amount Of Repayment": _live_column()})
        row = ce.build_column_evidence(df, source_file="f.xlsx",
                                       sheet_name="LoanExtractOne")[0]
        assert row["null_count"] == 549
        assert row["null_rate"] == pytest.approx(0.9665, abs=0.0001)
        assert row["data_type_guess"] == "amount"
        assert (row["min_value"], row["max_value"]) == ("1000.0", "1018.0")


class TestAnAssertionIsNotAnAbsence:
    """A lender saying "not applicable" is data. A cell nobody filled is not."""

    @pytest.mark.parametrize("kept", ["N/A", "n/k", "-", "TBC", "unknown", "0"])
    def test_a_written_value_survives(self, kept):
        out = blanks_as_null(pd.Series([1.0, kept, 2.0]))
        assert list(out)[1] == kept

    @pytest.mark.parametrize("blank", ["", " ", "   ", "\t", "\n"])
    def test_whitespace_becomes_blank(self, blank):
        out = blanks_as_null(pd.Series([1.0, blank, 2.0]))
        assert out.isna().sum() == 1

    def test_a_zero_is_a_value(self):
        """The most important thing this must never do."""
        p = _profile(pd.Series([0.0, 0.0, 1.0]))
        assert p.non_null_count == 3
        assert p.null_rate == 0.0


class TestNothingElseMoves:

    def test_a_numeric_column_is_untouched(self):
        s = pd.Series([1.0, 2.0, np.nan])
        pd.testing.assert_series_equal(blanks_as_null(s), s)

    def test_a_date_column_is_untouched(self):
        s = pd.to_datetime(pd.Series(["2026-08-31", None]))
        pd.testing.assert_series_equal(blanks_as_null(s), s)

    def test_a_genuine_enum_is_still_an_enum(self):
        """Correcting the counts must not retype a real category."""
        df = pd.DataFrame({"Month Run": ["August"] * 500 + ["July"] * 68})
        row = ce.build_column_evidence(df, source_file="f.xlsx", sheet_name="s")[0]
        assert row["data_type_guess"] == "enum"

    def test_a_full_amount_column_is_still_an_amount(self):
        df = pd.DataFrame({"Current Balance": [100000.0 + i for i in range(568)]})
        row = ce.build_column_evidence(df, source_file="f.xlsx", sheet_name="s")[0]
        assert row["data_type_guess"] == "amount"
        assert row["null_rate"] == 0.0

    def test_an_identifier_is_still_an_identifier(self):
        df = pd.DataFrame({"Loan Policy Number": [f"ERE{i:04d}" for i in range(568)]})
        row = ce.build_column_evidence(df, source_file="f.xlsx", sheet_name="s")[0]
        assert row["data_type_guess"] == "identifier"

    def test_a_column_that_is_entirely_blank_reports_as_such(self):
        df = pd.DataFrame({"Never Supplied": [""] * 100})
        row = ce.build_column_evidence(df, source_file="f.xlsx", sheet_name="s")[0]
        assert row["null_count"] == 100
        assert row["null_rate"] == 1.0
        assert (row["min_value"], row["max_value"]) == ("", "")
