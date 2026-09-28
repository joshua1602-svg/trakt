"""A blank header cell is not a column called ``nan``, and twice is not once.

The live delivery's second halt::

    ValueError: The truth value of a Series is ambiguous.
    Use a.empty, a.bool(), a.item(), a.any() or a.all().

``redetect_header`` rescans a workbook whose real header is not row 0 and
assigns that row as the columns BY HAND::

    new_cols = [str(v).strip() or f"col_{j}" for j, v in enumerate(...)]
    out.columns = new_cols

Two things ``pandas`` does for a header it reads itself are missing there:

  * it names the blank cells. ``str(float("nan"))`` is ``"nan"`` — a non-empty,
    truthy string — so the ``or f"col_{j}"`` fallback never ran, and a blank
    header cell became a column literally called ``nan``.
  * it suffixes the repeated ones. Two blank cells therefore gave one frame two
    columns of one name.

``frame[name]`` is a DataFrame when the name is duplicated, and this package is
written throughout for a Series. ``central_tape_builder._reference_code_columns``
asks ``s.isna().any() or ...``; on a DataFrame ``.any()`` is a Series, and
testing a Series raises. A static scan of every boolean test in the package
found exactly one site shaped that way — which is to say the loan tape had one
chance in the delivery to survive a lender's spacer column, and did not take it.

THE INVARIANT IS THE POINT, not the two symptoms: a frame this function returns
can be indexed column by column, because everything downstream does that.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from engine.onboarding_agent import central_tape_builder as ctb
from engine.onboarding_agent.source_table_loader import redetect_header


def _workbook(header_row, *data_rows):
    """A sheet whose real header sits below a title band.

    The title band leaves >40% of pandas' own columns as ``Unnamed: N``, which
    is what puts ``redetect_header`` to work in the first place.
    """
    width = len(header_row)
    rows = [[""] * width, list(header_row)] + [list(r) for r in data_rows]
    frame = pd.DataFrame(rows).T.set_axis(
        ["Loan Book"] + [f"Unnamed: {i}" for i in range(1, len(rows))], axis=1).T
    frame.columns = ["Loan Book"] + [f"Unnamed: {i}" for i in range(1, width)]
    return frame


class TestEveryColumnCanBeIndexed:
    """The invariant the rest of the package relies on."""

    def test_a_frame_of_spacer_columns_still_indexes_column_by_column(self):
        out, _row, _failed = redetect_header(_workbook(
            ["Loan Policy Number", "Current Balance", np.nan, np.nan],
            ["ERE0001", 120000.0, "", ""],
            ["ERE0002", 98000.0, "", ""]))
        assert not out.columns.duplicated().any()
        for col in out.columns:
            assert isinstance(out[col], pd.Series), f"{col!r} indexes to a frame"

    def test_the_loan_key_is_found_rather_than_raised_over(self):
        """The failure this actually caused, end to end."""
        out, _row, _failed = redetect_header(_workbook(
            ["Loan Policy Number", "Current Balance", np.nan, np.nan],
            ["ERE0001", 120000.0, "", ""],
            ["ERE0002", 98000.0, "", ""]))
        assert ctb._reference_code_columns(out) == ["Loan Policy Number"]


class TestABlankCellIsNamedForItsPosition:

    @pytest.mark.parametrize("blank", [np.nan, None, "", "   ", pd.NaT])
    def test_no_form_of_empty_becomes_a_column_name(self, blank):
        out, _row, _failed = redetect_header(_workbook(
            ["Loan Policy Number", blank], ["ERE0001", ""], ["ERE0002", ""]))
        assert list(out.columns) == ["Loan Policy Number", "col_1"]

    def test_a_real_header_is_left_exactly_as_written(self):
        """Including its spacing and case — it is the lender's vocabulary."""
        out, _row, _failed = redetect_header(_workbook(
            ["Loan Policy Number", "Month Run"], ["ERE0001", "August"]))
        assert list(out.columns) == ["Loan Policy Number", "Month Run"]


class TestARepeatedHeaderIsSuffixedTheWayPandasWouldHave:

    def test_the_second_use_of_a_name_moves_aside(self):
        out, _row, _failed = redetect_header(_workbook(
            ["Ref", "Balance", "Ref"], ["A", 1.0, "A"]))
        assert list(out.columns) == ["Ref", "Balance", "Ref.1"]

    def test_a_third_use_keeps_counting(self):
        # Numeric data rows, so the header row is the stringiest and wins the
        # rescan — a row of repeated labels scores poorly on distinctness.
        out, _row, _failed = redetect_header(_workbook(
            ["Ref", "Ref", "Ref"], [1.0, 2.0, 3.0]))
        assert list(out.columns) == ["Ref", "Ref.1", "Ref.2"]

    def test_a_column_genuinely_called_ref_1_keeps_its_name(self):
        """The suffix must not take a name the lender is already using."""
        out, _row, _failed = redetect_header(_workbook(
            ["Ref", "Ref.1", "Ref"], [1.0, 2.0, 3.0]))
        assert list(out.columns) == ["Ref", "Ref.1", "Ref.2"]
        assert not out.columns.duplicated().any()


class TestAWorkbookThatWasAlreadyFineIsUntouched:
    """A header pandas read itself is already named and already suffixed."""

    def test_a_valid_header_is_not_rescanned(self):
        frame = pd.DataFrame({"Loan Policy Number": ["ERE0001"],
                              "Current Balance": [120000.0]})
        out, row, failed = redetect_header(frame)
        assert list(out.columns) == ["Loan Policy Number", "Current Balance"]
        assert (row, failed) == (0, False)
        pd.testing.assert_frame_equal(out, frame)
