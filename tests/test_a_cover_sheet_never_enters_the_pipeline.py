"""The cover sheet is not a source, so it does not enter the pipeline.

`LoanExtract One` carries a seven-row `Summary` in front of a 568-row loan
book, and the cover sheet's columns are named exactly like the book's —
`Current Outstanding Balance`, `Original Loan Amount`. Every layer that met
both of them had to choose between two identical-looking candidates, and each
chose separately:

  * the inventory recorded whichever sheet came first (fixed in
    `file_classifier`, which now takes the widest sheet with the most rows);
  * target coverage tied on confidence and fell through to a tie-break on file
    and column name, selecting the SUMMARY as the authoritative source for
    `current_outstanding_balance` — and then asked the operator to confirm that
    for every future delivery of the portfolio (fixed in `_match_candidates`,
    which now ranks a small sheet below its larger sibling);
  * universe selection ranks overlap ahead of row count, so a cover sheet whose
    few keys all appear elsewhere could still outrank the book itself.

Three patches, three layers, and a fourth waiting to be found. So the sheet is
set aside ONCE, in the loader, before any of them sees it.

SET ASIDE, NOT HIDDEN: it is recorded in the file's coverage with the row
counts that decided it, because a sheet that silently disappears is its own
kind of defect.
"""

from __future__ import annotations

import pandas as pd
import pytest

from engine.onboarding_agent import source_table_loader as stl


def _workbook(tmp_path, name="LoanExtract One.xlsx", **sheets):
    """A workbook of {sheet_name: n_rows}, every sheet named like a loan book."""
    path = tmp_path / name
    with pd.ExcelWriter(path) as writer:
        for sheet, rows in sheets.items():
            pd.DataFrame({
                "Loan Policy Number": [f"ERE{i:04d}" for i in range(rows)],
                "Current Outstanding Balance": [100000.0 + i for i in range(rows)],
            }).to_excel(writer, sheet_name=sheet, index=False)
    return [{"file_path": str(path), "file_name": name, "file_type": "xlsx"}]


def _load(inventory):
    return stl.load_source_tables(inventory)


class TestTheLivePack:

    def test_the_cover_sheet_is_not_loaded(self, tmp_path):
        tables, _cov, _sc = _load(_workbook(tmp_path, Summary=7, LoanExtractOne=568))
        assert [t.sheet_name for t in tables] == ["LoanExtractOne"]

    def test_the_loan_book_is_loaded_whole(self, tmp_path):
        tables, _cov, _sc = _load(_workbook(tmp_path, Summary=7, LoanExtractOne=568))
        assert len(tables[0].df) == 568

    def test_the_operator_can_see_it_was_set_aside_and_why(self, tmp_path):
        _t, cov, sheets = _load(_workbook(tmp_path, Summary=7, LoanExtractOne=568))
        aside = [s for s in sheets if s.sheet_name == "Summary"]
        assert len(aside) == 1
        assert aside[0].parse_status == stl.SET_ASIDE
        assert aside[0].rows == 7
        assert "568" in aside[0].parse_error and "7 row" in aside[0].parse_error
        assert any("Summary" in s for s in cov[0].sheets_skipped)

    def test_the_file_still_counts_as_parsed(self, tmp_path):
        """Setting a sheet aside is not a parse failure."""
        _t, cov, _s = _load(_workbook(tmp_path, Summary=7, LoanExtractOne=568))
        assert cov[0].parse_status == stl.PARSED


class TestTheRuleItself:

    @pytest.mark.parametrize("rows,largest,expected", [
        (7, 568, True), (1, 100, True), (10, 20, True),
        (11, 20, False), (500, 480, False), (568, 568, False),
        (0, 0, False), (5, 0, False),
    ])
    def test_less_than_half_its_largest_sibling(self, rows, largest, expected):
        assert stl.is_summary_sheet(rows, largest) is expected


class TestNothingElseIsSetAside:

    def test_a_single_sheet_workbook_is_never_affected(self, tmp_path):
        """Even a tiny one — there is nothing for it to be a summary OF."""
        tables, _c, sheets = _load(_workbook(tmp_path, OnlySheet=3))
        assert [t.sheet_name for t in tables] == [""]        # not multi-sheet
        assert all(s.parse_status != stl.SET_ASIDE for s in sheets)

    def test_sheets_of_comparable_size_are_all_kept(self, tmp_path):
        tables, _c, _s = _load(_workbook(tmp_path, BookA=500, BookB=480))
        assert sorted(t.sheet_name for t in tables) == ["BookA", "BookB"]

    def test_two_small_sheets_under_one_big_one_both_go(self, tmp_path):
        tables, _c, _s = _load(_workbook(tmp_path, Summary=7, Notes=4, Book=568))
        assert [t.sheet_name for t in tables] == ["Book"]

    def test_an_empty_sheet_is_still_reported_as_empty(self, tmp_path):
        """The existing reason must not be replaced by the new one."""
        path = tmp_path / "w.xlsx"
        with pd.ExcelWriter(path) as writer:
            pd.DataFrame({"A": [1] * 200}).to_excel(writer, sheet_name="Book", index=False)
            pd.DataFrame({"A": []}).to_excel(writer, sheet_name="Blank", index=False)
        _t, _c, sheets = _load([{"file_path": str(path), "file_name": "w.xlsx",
                                 "file_type": "xlsx"}])
        blank = [s for s in sheets if s.sheet_name == "Blank"][0]
        assert blank.parse_status == stl.EMPTY

    def test_a_csv_is_untouched(self, tmp_path):
        path = tmp_path / "book.csv"
        pd.DataFrame({"Loan Policy Number": ["ERE0001"]}).to_csv(path, index=False)
        tables, _c, _s = _load([{"file_path": str(path), "file_name": "book.csv",
                                 "file_type": "csv"}])
        assert len(tables) == 1


class TestTheWholePipelineNeverSeesIt:
    """The point of doing this in the loader rather than in each consumer."""

    def test_no_loaded_frame_carries_the_cover_sheet_columns(self, tmp_path):
        tables, _c, _s = _load(_workbook(tmp_path, Summary=7, LoanExtractOne=568))
        # Every frame that reaches inventory, profiling, evidence, eligibility,
        # coverage and the tape comes from this list.
        assert all(len(t.df) == 568 for t in tables)
        assert "Summary" not in {t.sheet_name for t in tables}
