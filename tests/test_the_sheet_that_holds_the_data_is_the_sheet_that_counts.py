"""A loan extract that opens on a cover sheet was read as the cover sheet.

The delivery's diagnosis, from the live pack::

    LoanExtract One - OMNI 2026_09_01.xlsx
        sheet        : Summary
        rows x cols  : 7 x 5

next to siblings of 567 and 569 rows. The loan book is on another sheet of the
same workbook, and ``source_table_loader`` has always read every sheet::

    frames = [(sh, xl.parse(sh)) for sh in xl.sheet_names]

so period eligibility DID find it, dated it 2026-08 from its own ``Month Run``
column at 0.95 confidence, and named it the loan listing. Two rows for one file
in 04c say so. But the inventory recorded ``sheet_names[0]``, the central tape
sources from the inventory, and so the build opened the cover sheet:

    FILES SET ASIDE:
        LoanExtract One - OMNI 2026_09_01.xlsx: future_period (…, 4 rows)

Four rows. One workbook, two readings, and the layer that DECIDED was right
while the layer that ACTED was wrong.

WHAT MAKES IT WORSE THAN AN EMPTY TAPE. With the period gate answering
correctly the cover sheet becomes ELIGIBLE, and a seven-row cover sheet is then
a perfectly serviceable loan listing: a delivery that used to halt would
instead publish a month's management information over four loans. A wrong
number reported confidently is worse than no number, so this is not a tidy-up.
"""

from __future__ import annotations

import pandas as pd
import pytest

from engine.onboarding_agent.file_classifier import classify_file

#: The shape of the real pack: a cover sheet in front of the loan book.
COVER = pd.DataFrame({"Report": ["ERE Funded Book"], "Run": ["2026_09_01"],
                      "Prepared by": ["OMNI"], "Rows": [568], "Note": ["-"]})
LOANS = pd.DataFrame({
    "Loan Policy Number": [f"760341{i:02d}" for i in range(20)],
    "Month Run": ["2026-08"] * 20,
    "Current Balance": [100000.0 + i for i in range(20)],
    "Current Valuation": [400000.0] * 20,
})


def _book(tmp_path, sheets, name="LoanExtract One - OMNI 2026_09_01.xlsx"):
    p = tmp_path / name
    with pd.ExcelWriter(p) as w:
        for sheet, df in sheets:
            df.to_excel(w, sheet_name=sheet, index=False)
    return p


class TestTheDataSheetIsTheOneReported:

    def test_a_cover_sheet_in_front_does_not_become_the_file(self, tmp_path):
        item = classify_file(_book(tmp_path, [("Summary", COVER),
                                              ("Loan Book", LOANS)]))
        assert item.sheet_name == "Loan Book"
        assert item.row_count == 20

    def test_the_sheet_named_is_the_one_holding_the_loan_key(self, tmp_path):
        """What the tape build takes from the inventory is the SHEET, and it
        then reads that sheet for a loan identifier. The cover sheet has none.

        Asserted by reading the named sheet back, because the inventory records
        a sheet name and a column COUNT but not the column names — so nothing
        on the item itself can show which columns came with the choice.
        """
        p = _book(tmp_path, [("Summary", COVER), ("Loan Book", LOANS)])
        item = classify_file(p)
        cols = [str(c) for c in pd.read_excel(p, sheet_name=item.sheet_name)]
        assert "Loan Policy Number" in cols

    def test_the_period_column_the_gate_reads_comes_with_it(self, tmp_path):
        """04c dated the loan sheet 2026-08 from `Month Run` at 0.95 while the
        cover sheet fell back to the filename and was set aside. The inventory
        now points at the sheet that carries the confident answer."""
        p = _book(tmp_path, [("Summary", COVER), ("Loan Book", LOANS)])
        item = classify_file(p)
        cols = [str(c) for c in pd.read_excel(p, sheet_name=item.sheet_name)]
        assert "Month Run" in cols

    def test_order_does_not_matter(self, tmp_path):
        item = classify_file(_book(tmp_path, [("Loan Book", LOANS),
                                              ("Summary", COVER)]))
        assert item.sheet_name == "Loan Book"

    def test_the_widest_of_two_equal_length_sheets_wins(self, tmp_path):
        narrow = LOANS[["Loan Policy Number"]]
        item = classify_file(_book(tmp_path, [("Narrow", narrow),
                                              ("Wide", LOANS)]))
        assert item.sheet_name == "Wide"


class TestNothingElseChanges:

    def test_a_single_sheet_workbook_is_read_exactly_as_before(self, tmp_path):
        item = classify_file(_book(tmp_path, [("Sheet1", LOANS)]))
        assert item.sheet_name == "Sheet1"
        assert item.row_count == 20

    def test_a_csv_has_no_sheet_and_is_untouched(self, tmp_path):
        p = tmp_path / "loans.csv"
        LOANS.to_csv(p, index=False)
        item = classify_file(p)
        assert item.sheet_name == ""
        assert item.row_count == 20

    def test_a_workbook_of_empty_sheets_reports_nothing_rather_than_guessing(
            self, tmp_path):
        item = classify_file(_book(tmp_path, [("A", pd.DataFrame()),
                                              ("B", pd.DataFrame())]))
        assert item.classification == "unknown"
        assert item.row_count is None

    def test_a_sheet_that_will_not_parse_does_not_lose_the_others(
            self, tmp_path, monkeypatch):
        p = _book(tmp_path, [("Summary", COVER), ("Loan Book", LOANS)])
        real = pd.ExcelFile.parse

        def _parse(self, name, *a, **k):
            if name == "Summary":
                raise ValueError("cannot read this sheet")
            return real(self, name, *a, **k)

        monkeypatch.setattr(pd.ExcelFile, "parse", _parse)
        assert classify_file(p).sheet_name == "Loan Book"


class TestTheFileStillClassifiesAsALoanReport:
    """Choosing the sheet must not change WHAT the file is taken to be — the
    role drives the period gate and the universe selection."""

    def test_a_loan_extract_is_still_a_current_loan_report(self, tmp_path):
        item = classify_file(_book(tmp_path, [("Summary", COVER),
                                              ("Loan Book", LOANS)]))
        assert item.classification == "current_loan_report"

    @pytest.mark.parametrize("name,expected", [
        ("PropertyExtract - Omni 2026_09_01.xlsx", "collateral_report"),
    ])
    def test_a_property_extract_is_still_collateral(self, tmp_path, name,
                                                    expected):
        item = classify_file(_book(tmp_path, [("Summary", COVER),
                                              ("PG_PropertyExtract", LOANS)],
                                   name=name))
        assert item.classification == expected
