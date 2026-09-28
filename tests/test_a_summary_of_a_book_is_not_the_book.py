"""The cover sheet was about to become the authoritative source for balances.

From the live delivery's coverage matrix, with the operator being asked to
confirm each one for every future delivery of the portfolio::

    current_outstanding_balance   sheet='Summary'   col=Current Outstanding Balance
    original_principal_balance    sheet='Summary'   col=Original Loan Amount

`LoanExtract One` opens on a seven-row `Summary` in front of a 568-row loan
book, and the cover sheet's columns carry the SAME NAMES as the book's. Both
reach `_match_candidates` at the same confidence, and the winner fell out of
the remaining tie-break::

    cands.sort(key=lambda c: (-c["confidence"], c["source_file"],
                              c["source_column"]))

— same file, same column name, so the order was whatever the evidence rows
happened to be in. For `current_outstanding_balance`, the most important field
in an equity release pack, it picked the summary.

WHAT MADE IT DANGEROUS WAS THE QUESTION, NOT THE GUESS. The queue offered
"Accept the suggested match" at portfolio scope. One click would have recorded
a governed rule sourcing every future month's outstanding balance from a cover
sheet — and the run would have completed, published, and looked right.

Fixing the inventory's sheet was not enough: `source_table_loader` reads EVERY
sheet, so the cover sheet's columns stayed in the candidate pool. Coverage is
now told how many rows each sheet holds, and within one file a sheet less than
half the size of that file's largest is a summary of it and ranks below it. It
stays visible as an alternative — only an operator can say that a lender really
does report a field on a small sheet.
"""

from __future__ import annotations

import pandas as pd
import pytest

from engine.onboarding_agent import column_evidence as ce
from engine.onboarding_agent.target_coverage import _match_candidates

FIELD = {"target_field": "current_outstanding_balance",
         "match_field": "current_outstanding_balance",
         "target_label": "Current Outstanding Balance", "target_domain": "loan",
         "required_status": "required", "applicability_status": "applicable"}

COL = "Current Outstanding Balance"


def _evidence(*sheets):
    """(sheet_name, n_rows) pairs from one workbook, as evidence rows."""
    rows = []
    for name, n in sheets:
        df = pd.DataFrame({COL: [100000.0 + i for i in range(n)]})
        rows += ce.build_column_evidence(df, source_file="LoanExtract One.xlsx",
                                         sheet_name=name)
    return rows


class TestTheLiveSelection:

    def test_the_loan_book_outranks_the_cover_sheet(self):
        cands = _match_candidates(FIELD, _evidence(("Summary", 7),
                                                   ("LoanExtractOne", 568)), {})
        assert cands[0]["source_sheet"] == "LoanExtractOne"

    def test_the_cover_sheet_is_still_offered_as_an_alternative(self):
        """Demoted, not deleted: only an operator can rule it out."""
        cands = _match_candidates(FIELD, _evidence(("Summary", 7),
                                                   ("LoanExtractOne", 568)), {})
        assert [c["source_sheet"] for c in cands] == ["LoanExtractOne", "Summary"]

    def test_the_reason_is_recorded_on_the_candidate(self):
        cands = _match_candidates(FIELD, _evidence(("Summary", 7),
                                                   ("LoanExtractOne", 568)), {})
        by_sheet = {c["source_sheet"]: c for c in cands}
        assert by_sheet["Summary"]["summary_of_a_larger_sheet"] is True
        assert by_sheet["LoanExtractOne"]["summary_of_a_larger_sheet"] is False

    def test_the_row_counts_reach_coverage_at_all(self):
        """The fact the ranking needs, which it did not previously have."""
        ev = _evidence(("Summary", 7), ("LoanExtractOne", 568))
        assert sorted(r["source_row_count"] for r in ev) == [7, 568]


class TestAWorkbookOfRealSheetsIsNotThisCase:

    @pytest.mark.parametrize("a,b", [(500, 480), (568, 567), (100, 51), (9, 9)])
    def test_sheets_of_comparable_size_are_both_kept_in_the_running(self, a, b):
        cands = _match_candidates(FIELD, _evidence(("BookA", a), ("BookB", b)), {})
        assert not any(c["summary_of_a_larger_sheet"] for c in cands)

    @pytest.mark.parametrize("small,large", [(7, 568), (1, 100), (10, 20)])
    def test_a_sheet_less_than_half_the_largest_is_a_summary(self, small, large):
        cands = _match_candidates(FIELD, _evidence(("Small", small),
                                                   ("Large", large)), {})
        assert cands[0]["source_sheet"] == "Large"

    def test_a_single_sheet_file_is_untouched(self):
        cands = _match_candidates(FIELD, _evidence(("OnlySheet", 568)), {})
        assert len(cands) == 1
        assert cands[0]["summary_of_a_larger_sheet"] is False


class TestTheRuleIsPerFileNotAcrossFiles:
    """A small FILE is a different question — that is the operator's to answer,
    and two files disagreeing about a field is a real conflict, not a layout
    accident inside one workbook."""

    def test_a_small_file_is_not_demoted_by_a_large_one(self):
        big = pd.DataFrame({COL: [1.0] * 568})
        small = pd.DataFrame({COL: [2.0] * 7})
        ev = (ce.build_column_evidence(big, source_file="LoanExtract.xlsx",
                                       sheet_name="LoanExtractOne")
              + ce.build_column_evidence(small, source_file="Servicer Note.xlsx",
                                         sheet_name=""))
        cands = _match_candidates(FIELD, ev, {})
        assert not any(c["summary_of_a_larger_sheet"] for c in cands)
        assert {c["source_file"] for c in cands} == {"LoanExtract.xlsx",
                                                     "Servicer Note.xlsx"}


class TestConfidenceStillDecidesAmongPeers:
    """The guardrail separates a summary from a book. It must not otherwise
    take over the ranking."""

    def test_the_better_match_still_wins_between_comparable_sheets(self):
        exact = pd.DataFrame({COL: [1.0] * 500})
        vague = pd.DataFrame({"Balance Amt": [2.0] * 500})
        ev = (ce.build_column_evidence(exact, source_file="f.xlsx", sheet_name="A")
              + ce.build_column_evidence(vague, source_file="f.xlsx", sheet_name="B"))
        cands = _match_candidates(FIELD, ev, {})
        assert cands[0]["source_column"] == COL
