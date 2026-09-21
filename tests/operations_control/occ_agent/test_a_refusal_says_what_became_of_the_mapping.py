"""A refusal about a mapped field has to say what happened to the mapping.

REPORTED FROM THE LIVE CASE, twice, after two deploys that each fixed a real
defect and neither of which was the whole answer:

    "current_principal_balance: CORE001 affects 0 record(s) (0.0%) —
     materiality BLOCKING"

    "I have already tagged [it] as being mapped to a field from the P&I file.
     Why isn't this mapping being picked up?"

Both screens were telling the truth. The mapping table said the column was
confirmed at 100%; the blocker said the field was absent. Whenever the
consolidation drops a file, both are true at once — and the reason was recorded
on the ONBOARD stage, which is not where anyone reading a validation refusal is
looking, and which does not reach ``run.blockers`` at all because that list is
built only from stages that HARD BLOCKED.

So the operator was handed an unanswerable line. There was nothing on it to act
on, no way to tell "the client never sent this" from "the file was dropped for
a reason with your name on it", and the only way forward was to guess or to ask
somebody to read the run document.

The consolidation knows precisely what happened. These tests hold it to saying
so on the line that stops the run, for every way a mapped field can fail to
arrive — and to staying quiet about a field nobody mapped, where there is
genuinely nothing to add.
"""

from __future__ import annotations

import pandas as pd

from operations_control.occ_agent.execution import (
    consolidate_pack,
    why_absent,
)

PRIMARY = "LoanExtract One - OMNI 2026_09_01.xlsx"
CASHFLOW = "Principal And Interest - OMNI 2026_09_01.xlsx"
FIELD = "current_principal_balance"


def _explain(cashflow, resolved_cashflow, primary=None, resolved_primary=None):
    """``(is_on_the_tape, what_the_blocker_would_say)``."""
    frames = {
        PRIMARY: primary if primary is not None else pd.DataFrame({
            "Loan Policy Number": ["76034101", "76034201"],
            "Reporting Date": ["2026-09-01", "2026-09-01"]}),
        CASHFLOW: cashflow,
    }
    resolved = {
        PRIMARY: resolved_primary if resolved_primary is not None else {
            "Loan Policy Number": "loan_identifier",
            "Reporting Date": "data_cut_off_date"},
        CASHFLOW: resolved_cashflow,
    }
    tape, report = consolidate_pack(frames, resolved, PRIMARY)
    return FIELD in tape.columns, why_absent(FIELD, resolved, report)


class TestEachWayAMappedFieldFailsToArrive:
    """Every one of these produced the same bare line, and they need different
    things done about them."""

    def test_the_file_has_no_mapped_loan_identifier(self):
        on_tape, said = _explain(
            pd.DataFrame({"Account Number": ["76034101", "76034201"],
                          "C/F Principal Balance": [100.0, 200.0]}),
            {"C/F Principal Balance": FIELD})
        assert on_tape is False
        assert CASHFLOW in said
        assert "no mapped loan identifier" in said

    def test_the_file_has_repeated_keys_and_no_reporting_date(self):
        on_tape, said = _explain(
            pd.DataFrame({"Account Number": ["76034101"] * 2 + ["76034201"] * 2,
                          "C/F Principal Balance": [90.0, 100.0, 180.0, 200.0],
                          "Period": ["2026-08-01", "2026-09-01"] * 2}),
            {"Account Number": "loan_identifier",
             "C/F Principal Balance": FIELD})
        assert on_tape is False
        assert "more than one row per loan" in said

    def test_the_two_files_do_not_agree_on_the_loan_identifier(self):
        """The one an operator can act on immediately, and the one a bare
        CORE001 hides most completely."""
        on_tape, said = _explain(
            pd.DataFrame({"Account Number": ["AAA", "BBB"],
                          "C/F Principal Balance": [100.0, 200.0]}),
            {"Account Number": "loan_identifier",
             "C/F Principal Balance": FIELD})
        assert on_tape is False
        assert "loan identifiers" in said
        assert "0%" in said

    def test_the_file_joined_and_every_value_was_blank(self):
        """THE SILENT ONE. The file reads as joined, the column is simply
        absent, and nothing anywhere said the values were empty."""
        on_tape, said = _explain(
            pd.DataFrame({"Account Number": ["76034101", "76034201"],
                          "C/F Principal Balance": [None, None]}),
            {"Account Number": "loan_identifier",
             "C/F Principal Balance": FIELD})
        assert on_tape is False
        assert "blank" in said

    def test_the_loan_tape_itself_maps_it(self):
        """Then the column is on the tape by construction, so a presence
        failure is emptiness rather than absence — a different thing to fix."""
        _on_tape, said = _explain(
            pd.DataFrame({"Account Number": ["76034101", "76034201"]}),
            {"Account Number": "loan_identifier"},
            primary=pd.DataFrame({
                "Loan Policy Number": ["76034101", "76034201"],
                "Balance": [None, None]}),
            resolved_primary={"Loan Policy Number": "loan_identifier",
                              "Balance": FIELD})
        assert "the loan tape itself" in said
        assert "the values are not" in said


class TestItStaysQuietWhenThereIsNothingToAdd:
    def test_a_field_nobody_mapped_reads_as_it_always_did(self):
        """"Nobody sent us this" is a complete explanation already. Adding a
        sentence to every finding would bury the ones that carry news."""
        _on_tape, said = _explain(
            pd.DataFrame({"Account Number": ["76034101", "76034201"]}),
            {"Account Number": "loan_identifier"})
        assert said == ""

    def test_an_empty_consolidation_is_not_an_error(self):
        """A one-file pack never consolidates, and a refusal about it must
        still be a refusal rather than a crash."""
        assert why_absent(FIELD, {}, {}) == ""
        assert why_absent(FIELD, {PRIMARY: {"x": FIELD}}, {}) != ""


class TestTheLoansASecondaryFileDoesNotCarry:
    """REPORTED FROM THE LIVE CASE, one step past the join finally working:

        current_principal_balance: CORE002 affects 1 record(s) (0.18%)
                                   — materiality BLOCKING

    CORE001 was gone, so the column was on the tape and the files had joined.
    What was left is subtler and reads like a data fault: one loan in 568 with
    no balance. It is not a fault. The loan extract is the spine and decides
    which loans exist; a loan it carries that the principal-and-interest
    extract does not gets a blank, and validation refuses the blank per row.

    That is a RECONCILIATION question for the lender — "this loan is in your
    loan extract and not in your balances extract" — and nothing on the
    blocker said so. An operator reading "1 record is missing a mandatory
    value" goes looking for a broken row; the row is not broken and does not
    exist.

    A join at 99.8% is still a join, so this must not block the join. It has to
    be SAID.
    """

    def _short_by_one(self, loans=568):
        ids = [f"{76034000 + i}01" for i in range(loans)]
        frames = {
            PRIMARY: pd.DataFrame({"Loan Policy Number": ids,
                                   "Rate": [4.5] * loans}),
            # every loan but the last
            CASHFLOW: pd.DataFrame({"Account Number": ids[:-1],
                                    "C/F Principal Balance": [100.0] * (loans - 1)}),
        }
        resolved = {
            PRIMARY: {"Loan Policy Number": "loan_identifier",
                      "Rate": "current_interest_rate"},
            CASHFLOW: {"Account Number": "loan_identifier",
                       "C/F Principal Balance": FIELD},
        }
        tape, report = consolidate_pack(frames, resolved, PRIMARY)
        return tape, why_absent(FIELD, resolved, report), ids

    def test_the_file_still_joins(self):
        """99.8% is a join. One loan short is not a reason to throw the other
        567 balances away."""
        tape, _said, _ids = self._short_by_one()
        assert FIELD in tape.columns
        assert int(tape[FIELD].notna().sum()) == 567

    def test_the_blank_is_exactly_the_loan_that_is_missing(self):
        tape, _said, ids = self._short_by_one()
        missing = tape.loc[tape[FIELD].isna(), "loan_identifier"].tolist()
        assert missing == [ids[-1]]

    def test_the_refusal_says_the_loan_has_no_row_in_that_file(self):
        """THE SENTENCE. "1 record is blank" sends an operator hunting for a
        broken row; "1 loan has no row in the P&I extract" is a question they
        can put to the lender."""
        _tape, said, _ids = self._short_by_one()
        assert "no row in" in said
        assert CASHFLOW in said

    def test_it_names_the_loan(self):
        """A count with no identifier attached cannot be looked up."""
        _tape, said, ids = self._short_by_one()
        assert ids[-1] in said

    def test_a_file_that_covers_every_loan_says_nothing(self):
        """The sentence is for news. A complete join is not news, and adding a
        line to every finding would bury the ones that carry some."""
        frames, resolved = _explain_pack_complete()
        tape, report = consolidate_pack(frames, resolved, PRIMARY)
        assert int(tape[FIELD].isna().sum()) == 0
        assert why_absent(FIELD, resolved, report) == ""


def _explain_pack_complete():
    ids = ["76034101", "76034201"]
    frames = {
        PRIMARY: pd.DataFrame({"Loan Policy Number": ids, "Rate": [4.5, 5.0]}),
        CASHFLOW: pd.DataFrame({"Account Number": ids,
                                "C/F Principal Balance": [100.0, 200.0]}),
    }
    resolved = {
        PRIMARY: {"Loan Policy Number": "loan_identifier",
                  "Rate": "current_interest_rate"},
        CASHFLOW: {"Account Number": "loan_identifier",
                   "C/F Principal Balance": FIELD},
    }
    return frames, resolved


class TestTheFieldItIsAbout:
    def test_it_speaks_only_about_the_field_in_the_finding(self):
        """A pack drops a file carrying six mapped columns; a finding about one
        of them must not recite the other five."""
        _on_tape, said = _explain(
            pd.DataFrame({"Account Number": ["AAA", "BBB"],
                          "C/F Principal Balance": [100.0, 200.0],
                          "Rate": [4.5, 5.0]}),
            {"Account Number": "loan_identifier",
             "C/F Principal Balance": FIELD,
             "Rate": "current_interest_rate"})
        assert "Rate" not in said
        assert "current_interest_rate" not in said
