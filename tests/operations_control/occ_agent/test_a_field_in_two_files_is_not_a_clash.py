"""Two files carrying one field is the shape of a delivery, not a problem.

REPORTED FROM THE LIVE CASE, on a row the operator had just got RIGHT. The
principal-and-interest extract had been dropped for having no mapped loan
identifier, so they mapped its ``Account Number`` to ``loan_identifier`` — the
one action that makes a multi-file pack assemble at all, because a file with no
identifier cannot be joined and contributes nothing. The row came back in
orange:

    Account Number   Ready to confirm   2 columns claim this   You said: loan identifier

    "This does not make any sense. I mapped to Loan_identifier."

They were right, and the server had always known it. ``_mark_contested`` records
``same_file`` on every claimant for exactly this reason — "the two read
differently to an operator and must not be conflated" — and both cases were then
rendered in one orange chip, with the distinction left in a tooltip nobody
hovers.

So the screen contradicted the engine. Two columns of ONE file claiming one
field is an ambiguity nothing can resolve and the run blocks on it. The SAME
field in two files is ordinary, is reconciled by the join, and for the loan
identifier is mandatory: every extract needs one. Told they had a clash, an
operator's reasonable move is to undo the mapping — which is precisely what
leaves the balance off the tape.

These tests hold the two apart where the screen reads them: on the row, and in
the count the filter is built from.
"""

from __future__ import annotations

from operations_control.occ_agent.mapping_view import _mark_contested

LOAN = "LoanExtract One - OMNI 2026_09_01.xlsx"
CASHFLOW = "Principal And Interest - OMNI 2026_09_01.xlsx"
PROPERTY = "PropertyExtract - Omni 2026_09_01.xlsx"


def _row(source_file, source_column, canonical_field, *, state="proposed",
         staged_action="", staged_field=""):
    return {"source_file": source_file, "source_column": source_column,
            "canonical_field": canonical_field, "state": state,
            "staged_action": staged_action, "staged_field": staged_field}


def _ere_pack():
    """Every file naming its own loan identifier, which is what the join needs,
    plus one field only the cashflow extract carries."""
    rows = [
        _row(LOAN, "Loan Policy Number", "loan_identifier"),
        _row(CASHFLOW, "Account Number", "loan_identifier",
             state="staged", staged_action="confirm",
             staged_field="loan_identifier"),
        _row(PROPERTY, "LoanID", "loan_identifier"),
        _row(CASHFLOW, "C/F Principal Balance", "current_principal_balance"),
    ]
    _mark_contested(rows)
    return {(r["source_file"], r["source_column"]): r for r in rows}


class TestTheSameFieldInSeveralFiles:
    def test_the_loan_identifier_is_claimed_by_every_file(self):
        """Not a defect in the fixture — the join is built on it."""
        pack = _ere_pack()
        claimed = pack[(CASHFLOW, "Account Number")]["also_claimed_by"]
        assert {c["source_column"] for c in claimed} == {"Loan Policy Number",
                                                         "LoanID"}

    def test_none_of_those_claims_is_in_the_same_file(self):
        """THE DISTINCTION THE SCREEN THREW AWAY. Every claimant here is
        another FILE, so nothing on this row is a question."""
        pack = _ere_pack()
        for column in ("Account Number", "Loan Policy Number", "LoanID"):
            file_name = {"Account Number": CASHFLOW,
                         "Loan Policy Number": LOAN,
                         "LoanID": PROPERTY}[column]
            claimed = pack[(file_name, column)]["also_claimed_by"]
            assert claimed, f"{column} should know about the others"
            assert not any(c["same_file"] for c in claimed), \
                f"{column} must not read as an ambiguity"

    def test_a_field_only_one_file_carries_is_not_marked(self):
        pack = _ere_pack()
        assert pack[(CASHFLOW, "C/F Principal Balance")]["also_claimed_by"] \
            == []

    def test_the_operator_s_staged_answer_is_what_counts(self):
        """The row that started this was STAGED, not yet committed. Reading the
        report instead of the draft, the clash an operator had just resolved
        would go on being reported — and the one they had just created would
        not be."""
        pack = _ere_pack()
        staged = pack[(CASHFLOW, "Account Number")]
        assert staged["staged_field"] == "loan_identifier"
        assert len(staged["also_claimed_by"]) == 2


class TestTwoColumnsOfOneFile:
    """The case that IS a question, which must keep reading as one."""

    def _ambiguous(self):
        rows = [
            _row(LOAN, "Loan Policy Number", "loan_identifier"),
            _row(LOAN, "Current Balance", "current_principal_balance"),
            _row(LOAN, "Principal Balance", "current_principal_balance"),
        ]
        _mark_contested(rows)
        return {(r["source_file"], r["source_column"]): r for r in rows}

    def test_it_is_marked_as_being_in_the_same_file(self):
        pack = self._ambiguous()
        claimed = pack[(LOAN, "Current Balance")]["also_claimed_by"]
        assert len(claimed) == 1
        assert claimed[0]["same_file"] is True
        assert claimed[0]["source_column"] == "Principal Balance"

    def test_both_sides_of_it_are_marked(self):
        """An operator has to be able to pick either one, so both rows carry
        the question."""
        pack = self._ambiguous()
        for column in ("Current Balance", "Principal Balance"):
            assert any(c["same_file"]
                       for c in pack[(LOAN, column)]["also_claimed_by"])


class TestAPackThatHasBoth:
    """A real pack has loan identifiers in every file AND may have one file
    that names a field twice. The counts must not roll them together: the
    filter an operator reaches for is "show me what I have to decide"."""

    def _mixed(self):
        rows = [
            _row(LOAN, "Loan Policy Number", "loan_identifier"),
            _row(CASHFLOW, "Account Number", "loan_identifier"),
            _row(PROPERTY, "LoanID", "loan_identifier"),
            _row(LOAN, "Current Balance", "current_principal_balance"),
            _row(LOAN, "Principal Balance", "current_principal_balance"),
        ]
        _mark_contested(rows)
        return rows

    def test_every_claim_is_counted_as_a_claim(self):
        rows = self._mixed()
        assert sum(1 for r in rows if r["also_claimed_by"]) == 5

    def test_only_the_in_file_ones_count_as_a_question(self):
        """Three of those five are loan identifiers doing their job. Counting
        them as questions puts a 5 in front of an operator who has two."""
        rows = self._mixed()
        ambiguous = sum(1 for r in rows
                        if any(c["same_file"] for c in r["also_claimed_by"]))
        assert ambiguous == 2
