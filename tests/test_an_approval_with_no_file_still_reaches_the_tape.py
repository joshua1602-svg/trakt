"""Fifty-five approved mappings, and the tape received none of them.

`12_approved_mapping_overrides.yaml` as the live three-file delivery received
it — every entry, without exception::

    current_interest_rate       : (any file) :: Current Interest Rate  [operator_approved]
    current_outstanding_balance : (any file) :: Current Outstanding Balance
    current_valuation_amount    : (any file) :: Latest Valuation
    ...

and the function that turns overrides into tape sources::

    def add(file_name, column, canon, method, conf):
        if not canon or not file_name or not column:
            return                    # a blank source_file is DISCARDED

Neither half is wrong on its own. The Operations Control Centre leaves the file
blank deliberately — ``_sole_delivery_file`` returns "" for anything but a
single-file delivery, because "guessing which file an operator meant is exactly
the kind of silent decision this sprint exists to remove". And a tape source
does need a file: a column name alone does not say which of three workbooks to
read it from.

Between them, on any pack of more than one file — which is the normal case —
every operator approval was dropped before the tape saw one. Nothing reported
it. The coverage matrix simply kept re-deriving mappings the operator had
already confirmed, and the operator kept being asked to confirm them again.

THE PACK ITSELF SETTLES IT. An override with no file applies to every file that
actually carries that column: one file is no guess at all, and several is the
overlap question coverage already asks.
"""

from __future__ import annotations

import pytest

from engine.onboarding_agent.central_tape_builder import _collect_field_sources

BALANCE = "current_outstanding_balance"
COLUMN = "Current Outstanding Balance"

INVENTORY = {
    "LoanExtract One.xlsx": {"file_path": "/x/a.xlsx", "sheet_name": "LoanExtractOne"},
    "PropertyExtract.xlsx": {"file_path": "/x/b.xlsx", "sheet_name": ""},
    "Principal And Interest.xlsx": {"file_path": "/x/c.xlsx", "sheet_name": ""},
}


def _override(column=COLUMN, canonical=BALANCE, source_file=""):
    return {"source_file": source_file, "source_column": column,
            "canonical_field": canonical, "method": "operator_approved",
            "confidence": 1.0}


def _candidate(file_name, column=COLUMN, canonical=BALANCE, conf=0.9):
    return {"source_file": file_name, "source_column": column,
            "candidate_canonical_field": canonical, "confidence": conf}


def _sources(overrides, candidates):
    return _collect_field_sources(candidates, {"user_overrides": overrides},
                                  INVENTORY, set())


class TestTheLiveDelivery:

    def test_an_approval_with_no_file_reaches_the_tape(self):
        src = _sources([_override()], [_candidate("LoanExtract One.xlsx")])
        assert [s.file_name for s in src[BALANCE]] == ["LoanExtract One.xlsx"]

    def test_it_arrives_as_the_operator_s_own_decision(self):
        """Not as a re-derived candidate that happens to agree."""
        src = _sources([_override()], [_candidate("LoanExtract One.xlsx")])
        assert src[BALANCE][0].method == "operator_approved"
        assert src[BALANCE][0].confidence == 1.0

    def test_it_carries_the_file_s_path_and_sheet(self):
        src = _sources([_override()], [_candidate("LoanExtract One.xlsx")])
        assert src[BALANCE][0].file_path == "/x/a.xlsx"
        assert src[BALANCE][0].sheet == "LoanExtractOne"

    def test_several_approvals_across_several_files(self):
        src = _sources(
            [_override(), _override(column="Latest Valuation",
                                    canonical="current_valuation_amount")],
            [_candidate("LoanExtract One.xlsx"),
             _candidate("PropertyExtract.xlsx", column="Latest Valuation",
                        canonical="current_valuation_amount")])
        assert src[BALANCE][0].file_name == "LoanExtract One.xlsx"
        assert src["current_valuation_amount"][0].file_name == "PropertyExtract.xlsx"


class TestWhenSeveralFilesCarryTheColumn:
    """Not a guess to make — every one is offered, and the overlap question
    coverage already asks decides which is authoritative."""

    def test_every_file_holding_the_column_is_offered(self):
        src = _sources([_override()],
                       [_candidate("LoanExtract One.xlsx"),
                        _candidate("Principal And Interest.xlsx")])
        assert sorted(s.file_name for s in src[BALANCE]) == [
            "LoanExtract One.xlsx", "Principal And Interest.xlsx"]

    def test_all_of_them_arrive_as_approvals(self):
        src = _sources([_override()],
                       [_candidate("LoanExtract One.xlsx"),
                        _candidate("Principal And Interest.xlsx")])
        assert {s.method for s in src[BALANCE]} == {"operator_approved"}


class TestANamedFileIsStillObeyed:

    def test_an_override_naming_its_file_is_approved_for_that_file_only(self):
        """The other file may still appear as an ordinary candidate — what must
        not happen is the operator's DECISION being spread across both."""
        src = _sources([_override(source_file="PropertyExtract.xlsx")],
                       [_candidate("LoanExtract One.xlsx"),
                        _candidate("PropertyExtract.xlsx")])
        approved = [s for s in src[BALANCE] if s.method == "operator_approved"]
        assert [s.file_name for s in approved] == ["PropertyExtract.xlsx"]

    def test_a_named_file_is_never_swapped_for_another(self):
        """An approval from an earlier month may name a file this pack does not
        have. It must not quietly attach itself to whichever file does carry
        the column — that would apply a decision the operator never made."""
        src = _sources([_override(source_file="Last Month.xlsx")],
                       [_candidate("LoanExtract One.xlsx")])
        approved = [s for s in src.get(BALANCE, []) if s.method == "operator_approved"]
        assert [s.file_name for s in approved] == ["Last Month.xlsx"]


class TestNothingIsInvented:

    def test_a_column_no_file_carries_yields_no_source(self):
        src = _sources([_override(column="A Column Nobody Sent")],
                       [_candidate("LoanExtract One.xlsx")])
        approved = [s for ss in src.values() for s in ss
                    if s.method == "operator_approved"]
        assert approved == []

    def test_an_override_with_no_canonical_field_is_ignored(self):
        src = _sources([_override(canonical="")], [_candidate("LoanExtract One.xlsx")])
        assert all(s.method != "operator_approved"
                   for ss in src.values() for s in ss)

    @pytest.mark.parametrize("column", ["", "   "])
    def test_an_override_with_no_column_is_ignored(self, column):
        src = _sources([_override(column=column)], [_candidate("LoanExtract One.xlsx")])
        assert all(s.method != "operator_approved"
                   for ss in src.values() for s in ss)

    def test_the_column_is_matched_however_it_is_spelled(self):
        """The approval and the pack need not agree on spacing or case."""
        src = _sources([_override(column="current outstanding  balance")],
                       [_candidate("LoanExtract One.xlsx")])
        assert [s.file_name for s in src[BALANCE]] == ["LoanExtract One.xlsx"]
