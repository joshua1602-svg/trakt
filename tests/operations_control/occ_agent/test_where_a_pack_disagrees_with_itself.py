"""A delivery is several files, and they do not always agree.

In a real equity release pack the current interest rate arrives three times —
``Loan Interest Rate`` in the loan tape, ``Current Interest Rate`` in the
payments tape, ``Interest Rate`` in the property tape — and the origination
date, the redemption date, the outstanding balance and the pool identifier all
arrive twice.

When the files agree that is free reconciliation, and the central tape builder
records the field as validated. When they disagree it raises a BLOCKING
``value_conflict`` gap whose remedy is a source-precedence rule.

The defect was WHEN you found out. The rehearsal mapped every file and
compared nothing across them — duplicates were only ever detected inside the
one file the canonical tape is built from — so a pack whose files disagreed
looked clean in rehearsal and stopped after activation. That is the one place
a rehearsal exists to have looked first.

Two properties are load-bearing here and are tested directly:

* **the comparison must agree with the builder's**, or the rehearsal passes
  packs the builder then stops, which is worse than not looking;
* **it must not block**, because the remedy is a governed artefact this
  surface cannot write, and blocking an operator in front of a question they
  cannot answer from this screen is a trap this codebase has sprung before.
"""

from __future__ import annotations

import pandas as pd
import pytest

from operations_control.occ_agent import cross_file as cf

LOAN = "LoanExtract One - OMNI 2026_09_01.xlsx"
CASH = "Principal And Interest - OMNI 2026_09_01.xlsx"
PROP = "PropertyExtract - Omni 2026_09_01.xlsx"


def row(file_name: str, column: str, canonical: str, note: str = ""):
    return {"source_file": file_name, "source_column": column,
            "canonical_field": canonical, "note": note}


REPORT = [
    row(LOAN, "Loan Policy Number", "loan_identifier"),
    row(LOAN, "Loan Interest Rate", "current_interest_rate"),
    row(LOAN, "Current Outstanding Balance", "current_outstanding_balance"),
    row(LOAN, "Broker", "broker_channel"),
    row(CASH, "Account Number", "loan_identifier"),
    row(CASH, "Current Interest Rate", "current_interest_rate"),
    row(PROP, "Loan ID", "loan_identifier"),
    row(PROP, "Interest Rate", "current_interest_rate"),
    row(PROP, "Total OSBalance", "current_outstanding_balance"),
    row(PROP, "Acerage", "", "below the confidence threshold"),
]


def frames(prop_rates=(4.85, 5.10, 3.99)):
    return {
        LOAN: pd.DataFrame({
            "Loan Policy Number": ["L1", "L2", "L3"],
            "Loan Interest Rate": [4.85, 5.10, 3.99],
            "Current Outstanding Balance": [100000, 200000, 300000],
            "Broker": ["A", "B", "C"]}),
        CASH: pd.DataFrame({
            "Account Number": ["L1", "L2", "L3"],
            "Current Interest Rate": [4.85, 5.10, 3.99]}),
        PROP: pd.DataFrame({
            "Loan ID": ["L1", "L2", "L3"],
            "Interest Rate": list(prop_rates),
            "Total OSBalance": [100000, 200000, 300000],
            "Acerage": [0.1, 0.2, 0.3]}),
    }


def by_field(comparisons):
    return {c.canonical_field: c for c in comparisons}


# --------------------------------------------------------------------------- #
# Which fields are even in question
# --------------------------------------------------------------------------- #

class TestWhichFieldsAreShared:

    def test_a_field_several_files_carry_is_picked_up(self):
        shared = cf.shared_fields(REPORT)
        assert "current_interest_rate" in shared
        assert len(shared["current_interest_rate"]) == 3

    def test_a_field_only_one_file_carries_is_not(self):
        """Nothing to compare it against."""
        assert "broker_channel" not in cf.shared_fields(REPORT)

    def test_a_column_below_the_threshold_is_never_compared(self):
        """It is not yet CLAIMED to be the field, so comparing it would
        manufacture a disagreement out of the mapper's own uncertainty."""
        assert "" not in cf.claimed_by_file(REPORT)
        for sources in cf.shared_fields(REPORT).values():
            assert ("PropertyExtract - Omni 2026_09_01.xlsx",
                    "Acerage") not in sources


# --------------------------------------------------------------------------- #
# Whether they agree
# --------------------------------------------------------------------------- #

class TestWhetherTheFilesAgree:

    def test_files_that_agree_are_reported_as_agreeing(self):
        found = by_field(cf.compare(REPORT, frames()))
        rate = found["current_interest_rate"]
        assert not rate.conflicted
        assert (rate.agreed, rate.differed) == (3, 0)
        assert "agree on all 3 loans" in rate.sentence()

    def test_one_disagreeing_loan_is_a_conflict(self):
        """The real risk: three files, one of them out of step on one loan."""
        found = by_field(cf.compare(REPORT, frames(prop_rates=(4.85, 5.90,
                                                               3.99))))
        rate = found["current_interest_rate"]
        assert rate.conflicted
        assert (rate.agreed, rate.differed) == (2, 1)

    def test_the_disagreeing_loan_is_named_with_every_value(self):
        found = by_field(cf.compare(REPORT, frames(prop_rates=(4.85, 5.90,
                                                               3.99))))
        loan, values = found["current_interest_rate"].examples[0]
        assert loan == "L2"
        assert {v for _f, _c, v in values} == {"5.1", "5.9"}
        assert {f for f, _c, _v in values} == {LOAN, CASH, PROP}

    def test_examples_are_bounded(self):
        """Enough to recognise the shape of the problem, not a data dump."""
        found = by_field(cf.compare(REPORT, frames(prop_rates=(9.9, 9.9,
                                                               9.9))))
        rate = found["current_interest_rate"]
        assert rate.differed == 3
        assert len(rate.examples) == cf.EXAMPLE_LIMIT

    def test_a_field_carried_by_two_files_that_agree_says_so(self):
        found = by_field(cf.compare(REPORT, frames()))
        assert found["current_outstanding_balance"].agreed == 3


# --------------------------------------------------------------------------- #
# The comparison must be the platform's own
# --------------------------------------------------------------------------- #

class TestItJudgesEqualityTheSameWayTheBuilderDoes:
    """If the rehearsal called two values equal and the builder later called
    them different, the rehearsal would pass packs the builder then stops."""

    def test_the_comparison_matches_the_builders(self):
        from engine.onboarding_agent.central_tape_builder import (
            _values_match as builders,
        )
        cases = [(4.85, 4.85), (4.85, 4.8500001), (4.85, 5.90), (100, 100.5),
                 ("A", "A"), ("A", " A "), ("A", "B"), ("", "A"),
                 (None, "A"), ("4.85", 4.85), (0, 0.0)]
        for a, b in cases:
            assert cf._values_match(a, b) == builders(a, b), (a, b)

    def test_a_blank_never_matches(self):
        """A file that simply does not carry a loan is not a disagreement."""
        assert not cf._values_match("", "4.85")
        assert not cf._values_match(None, None)

    def test_a_loan_only_one_file_carries_is_not_a_disagreement(self):
        f = frames()
        f[PROP] = f[PROP].iloc[:2]      # the property file is missing L3
        found = by_field(cf.compare(REPORT, f))
        rate = found["current_interest_rate"]
        assert rate.differed == 0
        assert rate.agreed == 3, "L3 is still checked across the other two"


# --------------------------------------------------------------------------- #
# What it does when it cannot compare
# --------------------------------------------------------------------------- #

class TestWhenItCannotCompare:

    def test_a_source_with_no_loan_identifier_is_left_out_and_said_so(self):
        """Two of the three files can still be checked against each other.
        What must not happen is the report reading as though all three were:
        "these three agree" when only two were compared is exactly the quiet
        overstatement this module exists to prevent."""
        report = [r for r in REPORT
                  if not (r["source_file"] == PROP
                          and r["canonical_field"] == "loan_identifier")]
        rate = by_field(cf.compare(report, frames()))["current_interest_rate"]

        assert rate.agreed == 3, "the other two files were still compared"
        assert rate.uncompared_sources == [(PROP, "Interest Rate")]
        assert "could not be checked" in rate.sentence()
        assert not rate.conflicted, "an unchecked source is not a disagreement"

    def test_no_loan_identifier_anywhere_means_it_says_so_rather_than_guessing(
            self):
        report = [r for r in REPORT
                  if r["canonical_field"] != "loan_identifier"]
        rate = by_field(cf.compare(report, frames()))["current_interest_rate"]
        assert rate.not_compared
        assert "loan identifier" in rate.sentence()
        assert not rate.conflicted, "an uncompared field is not a conflict"

    def test_an_unreadable_file_is_left_out(self):
        f = frames()
        f[PROP] = None
        found = by_field(cf.compare(REPORT, f))
        assert found["current_interest_rate"].agreed == 3


# --------------------------------------------------------------------------- #
# What reaches the operator
# --------------------------------------------------------------------------- #

class TestWhatIsReported:

    def test_only_conflicts_become_findings(self):
        agreeing = cf.findings(cf.compare(REPORT, frames()))
        assert agreeing == []

    def test_a_conflict_names_the_field_and_the_remedy(self):
        found = cf.findings(cf.compare(REPORT, frames(prop_rates=(4.85, 5.90,
                                                                  3.99))))
        assert len(found) == 1
        assert found[0]["canonical_field"] == "current_interest_rate"
        assert "believe" in found[0]["remedy"]

    def test_conflicts_are_listed_before_the_rest(self):
        """An operator reading this wants what will stop the delivery, not a
        census of everything that happens to appear twice."""
        rows = cf.compare(REPORT, frames(prop_rates=(4.85, 5.90, 3.99)))
        assert rows[0].conflicted
        assert not rows[-1].conflicted

    def test_a_finding_is_not_a_decision(self):
        """The remedy is a source-precedence rule, which this surface cannot
        write. Presenting it as an answerable question would be asking an
        operator for something they cannot give here — and that trap has been
        sprung in this codebase before."""
        found = cf.findings(cf.compare(REPORT, frames(prop_rates=(4.85, 5.90,
                                                                  3.99))))
        assert "decision_id" not in found[0]
        assert "available_actions" not in found[0]
        assert found[0]["severity"] == "conflict"
