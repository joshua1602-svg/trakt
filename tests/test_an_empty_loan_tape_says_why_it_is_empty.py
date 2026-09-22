"""A delivery halted with nothing on it an operator could act on.

    Trakt could not build the combined loan record from these files.
    The files may be missing a loan listing.

    "Files sent to blob but onboarding blocked and there is no way for me to
     understand the blocker or to unblock it."

That sentence is not a finding. It is a fallback, reached because three layers
each threw information away:

1.  ``build_central_tapes`` records exactly what happened — which file it chose
    as the loan listing, the key column and normalisation rule it used, how many
    raw rows it read, how many survived as identifiers, and every source it
    excluded with the reason and row count. All of it goes into
    ``universe_debug`` and was then dropped: the caller reported
    ``onboarding did not produce a central lender tape`` and nothing else.

    Its sibling three lines above, for the PIPELINE tape, says "no
    pipeline_report file was classified or no application/KFI key column was
    found (check the workbook's pipeline/KFI sheet)". One of the two can be
    acted on.

2.  ``_resolve_central_tape`` caught every exception and returned "no tape",
    so ``ExpectedBalanceCheckError`` — which names the tolerance it broke and
    the debug file showing the working — reached the operator as nothing.

3.  ``humanise_blocker`` replaced anything it did not recognise with
    "Something did not go as expected on our side", which is right for a
    traceback and would have destroyed the sentences 1 now produces.
"""

from __future__ import annotations

from engine.onboarding_agent.central_tape_builder import (
    explain_empty_lender_tape,
)
from operations_control.language import GENERIC_PROBLEM, humanise_blocker

LOAN = "LoanExtract One - OMNI 2026_09_01.xlsx"
CASH = "Principal And Interest - OMNI 2026_09_01.xlsx"


def _result(**debug):
    return {"lender_summary": {"universe_debug": debug}}


class TestTheBuildSaysWhyThereIsNoTape:

    def test_every_file_set_aside_for_the_period_names_file_and_period(self):
        said = explain_empty_lender_tape(_result(
            selected_universe_source_file="",
            run_reporting_period="2026-08", period_gate_active=True,
            excluded_sources=[{"source_file": LOAN, "reason": "period_mismatch",
                               "inferred_reporting_period": "2026-09",
                               "row_count": 568}]))
        first = said[0]
        assert LOAN in first
        assert "2026-09" in first and "2026-08" in first
        assert "568 row(s)" in first

    def test_no_loan_listing_says_what_a_loan_listing_is_for(self):
        said = explain_empty_lender_tape(_result(
            selected_universe_source_file="", excluded_sources=[]))
        assert "recognised as the loan listing" in said[0]
        assert "which loans exist" in said[0]

    def test_rows_read_but_no_identifier_names_the_column_and_rule(self):
        said = explain_empty_lender_tape(_result(
            selected_universe_source_file=LOAN,
            selected_universe_key_column="Loan Policy Number",
            selected_universe_normalisation_rule="strip_trailing_01",
            raw_universe_rows=568, canonical_universe_rows=0))
        assert "Loan Policy Number" in said[0]
        assert "strip_trailing_01" in said[0]
        assert "568 row(s)" in said[0]

    def test_a_file_that_was_mapped_and_then_not_used_is_named(self):
        said = " ".join(explain_empty_lender_tape(_result(
            selected_universe_source_file=LOAN,
            raw_universe_rows=568, canonical_universe_rows=0,
            excluded_sources=[{"source_file": CASH, "reason": "period_mismatch",
                               "row_count": 568}])))
        assert CASH in said

    def test_a_diagnosis_never_replaces_the_failure_it_diagnoses(self):
        """Malformed input returns sentences or none — never an exception.

        A diagnosis that raises would replace the failure it was called to
        explain, which is how the operator ended up with nothing in the first
        place.
        """
        for bad in ({}, {"lender_summary": None}, None,
                    {"lender_summary": {"universe_debug": "not a dict"}},
                    {"lender_summary": {"universe_debug":
                                        {"excluded_sources": "not a list"}}}):
            assert isinstance(explain_empty_lender_tape(bad), list)


class TestASentenceWrittenForAHumanSurvivesTranslation:

    def test_the_new_diagnosis_is_kept_word_for_word(self):
        said = (f"{LOAN} was set aside (period_mismatch): Trakt read it as "
                "covering 2026-09, and this delivery is for 2026-08. It holds "
                "568 row(s).")
        assert humanise_blocker(said, allow=(LOAN,)) == said

    def test_internal_jargon_is_still_translated(self):
        out = humanise_blocker("onboarding did not produce a central lender tape")
        assert out != GENERIC_PROBLEM
        assert "combined loan record" in out

    def test_a_traceback_is_still_replaced(self):
        assert humanise_blocker(
            'Traceback (most recent call last): File "/home/user/x.py", line 3'
        ) == GENERIC_PROBLEM

    def test_an_empty_blocker_is_replaced(self):
        assert humanise_blocker("") == GENERIC_PROBLEM
