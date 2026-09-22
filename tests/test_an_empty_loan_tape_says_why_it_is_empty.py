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


class TestNothingChosenAndNothingSetAsideIsThreeDifferentSituations:
    """The first fix replaced one guess with another.

    "No file in this pack was recognised as the loan listing" was asserted
    whenever nothing was chosen and nothing was excluded — and ERE's delivery
    reached exactly that branch. But that state has three causes, and the
    sentence named one of them: no file was OPENED at all (nothing in the pack
    was reached by the approved mapping); files were opened and none was read
    as a loan listing; or one was and it produced no identifier. An operator
    told the second when it is the first goes looking at file classification
    while the mapping is what is wrong.
    """

    def test_no_file_opened_says_so_and_names_what_was_delivered(self):
        said = " ".join(explain_empty_lender_tape(_result(
            selected_universe_source_file="", excluded_sources=[],
            considered_sources=[],
            pack_files=[{"source_file": LOAN, "artefact_role": "current_loan_report"},
                        {"source_file": CASH, "artefact_role": "cashflow_report"}])))
        assert "No file in this delivery was opened" in said
        assert "approved mapping" in said
        assert LOAN in said and CASH in said

    def test_files_opened_but_none_qualifying_names_each_and_its_role(self):
        said = " ".join(explain_empty_lender_tape(_result(
            selected_universe_source_file="", excluded_sources=[],
            universe_roles=["current_loan_report", "funded_book"],
            considered_sources=[
                {"source_file": LOAN, "artefact_role": "collateral_report",
                 "key_column": "Loan Policy Number", "key_count": 568,
                 "file_in_inventory": True, "frame_loaded": True},
                {"source_file": CASH, "artefact_role": "cashflow_report",
                 "key_column": "Loan Policy Number", "key_count": 568,
                 "file_in_inventory": True, "frame_loaded": True}])))
        assert LOAN in said and "collateral_report" in said
        assert CASH in said and "cashflow_report" in said
        # And what WOULD have qualified, so the operator can act.
        assert "current_loan_report" in said and "funded_book" in said
        assert "which loans exist" in said

    def test_a_file_opened_with_no_key_column_says_that_rather_than_a_count(self):
        said = " ".join(explain_empty_lender_tape(_result(
            selected_universe_source_file="", excluded_sources=[],
            considered_sources=[{"source_file": LOAN, "artefact_role": "",
                                 "key_column": "", "key_count": 0,
                                 "file_in_inventory": True,
                                 "frame_loaded": True}])))
        assert "no loan identifier column was found" in said
        assert "not recognised as any known kind of file" in said

    def test_a_file_the_mapping_names_but_the_delivery_lacks_says_that(self):
        """The mapping was learned from a sample. If it names that sample's
        file and the delivery carries another name, "opened and found empty"
        and "never there" read identically — and only one of them is fixed by
        looking at the file."""
        said = " ".join(explain_empty_lender_tape(_result(
            selected_universe_source_file="", excluded_sources=[],
            considered_sources=[{"source_file": "LoanExtract One - OMNI.xlsx",
                                 "artefact_role": "current_loan_report",
                                 "key_column": "", "key_count": 0,
                                 "file_in_inventory": False,
                                 "frame_loaded": False}])))
        assert "named by the approved mapping" in said
        assert "not found in this delivery" in said
        # …and it does not then also claim the file failed to qualify.
        assert "None of them is the loan listing" not in said

    def test_it_never_claims_a_cause_it_was_given_no_evidence_for(self):
        """An older build writes no ``considered_sources``. Absent evidence the
        wording must not assert that files were opened and rejected."""
        said = " ".join(explain_empty_lender_tape(_result(
            selected_universe_source_file="", excluded_sources=[])))
        assert "None of them is the loan listing" not in said


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
