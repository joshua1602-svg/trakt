"""Diagnosing a halted delivery cost a deployment and a screenshot.

Everything that explains a halt — how each file was classified, what period it
was read as, which file became the loan listing, under which key column — is
written into the run's project directory, which lives on scratch and goes with
it. So the only way to see any of it was to change the code that reports it,
deploy, press the button and read the next screen: twenty minutes for one bit.

``scripts/diagnose_onboarding_pack`` runs the same two calls the live adapter
makes and prints what they recorded. These cover the part of it that could
lie — the reading of the artefacts — because a diagnostic that misreports is
worse than none: it is the third guess dressed as a finding.

IT REPORTS STRUCTURE, NOT DATA, so its output can be sent by somebody holding
a pack to somebody who may not see it. That is the property that makes it
usable at all here, so it is asserted rather than assumed.
"""

from __future__ import annotations

import json

from scripts.diagnose_onboarding_pack import (_eligibility, _inventory,
                                              _universe, _verdict)

LOAN = "LoanExtract One - OMNI 2026_09_01.xlsx"
CASH = "Principal And Interest - OMNI 2026_09_01.xlsx"
#: A value that must never be echoed: the report is shareable only if no cell
#: of any file reaches it.
SECRET = "76034101"


def _write(tmp_path, name, doc):
    (tmp_path / name).write_text(json.dumps(doc), encoding="utf-8")
    return tmp_path


class TestItReadsTheArtefactsItIsGiven:

    def test_the_inventory_names_each_file_and_what_it_was_read_as(
            self, tmp_path):
        _write(tmp_path, "01_file_inventory.json", [
            {"file_name": LOAN, "classification": "current_loan_report",
             "confidence": 0.67, "sheet_name": "Sheet1", "row_count": 568,
             "column_count": 41, "columns": ["Loan Policy Number", "Balance"]}])
        said = "\n".join(_inventory(tmp_path, True))
        assert LOAN in said
        assert "current_loan_report" in said
        assert "568 x 41" in said
        assert "Loan Policy Number" in said

    def test_headers_can_be_withheld(self, tmp_path):
        _write(tmp_path, "01_file_inventory.json", [
            {"file_name": LOAN, "columns": ["Loan Policy Number"]}])
        assert "Loan Policy Number" not in "\n".join(_inventory(tmp_path, False))

    def test_an_empty_inventory_says_no_file_was_read(self, tmp_path):
        _write(tmp_path, "01_file_inventory.json", [])
        assert "NOTHING" in "\n".join(_inventory(tmp_path, True))

    def test_eligibility_names_the_period_and_whether_it_is_the_listing(
            self, tmp_path):
        _write(tmp_path, "04c_source_period_eligibility.json", {"rows": [
            {"source_file": LOAN, "output_domain": "central_lender_tape",
             "artefact_role": "current_loan_report",
             "inferred_reporting_period": "2026-08",
             "run_reporting_period": "2026-08",
             "eligibility_basis": "filename_date", "confidence": 0.6,
             "is_period_eligible": True, "is_universe_source": True},
            {"source_file": CASH, "output_domain": "pipeline_mi"}]})
        said = "\n".join(_eligibility(tmp_path))
        assert "2026-08" in said and "filename_date" in said
        assert "IS LOAN LISTING : True" in said
        # The pipeline row belongs to another cadence and is not the question.
        assert CASH not in said

    def test_a_file_set_aside_says_why(self, tmp_path):
        _write(tmp_path, "04c_source_period_eligibility.json", {"rows": [
            {"source_file": LOAN, "output_domain": "central_lender_tape",
             "inferred_reporting_period": "2026-09",
             "run_reporting_period": "2026-08", "is_period_eligible": False,
             "reason_excluded": "future_period"}]})
        assert "future_period" in "\n".join(_eligibility(tmp_path))


class TestItReportsTheBuildAndItsVerdict:

    def test_it_names_the_chosen_listing_its_key_and_its_counts(self):
        said = "\n".join(_universe({"lender_summary": {"universe_debug": {
            "run_reporting_period": "2026-08", "period_gate_active": True,
            "selected_universe_source_file": LOAN,
            "selected_universe_key_column": "Loan Policy Number",
            "raw_universe_rows": 568, "canonical_universe_rows": 568,
            "universe_roles": ["current_loan_report"]}}}))
        assert LOAN in said and "Loan Policy Number" in said
        assert "568" in said and "current_loan_report" in said

    def test_a_file_opened_and_a_file_set_aside_are_both_listed(self):
        said = "\n".join(_universe({"lender_summary": {"universe_debug": {
            "considered_sources": [
                {"source_file": LOAN, "artefact_role": "current_loan_report",
                 "key_column": "Loan Policy Number", "key_count": 568,
                 "file_in_inventory": True, "frame_loaded": True}],
            "excluded_sources": [
                {"source_file": CASH, "reason": "future_period",
                 "inferred_reporting_period": "2026-09", "row_count": 12}]}}}))
        assert "FILES OPENED BY THE BUILD" in said and LOAN in said
        assert "FILES SET ASIDE" in said and CASH in said
        assert "future_period" in said

    def test_a_build_that_recorded_nothing_says_nothing_rather_than_guessing(
            self):
        assert "NOTHING" in "\n".join(_universe({}))

    def test_a_tape_with_loans_on_it_is_reported_without_an_explanation(self):
        said = "\n".join(_verdict({"loan_count": 568,
                                   "central_lender_tape_path": "/x/18.csv"}))
        assert "568" in said
        assert "WHY IT IS EMPTY" not in said

    def test_an_empty_tape_carries_the_sentence_the_operator_would_see(self):
        said = "\n".join(_verdict({"loan_count": 0, "lender_summary": {
            "universe_debug": {"selected_universe_source_file": "",
                               "excluded_sources": [], "considered_sources": [],
                               "pack_files": []}}}))
        assert "WHY IT IS EMPTY" in said
        assert "No file in this delivery was opened" in said


class TestNothingFromInsideAFileReachesTheReport:

    def test_no_cell_value_is_echoed_from_any_section(self, tmp_path):
        """Every artefact carries values somewhere. None may be printed."""
        _write(tmp_path, "01_file_inventory.json", [
            {"file_name": LOAN, "classification": "current_loan_report",
             "columns": ["Loan Policy Number"],
             "sample_values": [SECRET], "first_row": {"id": SECRET}}])
        _write(tmp_path, "04c_source_period_eligibility.json", {"rows": [
            {"source_file": LOAN, "output_domain": "central_lender_tape",
             "is_universe_source": True, "sample": SECRET}]})
        said = "\n".join(_inventory(tmp_path, True) + _eligibility(tmp_path)
                         + _universe({"lender_summary": {"universe_debug": {
                             "selected_universe_source_file": LOAN,
                             "sample_loan_ids": [SECRET]}}}))
        assert SECRET not in said
