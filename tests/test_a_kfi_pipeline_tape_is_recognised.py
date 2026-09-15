"""A "KFI and Pipeline" file is a pipeline report, and is recognised as one.

THE FILE THAT CAME BACK UNKNOWN

    M2L KFI and Pipeline 2026_09_14_103458.xlsx   ->  fallback_unknown, 0.00

``pipeline_report`` matched ``["pipeline","report"]``, ``["origination",
"pipeline"]`` or ``["origination"]``. This client's pipeline tape says
"Pipeline" but never "report", so every one of them fell through to unknown at
zero confidence — below ``min_required_role_confidence``, which parks the file
for operator confirmation. At a three-to-four day cadence that is a confirmation
every few days, forever, and roughly ninety of them in the historical load.

WHY NOT JUST MATCH "pipeline"

Because the rule table says not to, and it is right:

    A bare "pipeline" is deliberately NOT a rule. The pipeline BOOK's own loan
    tape is often named exactly that, and matching it here would classify a
    book's primary extract as a supplementary report — the same word meaning
    two things, resolved in favour of the wrong one.

So the pair is matched instead. A KFI is a Key Facts Illustration — an offer
document, produced at origination — so "KFI" beside "Pipeline" names a pipeline
report and cannot name a funded loan tape. The bare word stays unmatched, and
the distinction the table was protecting is left intact.
"""

from __future__ import annotations

import pytest

from apps.blob_trigger_app.file_roles import classify_pack

#: Columns are incidental here — every assertion is about the FILENAME rule.
#: A pipeline tape's headers vary by lender and must not be what carries this.
COLUMNS = ["case_id", "application_status", "requested_amount"]


def role_of(filename: str, columns=None):
    result = classify_pack([(filename, list(columns or COLUMNS))])
    return result.assignments[0]


class TestTheClientsPipelineTapeIsRecognised:
    @pytest.mark.parametrize("filename", [
        "M2L KFI and Pipeline 2026_09_14_103458.xlsx",
        "M2L KFI and Pipeline 2026_05_01.xlsx",
        "kfi and pipeline.xlsx",
        "KFI_Pipeline_2026-09-14.xlsx",
    ])
    def test_it_is_a_pipeline_report(self, filename):
        assert role_of(filename).assigned_role == "pipeline_report"

    def test_it_clears_the_confidence_floor(self):
        """Below the floor it parks for confirmation, which was the whole cost."""
        from operations_control.intake import load_requirements
        floor = float(load_requirements().get(
            "min_required_role_confidence", 0.4))
        assert role_of(
            "M2L KFI and Pipeline 2026_09_14_103458.xlsx").confidence >= floor

    def test_the_basis_is_the_filename_not_a_guess(self):
        assert role_of("M2L KFI and Pipeline 2026_09_14_103458.xlsx"
                       ).role_basis == "filename_keyword"


class TestTheDistinctionTheTableProtectsIsIntact:
    """The reason a bare "pipeline" is not a rule, still holding."""

    def test_a_bare_pipeline_is_still_not_a_pipeline_report(self):
        """It is usually the pipeline BOOK's own loan tape."""
        assert role_of("Pipeline 2026_09_14.xlsx").assigned_role \
            != "pipeline_report"

    def test_a_pipeline_loan_tape_is_still_a_loan_tape(self):
        assert role_of("Pipeline Loan Extract 2026_09_14.xlsx"
                       ).assigned_role == "loan_extract"

    def test_kfi_alone_claims_nothing(self):
        """Only the pair is evidence. A KFI on its own is not a pipeline tape."""
        assert role_of("KFI 2026_09_14.xlsx").assigned_role != "pipeline_report"


class TestTheOtherTwoFilesAreUnchanged:
    """The pack this client actually sends, classified together."""

    def test_the_whole_pack_reads_as_expected(self):
        result = classify_pack([
            ("LoanExtract One - OMNI 2026_05_01.xlsx",
             ["loan_id", "current_balance", "interest_rate"]),
            ("PropertyExtract - Omni 2026_05_01.xlsx",
             ["loan_id", "property_value", "postcode"]),
            ("M2L KFI and Pipeline 2026_09_14_103458.xlsx", COLUMNS),
        ])
        by_name = {a.filename: a.assigned_role for a in result.assignments}
        assert by_name["LoanExtract One - OMNI 2026_05_01.xlsx"] == "loan_extract"
        assert by_name["PropertyExtract - Omni 2026_05_01.xlsx"] == "property_extract"
        assert by_name["M2L KFI and Pipeline 2026_09_14_103458.xlsx"] == "pipeline_report"
