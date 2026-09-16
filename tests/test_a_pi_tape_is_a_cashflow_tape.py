"""A funder principal-and-interest tape is a cash-flow tape.

THE MIGRATION THAT STOPPED HALFWAY

``config/.../workflow_input_requirements.yaml`` retired two roles and said so
in its own words:

    Still recognised by the classifier so an existing delivery keeps its role,
    but no longer asked for separately. Both are captured by a tape above:
    property and valuation detail under the collateral tape, funder principal
    and interest under the cash-flow tape.

The catalogue was migrated. The classifier was not. So a real file named
``Principal And Interest - OMNI 2026_09_01.xlsx`` landed on ``funder_pi_extract``,
a role that is:

  * in NEITHER workflow's ``optional_roles``, so it satisfied nothing on the
    checklist and the "Cash-flow tape" box stayed empty beside it; and
  * absent from ``date_semantics.CASHFLOW_ROLES``, so it was excluded from
    ``FUNDED_BASIS_ROLES`` and did not even take the funded reporting-date
    basis its siblings in the same pack take.

It arrived, it was recorded, and it counted for nothing.

WHAT IS AND IS NOT CHANGED

Where a NEW file of that name lands. ``funder_pi_extract`` stays a role the
platform knows — its label, its aliases in ``occ_agent.input_roles``, and any
delivery already recorded under it are untouched, so an operator can still
assign it by hand and a historic delivery still reads correctly.
"""

from __future__ import annotations

import pytest

from apps.blob_trigger_app.file_roles import classify_pack

COLUMNS = ["loan_id", "principal_amount", "interest_amount", "payment_date"]


def role_of(filename: str, columns=None):
    return classify_pack([(filename, list(columns or COLUMNS))]).assignments[0]


class TestAPrincipalAndInterestTapeIsACashflowTape:
    @pytest.mark.parametrize("filename", [
        "Principal And Interest - OMNI 2026_09_01.xlsx",
        "Funder Principal Interest 2026_09.xlsx",
        "Funder P&I 2026_09.xlsx",
        "funder_pi_2026_09.csv",
    ])
    def test_it_is_a_cashflow_extract(self, filename):
        assert role_of(filename).assigned_role == "cashflow_extract"

    def test_it_now_satisfies_a_role_the_workflows_actually_ask_for(self):
        """The reason this matters, stated against the requirements file."""
        from operations_control.intake import load_requirements
        workflows = load_requirements()["workflows"]
        for name in ("mi", "mi_annex2"):
            optional = workflows[name]["optional_roles"]
            assert "cashflow_extract" in optional
            assert "funder_pi_extract" not in optional

    def test_it_now_shares_the_funded_reporting_date_basis(self):
        """It sits in the same monthly pack as the loan and property tapes."""
        from engine.onboarding_agent.date_semantics import (
            CASHFLOW_ROLES,
            FUNDED_BASIS_ROLES,
        )
        role = role_of("Principal And Interest - OMNI 2026_09_01.xlsx")
        assert role.assigned_role in CASHFLOW_ROLES
        assert role.assigned_role in FUNDED_BASIS_ROLES

    def test_a_plain_cashflow_tape_is_unchanged(self):
        assert role_of("Cashflow 2026_09.xlsx").assigned_role == "cashflow_extract"
        assert role_of("Cash Flow Extract.xlsx").assigned_role == "cashflow_extract"


class TestTheDistinctionsAroundItAreIntact:
    def test_a_loan_tape_that_mentions_principal_is_still_a_loan_tape(self):
        """``loan_extract`` is ordered first for exactly this reason."""
        assert role_of("Loan Principal And Interest Extract.xlsx"
                       ).assigned_role == "loan_extract"

    def test_the_rest_of_the_clients_pack_is_unchanged(self):
        result = classify_pack([
            ("LoanExtract One - OMNI 2026_09_01.xlsx",
             ["loan_id", "current_balance", "interest_rate"]),
            ("PropertyExtract - Omni 2026_09_01.xlsx",
             ["loan_id", "property_value", "postcode"]),
            ("Principal And Interest - OMNI 2026_09_01.xlsx", COLUMNS),
        ])
        by_name = {a.filename: a.assigned_role for a in result.assignments}
        assert by_name["LoanExtract One - OMNI 2026_09_01.xlsx"] == "loan_extract"
        assert by_name["PropertyExtract - Omni 2026_09_01.xlsx"] == "property_extract"
        assert by_name["Principal And Interest - OMNI 2026_09_01.xlsx"] \
            == "cashflow_extract"

    def test_the_retired_role_is_still_one_the_platform_knows(self):
        """An existing delivery recorded under it must still read correctly,
        and an operator must still be able to assign it by hand."""
        from operations_control.intake import load_requirements
        from operations_control.occ_agent.input_roles import artefact_vocabulary
        assert "funder_pi_extract" in load_requirements()["role_labels"]
        assert artefact_vocabulary().label("funder_pi_extract")
