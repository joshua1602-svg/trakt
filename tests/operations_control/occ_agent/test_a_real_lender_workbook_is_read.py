"""A lender's extract is a workbook, not a table.

WHAT THIS CLIENT'S FILES ACTUALLY LOOK LIKE

    LoanExtract One - OMNI 2026_09_01.xlsx   2 sheets; sheet 1 is a 7-row
                                             summary, sheet 2 holds 569 loans.
                                             Header on row 2.
    PropertyExtract - Omni 2026_09_01.xlsx   Header on row 2.
    Principal And Interest - OMNI …          Header on row 1. Read correctly.

WHAT THE CASE RECORDED BEFORE THIS

    sample.files[0].headers  Unnamed: 0 … Unnamed: 4      (7 records)
    sample.files[1].headers  Unnamed: 0 … Unnamed: 70     (569 records)

The loan book — the file the whole canonical tape is built from — profiled as
its cover page. Everything downstream would have been built on five unnamed
columns: the sample registered with Client Onboarding, the header mapping, the
mapping table an operator checks, and the tape itself.

Two causes, both in one reader. ``pd.read_excel(path)`` takes the FIRST sheet
and assumes the header is row one.

NEITHER IS NEWS TO THIS PLATFORM, WHICH IS THE POINT

``onboarding_orchestrator._load_structured_dataframes`` applies
``redetect_header`` and names PropertyExtract in the comment while doing it.
``schema_fingerprint._fingerprint_excel`` enumerates ``xl.sheet_names`` and
re-detects per sheet. ``router.py`` re-detects. The OCC Agent's artefact path
did neither — one path lagging three, not a new capability.

WHY THE CHOSEN SHEET IS REPORTED

"The sheet with the most rows" is a rule, and a rule that picks can pick wrong.
The sheet name travels on the artefact and on every mapping row it produced, so
an operator who sees the loan tape mapped from "Summary" can say so. One who is
never told cannot.
"""

from __future__ import annotations

import pytest

pd = pytest.importorskip("pandas")

from operations_control.occ_agent import workbook          # noqa: E402

from .conftest import ACTOR, TENANT_A                      # noqa: E402

OPENING = ("Onboard Northstar Lending. UK equity release. Monthly portfolio "
           "MI. Portfolio id direct_101.")

#: The real column names, on the row below a title block.
LOAN_COLUMNS = ["loan_id", "current_principal_balance", "interest_rate"]


def write_loan_workbook(path, *, summary_first=True, header_row=1):
    """A two-sheet workbook shaped like the client's: a small summary tab and
    a loan tab whose header sits below a title row."""
    loans = pd.DataFrame(
        [[f"L{i}", 1000 + i, 0.05] for i in range(40)],
        columns=LOAN_COLUMNS)
    if header_row:
        # A title block above the header, so row 1 is not the column names.
        loans = pd.concat([
            pd.DataFrame([["ERE Funding — loan extract", None, None]],
                         columns=LOAN_COLUMNS),
            pd.DataFrame([LOAN_COLUMNS], columns=LOAN_COLUMNS),
            loans], ignore_index=True)
    summary = pd.DataFrame([[i, i * 2] for i in range(7)],
                           columns=["Metric", "Value"])
    with pd.ExcelWriter(path) as writer:
        order = [("Summary", summary), ("Loan Data", loans)]
        if not summary_first:
            order.reverse()
        for name, frame in order:
            frame.to_excel(writer, sheet_name=name, index=False,
                           header=not header_row)
    return path


class TestTheDataSheetIsFound:
    def test_the_loans_are_read_not_the_summary_tab(self, tmp_path):
        table = workbook.read_table(write_loan_workbook(tmp_path / "loan.xlsx"))
        assert table.sheet == "Loan Data"
        assert table.row_count == 40

    def test_the_real_column_names_are_recovered(self, tmp_path):
        """Not `Unnamed: 0 … Unnamed: 4`, which is what the case recorded."""
        table = workbook.read_table(write_loan_workbook(tmp_path / "loan.xlsx"))
        assert table.columns == LOAN_COLUMNS
        assert not any(c.startswith("Unnamed") for c in table.columns)

    def test_it_does_not_simply_take_the_last_sheet(self, tmp_path):
        """The rule is the most rows, not the position — a workbook that puts
        its summary last must read the same way."""
        table = workbook.read_table(
            write_loan_workbook(tmp_path / "loan.xlsx", summary_first=False))
        assert table.sheet == "Loan Data"
        assert table.row_count == 40

    def test_the_sheet_it_chose_is_reported(self, tmp_path):
        """A rule that picks can pick wrong, so it has to say what it picked."""
        table = workbook.read_table(write_loan_workbook(tmp_path / "loan.xlsx"))
        assert table.sheets == ["Summary", "Loan Data"]
        assert table.sheet in table.sheets

    def test_a_header_already_on_row_one_is_left_alone(self, tmp_path):
        path = tmp_path / "clean.xlsx"
        pd.DataFrame([["L1", 100]], columns=["loan_id", "balance"]).to_excel(
            path, index=False)
        table = workbook.read_table(path)
        assert table.columns == ["loan_id", "balance"]
        assert table.header_row == 0

    def test_a_csv_has_no_sheet_and_still_reads(self, tmp_path):
        path = tmp_path / "loans.csv"
        path.write_text("loan_id,balance\nL1,100\n", encoding="utf-8")
        table = workbook.read_table(path)
        assert table.sheet == ""
        assert table.columns == ["loan_id", "balance"]

    def test_an_unreadable_file_is_empty_rather_than_a_crash(self, tmp_path):
        path = tmp_path / "not_a_workbook.xlsx"
        path.write_bytes(b"this is not a workbook")
        table = workbook.read_table(path)
        assert table.frame is None
        assert table.columns == []


class TestTheCaseSeesTheRealColumns:
    """End to end: registering the file, which is where it went wrong."""

    def test_the_artefact_profiles_the_loan_sheet(self, service, tmp_path):
        agent_case = service.create_case(tenant=TENANT_A,
                                         initiating_user=ACTOR,
                                         instruction=OPENING)
        path = write_loan_workbook(tmp_path / "LoanExtract One - OMNI.xlsx")
        service.register_synthetic_artefact(
            agent_case, filename=path.name, data=path.read_bytes(),
            actor=ACTOR)
        artefact = agent_case.run.received_artefacts[0]
        assert artefact["columns"] == LOAN_COLUMNS
        assert artefact["row_count"] == 40
        assert artefact["source_sheet"] == "Loan Data"

    def test_the_sample_registered_with_onboarding_is_the_real_one(
            self, service, tmp_path):
        """`register_sample` answers the file format, the expected names and
        often the asset class. Fed `Unnamed: 0` it answers from nothing."""
        agent_case = service.create_case(tenant=TENANT_A,
                                         initiating_user=ACTOR,
                                         instruction=OPENING)
        path = write_loan_workbook(tmp_path / "LoanExtract One - OMNI.xlsx")
        service.register_synthetic_artefact(
            agent_case, filename=path.name, data=path.read_bytes(),
            actor=ACTOR)
        after = service.classify_artefacts(agent_case, actor=ACTOR)
        headers = after.case.answers["sample"]["files"][0]["headers"]
        assert headers == LOAN_COLUMNS

    def test_the_file_is_recognised_as_a_loan_tape(self, service, tmp_path):
        """Header evidence beats the filename, so real headers matter to the
        role as well as to the mapping."""
        agent_case = service.create_case(tenant=TENANT_A,
                                         initiating_user=ACTOR,
                                         instruction=OPENING)
        path = write_loan_workbook(tmp_path / "LoanExtract One - OMNI.xlsx")
        service.register_synthetic_artefact(
            agent_case, filename=path.name, data=path.read_bytes(),
            actor=ACTOR)
        after = service.classify_artefacts(agent_case, actor=ACTOR)
        assert after.run.received_artefacts[0]["artefact_type"] == "loan_extract"

    def test_the_mapping_row_says_which_sheet_it_read(self, service, tmp_path):
        from engine.orchestrator_agent.adapters import PortfolioSpec
        from operations_control.occ_agent.execution import (
            SyntheticOnboardingAdapters,
        )
        agent_case = service.create_case(tenant=TENANT_A,
                                         initiating_user=ACTOR,
                                         instruction=OPENING)
        sandbox = service.store.case_dir(TENANT_A, agent_case.case_ref)
        path = write_loan_workbook(sandbox / "LoanExtract One - OMNI.xlsx")
        adapters = SyntheticOnboardingAdapters(
            artefact_paths=[path], policy=service.policy, sandbox=sandbox,
            case_id=agent_case.case_ref, tenant=TENANT_A)
        adapters.onboard(PortfolioSpec(source_portfolio_id="direct_101",
                                       input=""), sandbox / "work")
        assert adapters.mapping_report
        assert {r["source_sheet"] for r in adapters.mapping_report} \
            == {"Loan Data"}
        assert {r["source_column"] for r in adapters.mapping_report} \
            == set(LOAN_COLUMNS)
