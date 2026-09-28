"""ERE dates a funded pack for the month that has just closed.

    "The period is as stated: 2026_09_01 is August, 2026_08_01 is July."

Read verbatim, every file in that pack is a FUTURE period against the delivery
it belongs to. A future-period source is set aside — correctly, for a source
that really is from next month — so an August delivery of ERE's pack was
assessed as three files from September, the loan-tape universe came out empty,
and the run halted with "Trakt could not build the combined loan record".

``filename_delivery_offset_months`` exists for exactly this and its own comment
describes the case. It lived only in ``config/system/onboarding_agent.yaml``,
so the only way to state it for a lender whose packs close the prior month was
to restate it for every lender whose packs do not. A filing convention is a
fact about the client who files, so a client block now overlays the system one.

AND IT APPLIES TO THE DELIVERY, NOT TO ONE ROLE IN IT. Every file in ERE's pack
carries the same date. The cashflow and collateral extracts are ENRICHMENT
roles, so scoping this to the funded-book roles alone would have kept the loan
listing and still set aside the balances and the valuations — a loan tape with
no figures on it. Pipeline snapshots keep the system answer: one dated the
first was taken that day, and shifting it would move a real date backwards.
"""

from __future__ import annotations

import pandas as pd

from engine.onboarding_agent import source_period_eligibility as spe

#: One ERE delivery: the funded pack for August 2026, dated the first of
#: September, plus a pipeline file named the same way.
PACK = {
    "current_loan_report": "LoanExtract One - OMNI 2026_09_01.xlsx",
    "cashflow_report": "Principal And Interest - OMNI 2026_09_01.xlsx",
    "collateral_report": "PropertyExtract - Omni 2026_09_01.xlsx",
    "pipeline_report": "Pipeline - OMNI 2026_09_01.xlsx",
}
FUNDED_ROLES = ("current_loan_report", "cashflow_report", "collateral_report")

#: No period column, so the filename decides — which is ERE's real shape for
#: the cashflow and collateral extracts.
NO_PERIOD_COLUMN = pd.DataFrame({"Loan Policy Number": ["76034101"]})


def _period(role, client_id="", run_period="2026-08", run_year=2026):
    cfg = spe.load_config(client_id=client_id)
    return spe._infer_source_period(
        PACK[role], "x.xlsx", role, "", NO_PERIOD_COLUMN, cfg,
        run_year, run_period)["period"]


class TestEREsFundedPackReadsAsTheMonthItCloses:

    def test_every_funded_file_lands_on_the_month_that_closed(self):
        for role in FUNDED_ROLES:
            assert _period(role, "ERE") == "2026-08", role

    def test_the_balances_and_the_valuations_come_with_it(self):
        """The two enrichment roles, named because scoping this to the funded
        BOOK roles would have left the tape with no figures on it."""
        assert _period("cashflow_report", "ERE") == "2026-08"
        assert _period("collateral_report", "ERE") == "2026-08"

    def test_a_pipeline_snapshot_still_means_the_day_it_was_taken(self):
        assert _period("pipeline_report", "ERE") == "2026-09"

    def test_july_reads_from_an_august_filing(self):
        cfg = spe.load_config(client_id="ERE")
        got = spe._infer_source_period(
            "LoanExtract One - OMNI 2026_08_01.xlsx", "x.xlsx",
            "current_loan_report", "", NO_PERIOD_COLUMN, cfg, 2026, "2026-07")
        assert got["period"] == "2026-07"

    def test_a_year_boundary_goes_back_a_year(self):
        cfg = spe.load_config(client_id="ERE")
        got = spe._infer_source_period(
            "LoanExtract One - OMNI 2027_01_01.xlsx", "x.xlsx",
            "current_loan_report", "", NO_PERIOD_COLUMN, cfg, 2027, "2026-12")
        assert got["period"] == "2026-12"


class TestNoOtherClientIsAffected:

    def test_an_unnamed_client_keeps_the_system_answer(self):
        for role in PACK:
            assert _period(role) == "2026-09", role

    def test_a_client_with_no_configuration_of_its_own_keeps_it_too(self):
        for role in PACK:
            assert _period(role, "NOSUCHCLIENT") == "2026-09", role

    def test_the_system_default_is_still_nought(self):
        assert int(spe.load_config().get(
            "filename_delivery_offset_months", 0) or 0) == 0


class TestTheClientBlockOverlaysRatherThanReplaces:

    def test_a_client_keeps_every_system_answer_it_does_not_mention(self):
        system, ere = spe.load_config(), spe.load_config(client_id="ERE")
        for key in ("period_columns", "pipeline_roles", "enrichment_roles",
                    "funded_current_book_roles", "allow_unknown_period",
                    "output_domains"):
            assert ere[key] == system[key], key

    def test_and_states_the_one_thing_it_does(self):
        ere = spe.load_config(client_id="ERE")
        assert ere["funded_filename_delivery_offset_months"] == -1
        assert "funded_filename_delivery_offset_months" not in spe.load_config()


class TestTheIdentityTheLivePathActuallyCarries:
    """The overlay above was correct and unreachable.

    Onboarding's ``client_id`` argument is the SOURCE PORTFOLIO — the live
    adapter passes ``source_portfolio_id`` into it and puts the tenant in
    ``client_name``, and the orchestrator says so where it resolves mapping
    scope. So the client block was asked for under ``direct_001``, a file that
    does not exist, and ERE's delivery was read under the system convention it
    had just been excused from. The caller now passes every identity it holds.
    """

    def test_the_portfolio_id_alone_finds_nothing_and_that_is_the_bug(self):
        assert _period("current_loan_report", "direct_001") == "2026-09"

    def test_the_tenant_beside_the_portfolio_finds_the_client(self):
        assert _period("current_loan_report", ("ERE", "direct_001")) == "2026-08"

    def test_order_does_not_matter_when_only_one_of_them_is_a_client(self):
        assert _period("current_loan_report", ("direct_001", "ERE")) == "2026-08"

    def test_candidates_drop_blanks_and_duplicates_and_keep_order(self):
        assert spe.client_candidates(("ERE", "", None, "direct_001", "ERE")) == [
            "ERE", "direct_001"]
        assert spe.client_candidates("ERE") == ["ERE"]
        assert spe.client_candidates("") == []
        assert spe.client_candidates(()) == []

    def test_no_identity_at_all_is_still_the_system_answer(self):
        assert _period("current_loan_report", ("", "")) == "2026-09"

    def test_the_identity_survives_the_whole_resolution_not_just_the_lookup(
            self, tmp_path):
        """End to end through the call onboarding actually makes.

        The overlay being reachable from ``load_config`` proved nothing: the
        bug was a caller handing the wrong name down. This runs the resolution
        onboarding runs, under the two identities the live path carries, and
        reads the period off the artefact it writes.
        """
        src = tmp_path / PACK["current_loan_report"]
        pd.DataFrame({"Loan Policy Number": ["76034101"],
                      "Current Balance": [1.0]}).to_csv(src, index=False)
        inventory = [{"file_name": src.name, "file_path": str(src),
                      "file_type": "csv", "sheet_name": "",
                      "classification": "current_loan_report",
                      "detected_reporting_date": ""}]
        got = spe.resolve_and_write(
            inventory, "mi_2026_08", tmp_path / "out",
            input_dir=str(tmp_path), client_id=("ERE", "direct_001"))
        tape = [r for r in got["rows"]
                if r["output_domain"] == "central_lender_tape"]
        assert tape, got["rows"]
        assert tape[0]["inferred_reporting_period"] == "2026-08"
        assert tape[0]["is_period_eligible"] is True
        assert tape[0]["is_universe_source"] is True
