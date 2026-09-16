"""Every source column, and what became of it.

WHAT WAS INVISIBLE

``run.mapping_report`` has always held one row per source column — the column,
the canonical field the header mapper matched it to, the tier, the confidence,
the note. It travelled in the readiness package (``field_mapping_report``) and
in the review package, and it was typed in the frontend as
``mapping_report: Record<string, unknown>[]``. No screen rendered it, and the
mock returned an empty list.

So an operator saw only the columns the mapper could NOT settle, raised as
decisions. Every mapping it accepted on its own was invisible — and those are
the majority, and the ones nobody has checked. On a hundred-column tape that is
the difference between answering the questions asked and being able to see all
the answers.

WHY THE CLASSIFICATION IS SERVER-SIDE

"Accepted automatically" means the tier is trusted OR the confidence clears the
threshold — the same test ``execution`` applies when deciding whether to use a
mapping without asking anybody. Written again in TypeScript it becomes a screen
that can disagree with the engine about which mappings a human checked. The
constants are imported from the one place that defines them, and the test below
pins that: change the threshold and the classification moves with it.
"""

from __future__ import annotations

import pytest

from operations_control.occ_agent import mapping_view
from operations_control.occ_agent.execution import (
    LOW_CONFIDENCE,
    _TRUSTED_TIERS,
)

from .conftest import ACTOR, TENANT_A

OPENING = ("Onboard Northstar Lending. UK equity release. Monthly portfolio "
           "MI. Portfolio id direct_101.")


class Run:
    """The two fields the overview reads. Nothing else is needed."""

    def __init__(self, mapping_report, open_decisions=()):
        self.mapping_report = list(mapping_report)
        self.open_decisions = list(open_decisions)


def row(column, canonical="", tier="exact", confidence=1.0, note="",
        source_file="LoanExtract.xlsx", primary=True):
    return {"source_file": source_file, "source_column": column,
            "canonical_field": canonical, "tier": tier,
            "confidence": confidence, "note": note, "primary": primary}


class TestWhatBecameOfAColumn:
    def test_an_exact_match_was_accepted_without_asking(self):
        assert mapping_view.classify(
            row("loan_id", "loan_id", "exact", 1.0)) == mapping_view.ROW_AUTOMATIC

    def test_an_operator_confirmation_is_marked_as_theirs(self):
        """The one category a reader must be able to find: what a human said."""
        assert mapping_view.classify(
            row("Prop Val", "property_value", "operator_approved", 1.0)) \
            == mapping_view.ROW_CONFIRMED

    def test_a_column_nothing_matched_is_not_used(self):
        assert mapping_view.classify(
            row("Internal Ref", "", "unmapped", 0.0)) == mapping_view.ROW_UNUSED

    def test_a_low_confidence_guess_needs_a_person(self):
        assert mapping_view.classify(
            row("Val", "property_value", "fuzz_token_set", 0.62)) \
            == mapping_view.ROW_NEEDS_YOU

    def test_a_file_that_could_not_be_read_says_so(self):
        assert mapping_view.classify(
            row("", "", "unreadable", 0.0)) == mapping_view.ROW_UNREADABLE

    @pytest.mark.parametrize("tier", sorted(_TRUSTED_TIERS))
    def test_every_trusted_tier_is_automatic_however_low_its_score(self, tier):
        """Trust comes from the tier OR the score. The screen must not invent
        a third rule and disagree with the engine about what was checked."""
        assert mapping_view.classify(
            row("c", "loan_id", tier, 0.10)) == mapping_view.ROW_AUTOMATIC

    def test_the_threshold_is_the_engines_own(self):
        assert mapping_view.classify(
            row("c", "loan_id", "fuzz_ratio_norm", LOW_CONFIDENCE)) \
            == mapping_view.ROW_AUTOMATIC
        assert mapping_view.classify(
            row("c", "loan_id", "fuzz_ratio_norm", LOW_CONFIDENCE - 0.01)) \
            == mapping_view.ROW_NEEDS_YOU


class TestTheTableAReaderGets:
    def test_it_counts_what_is_actually_feeding_a_field(self):
        """A proposal waiting on an operator is not mapped. Counting it would
        make a blocked run read as more complete than a finished one."""
        overview = mapping_view.overview(Run([
            row("loan_id", "loan_id", "exact", 1.0),
            row("Prop Val", "property_value", "operator_approved", 1.0),
            row("Val", "valuation", "fuzz_token_set", 0.62),
            row("Internal Ref", "", "unmapped", 0.0),
        ]))
        assert overview["counts"]["columns"] == 4
        assert overview["counts"]["mapped"] == 2
        assert overview["counts"]["needs_you"] == 1
        assert overview["counts"]["unused"] == 1

    def test_the_rows_that_need_a_person_are_at_the_top(self):
        overview = mapping_view.overview(Run([
            row("aaa", "loan_id", "exact", 1.0),
            row("zzz", "valuation", "fuzz_token_set", 0.62),
        ]))
        assert [r["source_column"] for r in overview["rows"]] == ["zzz", "aaa"]

    def test_a_row_that_needs_you_carries_the_question_to_answer(self):
        """Otherwise the table says "needs you" and sends you hunting."""
        overview = mapping_view.overview(Run(
            [row("Val", "valuation", "fuzz_token_set", 0.62)],
            [{"decision_id": "map_val", "status": "open",
              "subject": {"source_column": "Val"}}]))
        assert overview["rows"][0]["decision_id"] == "map_val"

    def test_an_ambiguity_over_two_columns_reaches_both_rows(self):
        overview = mapping_view.overview(Run(
            [row("Val A", "valuation", "token_set", 0.7),
             row("Val B", "valuation", "token_set", 0.7)],
            [{"decision_id": "amb_valuation", "status": "open",
              "subject": {"source_columns": ["Val A", "Val B"]}}]))
        assert {r["decision_id"] for r in overview["rows"]} == {"amb_valuation"}

    def test_an_ambiguous_column_is_not_reported_as_settled(self):
        """The trap this table would otherwise fall into.

        Two columns claiming one canonical field is an ambiguity, and it is
        raised AFTER both rows are written to the report — at whatever tier
        they matched at, often exact or alias, because that is how both came to
        claim the same field. Reading the tier alone, the table would say
        "matched automatically" about a column the run is blocked on.
        """
        overview = mapping_view.overview(Run(
            [row("Current Balance", "current_principal_balance", "alias", 1.0),
             row("Principal Balance", "current_principal_balance", "alias", 1.0)],
            [{"decision_id": "amb_balance", "status": "open",
              "subject": {"source_columns": ["Current Balance",
                                             "Principal Balance"]}}]))
        assert {r["state"] for r in overview["rows"]} == {
            mapping_view.ROW_NEEDS_YOU}
        assert overview["counts"]["mapped"] == 0

    def test_a_settled_decision_is_not_offered_again(self):
        overview = mapping_view.overview(Run(
            [row("Val", "valuation", "fuzz_token_set", 0.62)],
            [{"decision_id": "map_val", "status": "resolved",
              "subject": {"source_column": "Val"}}]))
        assert overview["rows"][0]["decision_id"] == ""

    def test_a_tier_is_translated_out_of_the_mappers_vocabulary(self):
        """"fuzz_ratio_norm" tells an operator nothing about how firm it is."""
        overview = mapping_view.overview(Run([
            row("Val", "valuation", "fuzz_ratio_norm", 0.61)]))
        label = overview["rows"][0]["tier_label"]
        assert label and "fuzz" not in label

    def test_every_tier_the_mapper_can_return_has_words(self):
        """A tier with no label would render as a raw identifier."""
        from engine.gate_1_alignment import semantic_alignment  # noqa: F401
        for tier in ("exact", "normalized", "alias", "token_set",
                     "fuzz_token_set", "fuzz_ratio_norm", "unmapped", "empty",
                     "operator_approved", "unreadable"):
            assert tier in mapping_view.TIER_LABELS

    def test_the_files_are_listed_so_a_table_can_be_split_by_them(self):
        overview = mapping_view.overview(Run([
            row("a", "loan_id", "exact", 1.0, source_file="Loan.xlsx"),
            row("b", "property_value", "exact", 1.0,
                source_file="Property.xlsx", primary=False),
        ]))
        assert overview["files"] == [
            {"name": "Loan.xlsx", "primary": True, "columns": 1},
            {"name": "Property.xlsx", "primary": False, "columns": 1}]


class TestEveryFileInThePackIsReported:
    """A pack of three files used to produce one file's worth of rows.

    The canonical tape is built from the loan tape alone, so the mapping
    report only ever covered that file — and the property and cash-flow tapes
    appeared nowhere. That is a reporting gap, not a modelling one: the real
    orchestrator's ``_load_structured_dataframes`` loads every structured file
    in the inventory, and the central tape builder consolidates a loan-domain
    field "even when its authoritative source is the cashflow extract, because
    domain membership follows the canonical field, not the file".
    """

    PACK = [
        row("loan_id", "loan_id", "exact", 1.0, source_file="Loan.xlsx"),
        row("Val", "valuation", "fuzz_token_set", 0.62,
            source_file="Loan.xlsx"),
        row("property_value", "property_value", "exact", 1.0,
            source_file="Property.xlsx", primary=False),
        row("Prop Ref", "property_reference", "fuzz_ratio_norm", 0.58,
            source_file="Property.xlsx", primary=False),
        row("principal", "principal_amount", "exact", 1.0,
            source_file="PandI.xlsx", primary=False),
    ]

    def test_all_three_files_have_rows(self):
        overview = mapping_view.overview(Run(self.PACK))
        assert [f["name"] for f in overview["files"]] == [
            "Loan.xlsx", "PandI.xlsx", "Property.xlsx"]
        assert overview["counts"]["columns"] == 5

    def test_the_tape_the_canonical_file_is_built_from_comes_first(self):
        overview = mapping_view.overview(Run(self.PACK))
        assert overview["files"][0] == {"name": "Loan.xlsx", "primary": True,
                                        "columns": 2}

    def test_a_weak_match_outside_the_primary_tape_asks_nothing(self):
        """Calling it "Needs you" would point at a question that does not
        exist: no decision is raised for a file the tape is not built from."""
        overview = mapping_view.overview(Run(self.PACK))
        weak = next(r for r in overview["rows"] if r["source_column"] == "Prop Ref")
        assert weak["state"] == mapping_view.ROW_UNCHECKED
        assert weak["decision_id"] == ""

    def test_a_weak_match_in_the_primary_tape_still_needs_you(self):
        overview = mapping_view.overview(Run(self.PACK))
        weak = next(r for r in overview["rows"] if r["source_column"] == "Val")
        assert weak["state"] == mapping_view.ROW_NEEDS_YOU

    def test_the_two_are_counted_apart(self):
        counts = mapping_view.overview(Run(self.PACK))["counts"]
        assert counts["needs_you"] == 1
        assert counts["unchecked"] == 1

    def test_no_mapping_yet_is_an_empty_table_not_a_crash(self):
        overview = mapping_view.overview(Run([]))
        assert overview["rows"] == []
        assert overview["counts"]["columns"] == 0

    def test_a_confidence_that_is_not_a_number_does_not_break_the_table(self):
        overview = mapping_view.overview(Run([
            {"source_file": "f.xlsx", "source_column": "c",
             "canonical_field": "loan_id", "tier": "exact",
             "confidence": "high"}]))
        assert overview["rows"][0]["confidence"] is None


def _spec():
    from engine.orchestrator_agent.adapters import PortfolioSpec
    return PortfolioSpec(source_portfolio_id="direct_101", input="")


class TestTheAdapterReadsEveryFile:
    """Where the rows come from, exercised against the real header mapper."""

    def _adapters(self, service, tmp_path, files):
        from operations_control.occ_agent.execution import (
            SyntheticOnboardingAdapters,
        )
        agent_case = service.create_case(tenant=TENANT_A,
                                         initiating_user=ACTOR,
                                         instruction=OPENING)
        sandbox = service.store.case_dir(TENANT_A, agent_case.case_ref)
        paths = []
        for name, text in files:
            path = sandbox / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
            paths.append(path)
        return SyntheticOnboardingAdapters(
            artefact_paths=paths, policy=service.policy, sandbox=sandbox,
            case_id=agent_case.case_ref, tenant=TENANT_A), sandbox

    PACK = [
        ("loan_extract.csv", "loan_id,current_principal_balance\nL1,100\n"),
        ("property_extract.csv", "loan_id,property_value\nL1,250000\n"),
        ("principal_and_interest.csv",
         "loan_id,principal_amount,interest_amount\nL1,80,20\n"),
    ]

    def test_every_file_in_the_pack_contributes_rows(self, service, tmp_path):
        """The defect: three files produced one file's worth of rows."""
        adapters, sandbox = self._adapters(service, tmp_path, self.PACK)
        adapters.onboard(_spec(), sandbox / "work")
        files = {r["source_file"] for r in adapters.mapping_report}
        assert files == {"loan_extract.csv", "property_extract.csv",
                         "principal_and_interest.csv"}

    def test_the_secondary_columns_are_reported(self, service, tmp_path):
        adapters, sandbox = self._adapters(service, tmp_path, self.PACK)
        adapters.onboard(_spec(), sandbox / "work")
        columns = {(r["source_file"], r["source_column"])
                   for r in adapters.mapping_report}
        assert ("property_extract.csv", "property_value") in columns
        assert ("principal_and_interest.csv", "interest_amount") in columns

    def test_only_the_primary_tape_is_marked_primary(self, service, tmp_path):
        adapters, sandbox = self._adapters(service, tmp_path, self.PACK)
        adapters.onboard(_spec(), sandbox / "work")
        primary = {r["source_file"] for r in adapters.mapping_report
                   if r["primary"]}
        assert primary == {"loan_extract.csv"}

    def test_a_secondary_file_never_blocks_the_run(self, service, tmp_path):
        """A weak match in a file the canonical tape is not built from has no
        answer an operator could give, so it must not become a decision."""
        adapters, sandbox = self._adapters(service, tmp_path, [
            self.PACK[0],
            ("property_extract.csv", "loan_id,Prp Vl Amt Gbp\nL1,250000\n"),
        ])
        result = adapters.onboard(_spec(), sandbox / "work")
        assert result.ok is True
        assert result.blocking is False


class TestItReachesTheScreen:
    """The whole defect was that it did not."""

    def test_the_status_payload_carries_the_table(self, service):
        agent_case = service.create_case(tenant=TENANT_A,
                                         initiating_user=ACTOR,
                                         instruction=OPENING)
        status = service.status(agent_case)
        assert "mapping" in status
        assert status["mapping"]["rows"] == []
        assert status["mapping"]["counts"]["columns"] == 0

    def test_it_reports_what_the_run_recorded(self, service):
        agent_case = service.create_case(tenant=TENANT_A,
                                         initiating_user=ACTOR,
                                         instruction=OPENING)
        agent_case.run.mapping_report = [
            row("loan_id", "loan_id", "exact", 1.0),
            row("Val", "valuation", "fuzz_token_set", 0.62),
        ]
        mapping = service.status(agent_case)["mapping"]
        assert mapping["counts"] == {
            "needs_you": 1, "unreadable": 0, "unchecked": 0, "unused": 0,
            "confirmed": 0, "automatic": 1, "columns": 2, "mapped": 1}
